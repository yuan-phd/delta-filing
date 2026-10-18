#!/usr/bin/env python
"""mac_val_nf4_check.py — DECISIVE BACKEND TEST for the val_margin anomaly.

Context
-------
`train_pref.validate()` logs a `val_margin` on the FIXED 20-pair held-out subset
(`valid_pairs[0:20]`, DATA_SEED=42) every `eval_every` steps. Those training-time
logs (nf4 backend) report val_margin ~5-7 at the last eval (step 150). But an
fp16 recomputation of the SAME 20 pairs at each seed's FINAL adapter gives only
~2.5-3.8 (seed0 = 3.245). Two hypotheses explain the ~2x gap:

  (A) BACKEND:  nf4 quantization inflates the margin. If so, recomputing the 20
                pairs under nf4 (same BitsAndBytesConfig as training) at the final
                adapter should land ~5.5-6.5 and reproduce the logged value.
  (B) LOGGING:  the training-time val path is anomalous (a bug in how it computes/
                accumulates the margin). If so, an nf4 recompute lands ~3.0-3.5,
                i.e. it agrees with fp16 and NOT with the training log.

This script recomputes the beta-scaled margin on those exact 20 pairs, under nf4,
at every seed's FINAL adapter, using the SAME functions training uses
(`pref_losses.compute_log_probs`, completion-only + length-normalized), and prints
the three-way comparison (nf4 recompute vs logged val_margin vs fp16 recompute).

FAITHFUL METHOD (mirrors train_pref / mac_reward_decomposition)
  * pairs   : train_pref.build_pairs + split_pairs -> valid[:VAL_SUBSET_N] (the 20).
  * scoring : pref_losses.compute_log_probs (completion-only, length-normalized).
  * reward  : reward = policy_logp - reference_logp   (reference = SFT adapter).
  * margin  : beta * (reward_chosen - reward_rejected),  beta = train_pref.BETA (0.5).
  * quant   : bitsandbytes nf4 + double-quant + fp16 compute (== the Kaggle setup).

  IMPORTANT: nf4 (bitsandbytes) REQUIRES CUDA. It does NOT run on MPS/CPU. Run this
  on the same T4 backend training used (Kaggle). On a Mac, use --compare-only to
  print the logged-vs-fp16 comparison (and any cached nf4 result) without a GPU, or
  --quant fp16 to reproduce the fp16 control on whatever device is available.

Kaggle vs Mac paths
  On Mac, paths are repo-relative (adapters/sft_pytorch, training_data/*.jsonl,
  kaggle_result/final/seed{s}/...). On Kaggle nothing is at those paths — code,
  adapters, data and model live under /kaggle/input/**. The script AUTO-DETECTS
  Kaggle (presence of /kaggle/input) and resolves every path by glob, exactly like
  the existing kernels (matrix_kernel.py etc.). Force it with --kaggle if needed.
  Attach as Kaggle input datasets: (1) the repo (pref_losses.py + train_pref.py),
  (2) the Qwen3.5-4B model, (3) the SFT adapter (sft_pytorch/), (4) dpo_pytorch.jsonl,
  (5) the trained final adapters (…/track1_dpo_seed{0..7}/). Optional for the
  comparison columns: the training margin logs and the fp16 heldout jsonls.

Usage
  # Kaggle (T4): auto-detects, resolves /kaggle/input, writes /kaggle/working
  python mac_val_nf4_check.py --seed all
  python mac_val_nf4_check.py --seed 0                    # one seed (SFT ref cached once)
  # Mac
  python mac_val_nf4_check.py --compare-only              # no model load; logged-vs-fp16
  python mac_val_nf4_check.py --quant fp16 --base-model Qwen3.5-4B --seed 0
Output -> <workdir>/val_nf4_check.json   (Mac: analysis_ws/supplementary/, Kaggle: /kaggle/working/)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# Kaggle bootstrap — MUST run before importing train_pref / pref_losses,
# because on Kaggle the code lives under /kaggle/input and is not yet on the path.
# --------------------------------------------------------------------------
ON_KAGGLE = os.path.isdir("/kaggle/input") or "--kaggle" in sys.argv


def _glob_all(pattern):
    return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))


def _glob1(pattern):
    hits = _glob_all(pattern)
    return hits[0] if hits else None


if ON_KAGGLE:
    os.environ.setdefault("DELTA_HF_CACHE", "/kaggle/working/hf_cache")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if "--compare-only" not in sys.argv:
        # nf4 needs bitsandbytes; mirror the established kernel convention (idempotent).
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "bitsandbytes", "peft", "accelerate"], check=False)
    _code = _glob_all("pref_losses.py")
    assert _code, "NOT RUN: upload the repo dataset (pref_losses.py not found under /kaggle/input)"
    # prefer a code dir that has BOTH modules we import (avoid stale partial copies)
    _both = [os.path.dirname(c) for c in _code
             if os.path.exists(os.path.join(os.path.dirname(c), "train_pref.py"))]
    _CODE_DIR = (_both or [os.path.dirname(_code[0])])[0]
    if _CODE_DIR not in sys.path:
        sys.path.insert(0, _CODE_DIR)
    print(f"[kaggle] code dir -> {_CODE_DIR}")

import torch  # noqa: E402

import train_pref as T  # noqa: E402
import pref_losses as PL  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
BETA = T.BETA
QUANT_TAG = "4bit-nf4-doublequant-fp16compute"

# Mac (repo-relative) defaults. On Kaggle these are all overridden by glob resolvers.
MAC_OUT_DIR = os.path.join(ROOT, "analysis_ws", "supplementary")
MAC_LOG_DIR = os.path.join(ROOT, "analysis_ws", "logs")
MAC_RUN = "kaggle_result/final/seed{s}/adapters/pref/track1_dpo_seed{s}"

# Resolved at runtime (set by resolve_env()).
SFT_ADAPTER = None
PREF_DATA = None
OUT_DIR = None
OUT_JSON = None


# --------------------------------------------------------------------------
# Path resolution — Kaggle (glob /kaggle/input) vs Mac (repo-relative)
# --------------------------------------------------------------------------
def resolve_env():
    """Set the global SFT_ADAPTER / PREF_DATA / OUT_DIR / OUT_JSON for this host,
    and print what was chosen so it can be eyeballed before a long run."""
    global SFT_ADAPTER, PREF_DATA, OUT_DIR, OUT_JSON
    if ON_KAGGLE:
        os.makedirs("/kaggle/working", exist_ok=True)
        try:
            os.chdir("/kaggle/working")
        except OSError:
            pass
        sft = _glob1("sft_pytorch/adapter_config.json")
        assert sft, "NOT RUN: SFT adapter (sft_pytorch/) not found under /kaggle/input"
        SFT_ADAPTER = os.path.dirname(sft)
        pref = _glob1("dpo_pytorch.jsonl")
        assert pref, "NOT RUN: dpo_pytorch.jsonl not found under /kaggle/input"
        PREF_DATA = pref
        OUT_DIR = "/kaggle/working"
    else:
        SFT_ADAPTER = T.TRACK1["sft_adapter"]
        PREF_DATA = T.TRACK1["pref_data"]
        OUT_DIR = MAC_OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_JSON = os.path.join(OUT_DIR, "val_nf4_check.json")
    print(f"[paths] host={'KAGGLE' if ON_KAGGLE else 'MAC'}")
    print(f"[paths] SFT_ADAPTER = {SFT_ADAPTER}")
    print(f"[paths] PREF_DATA   = {PREF_DATA}")
    print(f"[paths] OUT_JSON    = {OUT_JSON}")


def final_adapter_path(seed):
    """Path to seed {seed}'s FINAL adapter (step ~180), NOT a ckpt_* subdir.
    The final dir is uniquely marked by containing checkpoints.json."""
    if not ON_KAGGLE:
        return MAC_RUN.format(s=seed)
    hits = _glob_all(f"track1_dpo_seed{seed}/checkpoints.json")
    if not hits:  # fallback: adapter_config not under a ckpt_ subdir
        hits = [os.path.join(os.path.dirname(h), "checkpoints.json")
                for h in _glob_all(f"track1_dpo_seed{seed}/adapter_config.json")
                if "/ckpt_" not in h]
    assert hits, f"NOT RUN: final adapter track1_dpo_seed{seed} not found under /kaggle/input"
    # prefer a path under a 'final' directory if several datasets are attached
    preferred = [h for h in hits if "final" in h.lower()] or hits
    return os.path.dirname(preferred[0])


def log_path(seed):
    if ON_KAGGLE:
        return _glob1(f"track1_dpo_seed{seed}_margin.jsonl")
    p = os.path.join(MAC_LOG_DIR, f"track1_dpo_seed{seed}_margin.jsonl")
    return p if os.path.exists(p) else None


def heldout_final_path(seed):
    if ON_KAGGLE:
        return _glob1(f"heldout_seed{seed}_final.jsonl")
    p = os.path.join(MAC_OUT_DIR, f"heldout_seed{seed}_final.jsonl")
    return p if os.path.exists(p) else None


def heldout_sft_path():
    if ON_KAGGLE:
        return _glob1("heldout_sft.jsonl")
    p = os.path.join(MAC_OUT_DIR, "heldout_sft.jsonl")
    return p if os.path.exists(p) else None


# ------------------------------------------------------------------ pairs
def build_val_pairs(tok):
    """The exact 20 val-subset pairs validate() uses: first VAL_SUBSET_N of the
    held-out split (mirrors train_pref.build_pairs/split_pairs, DATA_SEED=42)."""
    raw = [json.loads(l) for l in open(PREF_DATA) if l.strip()]
    pairs = []
    for idx, ex in enumerate(raw):
        sp = T.system_prompt_for("track1", ex.get("category", "analysis"))
        prompt_msgs = [{"role": "system", "content": sp},
                       {"role": "user", "content": ex["question"]}]
        ctext = tok.apply_chat_template(prompt_msgs + [{"role": "assistant", "content": ex["chosen"]}], tokenize=False)
        rtext = tok.apply_chat_template(prompt_msgs + [{"role": "assistant", "content": ex["rejected"]}], tokenize=False)
        cids = tok.encode(ctext, max_length=T.MAX_SEQ_LEN, truncation=True)
        rids = tok.encode(rtext, max_length=T.MAX_SEQ_LEN, truncation=True)
        ptext = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        pids = tok.encode(ptext, max_length=T.MAX_SEQ_LEN, truncation=True)
        cs = T._completion_boundary(cids, pids)
        rs = T._completion_boundary(rids, pids)
        if cs >= len(cids) or rs >= len(rids):
            continue
        cat = ex.get("category", "analysis")
        pairs.append({"pair_id": idx, "category": cat,
                      "axis": "tool_calling" if cat == "tool_calling" else "style",
                      "chosen_ids": torch.tensor(cids, dtype=torch.long), "chosen_start": cs,
                      "rejected_ids": torch.tensor(rids, dtype=torch.long), "rejected_start": rs})
    train, valid = T.split_pairs(pairs)
    n = min(T.VAL_SUBSET_N, len(valid))
    val = valid[:n]
    ok = len(train) == 549 and len(valid) == 98
    print(f"  built {len(pairs)} pairs -> train={len(train)} valid={len(valid)} "
          f"(expect 549/98: {ok}) -> using first {n} val pairs")
    print(f"  val pair_ids: {[p['pair_id'] for p in val]}")
    return val


def reconstruct_val_ids():
    """The 20 val pair_ids WITHOUT loading a tokenizer/model — reuses the real
    split_pairs on raw-index placeholders. Valid because no pairs are skipped
    (647 -> 549/98); build_val_pairs asserts that alignment when a model is loaded."""
    raw = [json.loads(l) for l in open(PREF_DATA) if l.strip()]
    placeholders = [{"pair_id": i} for i in range(len(raw))]
    _, valid = T.split_pairs(placeholders)
    n = min(T.VAL_SUBSET_N, len(valid))
    return [p["pair_id"] for p in valid[:n]]


# ------------------------------------------------------------------ scoring
def score(model, pairs, device):
    """Per-pair (chosen_logp, rejected_logp) via the training log-prob function."""
    out = {}
    with torch.no_grad():
        for p in pairs:
            cl = float(PL.compute_log_probs(model, p["chosen_ids"], p["chosen_start"], device))
            rl = float(PL.compute_log_probs(model, p["rejected_ids"], p["rejected_start"], device))
            out[p["pair_id"]] = (cl, rl)
    return out


def per_pair_margins(pairs, pol, ref):
    """[{pair_id, category, axis, chosen_logp, rejected_logp,
         reward_chosen, reward_rejected, margin}], mean_margin."""
    rows, ms = [], []
    for p in pairs:
        pid = p["pair_id"]
        cl, rl = pol[pid]
        rc = cl - ref[pid][0]
        rr = rl - ref[pid][1]
        m = BETA * (rc - rr)
        ms.append(m)
        rows.append({"pair_id": pid, "category": p["category"], "axis": p["axis"],
                     "chosen_logp": cl, "rejected_logp": rl,
                     "reward_chosen": rc, "reward_rejected": rr, "margin": m})
    return rows, (sum(ms) / len(ms) if ms else float("nan"))


# ------------------------------------------------------------------ reference tables (existing files)
def logged_val_margin(seed):
    """Last logged training-time val_margin (nf4 backend) and its step."""
    f = log_path(seed)
    if not f or not os.path.exists(f):
        return None, None
    vals = [r for r in (json.loads(l) for l in open(f)) if "val_margin" in r]
    if not vals:
        return None, None
    return vals[-1]["val_margin"], vals[-1]["step"]


def fp16_val_margin(seed, val_ids):
    """fp16 recomputation of the same 20-pair mean margin at the final adapter,
    from the cached supplementary held-out log-probs."""
    fp, sf = heldout_final_path(seed), heldout_sft_path()
    if not (fp and sf and os.path.exists(fp) and os.path.exists(sf)):
        return None
    pol = {r["pair_id"]: r for r in (json.loads(l) for l in open(fp))}
    ref = {r["pair_id"]: r for r in (json.loads(l) for l in open(sf))}
    ms = []
    for i in val_ids:
        if i not in pol or i not in ref:
            return None
        rc = pol[i]["chosen_logp"] - ref[i]["chosen_logp"]
        rr = pol[i]["rejected_logp"] - ref[i]["rejected_logp"]
        ms.append(BETA * (rc - rr))
    return sum(ms) / len(ms)


# ------------------------------------------------------------------ model loading
def load_policy_model(quant, device):
    """Return a PeftModel with the SFT adapter loaded (adapter_name='sft').
    nf4 -> bitsandbytes 4-bit (CUDA only); fp16 -> unquantized on `device`.
    Base model source is auto-detected by train_pref.resolve_model_source (which
    checks /kaggle/input on Kaggle, DELTA_MODEL_DIR / HF hub otherwise)."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    src, local_only = T.resolve_model_source(T.TRACK1["model_id"])
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        print(f"loading base {src} (nf4 + double-quant + fp16 compute) ...")
        base = AutoModelForCausalLM.from_pretrained(
            src, quantization_config=bnb, torch_dtype=torch.float16,
            trust_remote_code=True, local_files_only=local_only, device_map={"": 0})
        # NB: do NOT .to(device) a 4-bit model — bitsandbytes places it via device_map.
    else:
        print(f"loading base {src} (fp16, unquantized) ...")
        base = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=torch.float16, trust_remote_code=True,
            local_files_only=local_only).to(device)
    base.eval()
    pm = PeftModel.from_pretrained(base, SFT_ADAPTER, adapter_name="sft")
    pm.eval()
    return pm


# ------------------------------------------------------------------ persistence
def load_out():
    if OUT_JSON and os.path.exists(OUT_JSON):
        try:
            return json.load(open(OUT_JSON))
        except Exception:
            pass
    return {"backend": None, "quant": None, "beta": BETA,
            "reference": f"sft_baseline ({SFT_ADAPTER})",
            "val_subset_n": T.VAL_SUBSET_N, "val_pair_ids": None,
            "sft_logp": None, "by_seed": {}}


def save_out(out):
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"  wrote {OUT_JSON}")


# ------------------------------------------------------------------ comparison / verdict
def print_comparison(out, seeds):
    val_ids = out.get("val_pair_ids") or []
    print("\n" + "=" * 96)
    print(f"VAL-SUBSET BACKEND CHECK  (20 pairs, beta={BETA}, reference=SFT, margin=beta*(r_chosen-r_rejected))")
    print(f"backend={out.get('backend')}  quant={out.get('quant')}\n")
    hdr = f"  {'seed':<5}{'nf4_mean':>10}{'logged_val':>12}{'(step)':>8}{'fp16_recompute':>16}{'nf4-fp16':>10}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    nf4_means, logged_means, fp16_means = [], [], []
    for s in seeds:
        rec = out["by_seed"].get(str(s), {})
        nf4 = rec.get("nf4_mean_margin")
        lv, lstep = logged_val_margin(s)
        fp = fp16_val_margin(s, val_ids) if val_ids else None
        if nf4 is not None:
            nf4_means.append(nf4)
        if lv is not None:
            logged_means.append(lv)
        if fp is not None:
            fp16_means.append(fp)
        def f(x, w=10, p=3):
            return (f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}")
        delta = (nf4 - fp) if (nf4 is not None and fp is not None) else None
        print(f"  {s:<5}{f(nf4)}{f(lv,12)}{('%d'%lstep) if lstep is not None else '--':>8}"
              f"{f(fp,16)}{f(delta) if delta is not None else '--':>10}")
    def gm(xs):
        return sum(xs) / len(xs) if xs else None
    gnf4, glog, gfp = gm(nf4_means), gm(logged_means), gm(fp16_means)
    print("  " + "-" * (len(hdr) - 2))
    def g(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "--"
    print(f"  {'MEAN':<5}{g(gnf4):>10}{g(glog):>12}{'':>8}{g(gfp):>16}")
    print(f"\n  8-seed grand mean:  nf4={g(gnf4)}   logged_val(last)={g(glog)}   fp16_recompute={g(gfp)}")

    # verdict on the nf4 grand mean (thresholds per the pre-stated decision rule)
    if gnf4 is not None:
        if 5.5 <= gnf4 <= 6.5:
            verdict = ("BACKEND CONFIRMED — nf4 reproduces the training-time val_margin; "
                       "the fp16 recompute understates the margin because of the quantization backend.")
        elif 3.0 <= gnf4 <= 3.5:
            verdict = ("BACKEND REFUTED — nf4 agrees with fp16 (~3), NOT with the training log; "
                       "the training-time val_margin logging path is the anomaly.")
        else:
            verdict = (f"INCONCLUSIVE — nf4 grand mean {gnf4:.3f} falls outside both pre-stated "
                       "bands (5.5-6.5 confirm / 3.0-3.5 refute). Inspect per-seed rows.")
        print(f"\n  VERDICT: {verdict}")
    else:
        print("\n  VERDICT: no nf4 results yet — run on a CUDA box (Kaggle T4) to score, "
              "or --compare-only just prints logged-vs-fp16.")
    # honest caveat: logged val is at the last eval step (typically 150), the nf4/fp16
    # recomputes are at the FINAL adapter (step ~180). Not the identical checkpoint.
    print("\n  NOTE: logged_val is the LAST training eval (step ~150); nf4_mean and fp16_recompute\n"
          "        are at the FINAL adapter (step ~180). No adapter was saved at the eval step, so\n"
          "        this is the closest like-for-like the saved checkpoints allow.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="all", help="0..7 or 'all'")
    ap.add_argument("--quant", default="nf4", choices=["nf4", "fp16"],
                    help="nf4 == Kaggle training backend (CUDA only); fp16 == unquantized control")
    ap.add_argument("--device", default="mps", help="mps (default) / cuda / cpu; nf4 forces cuda")
    ap.add_argument("--base-model", default=None, help="local Qwen3.5-4B dir (sets DELTA_MODEL_DIR)")
    ap.add_argument("--kaggle", action="store_true", help="force Kaggle path resolution (auto-detected too)")
    ap.add_argument("--compare-only", action="store_true",
                    help="print the comparison table from existing files (no model load; Mac ok)")
    args = ap.parse_args()
    resolve_env()
    seeds = list(range(8)) if args.seed == "all" else [int(args.seed)]

    if args.compare_only:
        out = load_out()
        if not out.get("val_pair_ids"):
            out["val_pair_ids"] = reconstruct_val_ids()  # no model needed
        print_comparison(out, list(range(8)))
        return

    if args.base_model:
        os.environ["DELTA_MODEL_DIR"] = args.base_model

    # nf4 needs CUDA — fail loudly rather than silently produce a different backend.
    if args.quant == "nf4":
        if not torch.cuda.is_available():
            print("NOT RUN: nf4 (bitsandbytes) requires CUDA and cannot run on MPS/CPU.\n"
                  "         Run this on the Kaggle T4 backend training used, or use\n"
                  "         --quant fp16 (control) / --compare-only (Mac, from cached files).")
            sys.exit(2)
        device = torch.device("cuda")
        if args.device != "cuda":
            print(f"[note] --device {args.device} ignored for nf4; using cuda (bitsandbytes requirement)")
    else:
        device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                              else "cpu")
    print(f"quant={args.quant}  device={device}  beta={BETA}  seeds={seeds}")

    tok = T.build_tokenizer(T.TRACK1)
    pairs = build_val_pairs(tok)
    val_ids = [p["pair_id"] for p in pairs]

    pm = load_policy_model(args.quant, device)

    def set_adapter(path, name):
        if name != "sft" and name not in pm.peft_config:
            pm.load_adapter(path, adapter_name=name)
        pm.set_adapter(name)
        pm.eval()

    out = load_out()
    quant_tag = QUANT_TAG if args.quant == "nf4" else "fp16-unquantized"
    if out.get("quant") not in (None, quant_tag):  # switching backend -> reset accumulated rows
        out["by_seed"] = {}
    out["backend"] = args.quant
    out["quant"] = quant_tag
    out["val_pair_ids"] = val_ids

    # SFT reference (scored once, backend-matched)
    print("scoring SFT reference (once) ...")
    set_adapter(SFT_ADAPTER, "sft")
    t0 = time.time()
    sft = score(pm, pairs, device)
    out["sft_logp"] = {str(pid): {"chosen_logp": c, "rejected_logp": r} for pid, (c, r) in sft.items()}
    print(f"  SFT reference scored ({time.time()-t0:.0f}s)")

    # each seed: FINAL adapter only
    for s in seeds:
        path = final_adapter_path(s)
        assert os.path.isdir(path), f"missing final adapter: {path}"
        print(f"  seed{s}: final adapter -> {path}")
        set_adapter(path, f"seed{s}")
        t0 = time.time()
        pol = score(pm, pairs, device)
        rows, mean_margin = per_pair_margins(pairs, pol, sft)
        out["by_seed"][str(s)] = {"final_adapter": path, "pairs": rows,
                                  "nf4_mean_margin": mean_margin}
        lv, lstep = logged_val_margin(s)
        fp = fp16_val_margin(s, val_ids)
        print(f"  seed{s}: {args.quant} mean margin = {mean_margin:.4f}  "
              f"(logged_val step{lstep}={lv if lv is None else round(lv,3)}  "
              f"fp16={fp if fp is None else round(fp,3)})  [{time.time()-t0:.0f}s]")
        save_out(out)  # incremental: safe for --seed N accumulation

    # grand mean over whatever seeds are now cached
    means = [out["by_seed"][k]["nf4_mean_margin"] for k in out["by_seed"]]
    out["grand_mean_margin"] = sum(means) / len(means) if means else None
    out["per_seed_mean"] = {k: out["by_seed"][k]["nf4_mean_margin"] for k in out["by_seed"]}
    save_out(out)
    print_comparison(out, sorted(int(k) for k in out["by_seed"]))


if __name__ == "__main__":
    main()
