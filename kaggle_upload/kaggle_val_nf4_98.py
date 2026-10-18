#!/usr/bin/env python
"""kaggle_val_nf4_98.py — nf4 recomputation of the HELD-OUT preference margin on
ALL 98 pairs, per-axis, across 8 seeds x 4 checkpoints.

Why
---
Pre-registered secondary #3 (Note 2) is "held-out preference margin on all 98
pairs, per-category (style vs tool_calling)" — but Note 2 fixed no backend. The
fp16_mac recomputation (mac_supplementary.py Task A, analysis_ws/supplementary/
heldout_*) gives, at final, style < tool per seed (e.g. seed0 style 2.50 vs tool
4.03) — i.e. tool > style, which is the basis for falsification #3 (style per-pair
dominance FALSIFIED). This script recomputes the SAME 98-pair per-axis margin under
nf4 (the training / reward-decomposition backend) to answer:

    Does tool > style hold on the 98 held-out pairs under nf4?
    If style > tool under nf4, falsification #3 is BACKEND-DEPENDENT.

Method (faithful; mirrors train_pref + kaggle_reward_decomposition_kernel)
  * pairs   : train_pref.build_pairs + split_pairs -> the 98 valid pairs (69 style,
              29 tool_calling), the exact split Task A used (DATA_SEED=42).
  * scoring : pref_losses.compute_log_probs (completion-only, length-normalized).
  * reward  : reward = policy_logp - sft_logp   (SFT reference on the SAME nf4 model).
  * margin  : beta * (reward_chosen - reward_rejected),  beta = train_pref.BETA (0.5).
  * quant   : bitsandbytes nf4 + double-quant + fp16 compute (== the Kaggle setup).
  * adapters: matrix_adapters/seed{s}/{final, ckpt_margin1, ckpt_margin3, ckpt_margin5}.

  nf4 (bitsandbytes) REQUIRES CUDA — run on the Kaggle T4. Auto-detects /kaggle/input
  and resolves every path by glob (same pattern as the reward-decomposition kernel).

Usage
  python kaggle_val_nf4_98.py                 # --seed all (default), 8 seeds x 4 ckpts
  python kaggle_val_nf4_98.py --seed 0        # one seed (SFT ref scored once)
Output -> /kaggle/working/val_nf4_98.json  (Mac fallback: analysis_ws/supplementary/)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# Kaggle bootstrap — MUST precede importing train_pref / pref_losses.
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "bitsandbytes", "peft", "accelerate"], check=False)
    _code = _glob_all("pref_losses.py")
    assert _code, "NOT RUN: upload the repo dataset (pref_losses.py not found under /kaggle/input)"
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
CKPTS = ["margin1", "margin3", "margin5", "final"]

# Mac (repo-relative) fallbacks; overridden by glob on Kaggle.
MAC_MATRIX_ADAPTERS = os.path.join(ROOT, "kaggle_upload", "matrix_adapters")
MAC_OUT_DIR = os.path.join(ROOT, "analysis_ws", "supplementary")

# Resolved at runtime.
SFT_ADAPTER = None
PREF_DATA = None
MATRIX_ADAPTERS = None
OUT_DIR = None
OUT_JSON = None


# --------------------------------------------------------------------------
# Path resolution — Kaggle (glob /kaggle/input) vs Mac (repo-relative)
# --------------------------------------------------------------------------
def resolve_env():
    """Set SFT_ADAPTER / PREF_DATA / MATRIX_ADAPTERS / OUT_JSON for this host.
    On Kaggle, matrix_adapters/ is the canonical per-seed adapter tree for ALL seeds
    (0-7) — the same source the reward-decomposition kernel uses — which fixes the
    seed 2-7 path issue where kaggle_result/final/ only had a subset."""
    global SFT_ADAPTER, PREF_DATA, MATRIX_ADAPTERS, OUT_DIR, OUT_JSON
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
        # matrix_adapters root: .../matrix_adapters/seed0/final/adapter_config.json -> up 3
        madir = _glob1("matrix_adapters/seed0/final/adapter_config.json")
        assert madir, ("NOT RUN: matrix_adapters/ not found — attach the code dataset "
                       "version that includes kaggle_upload/matrix_adapters/seed{0..7}/")
        MATRIX_ADAPTERS = os.path.dirname(os.path.dirname(os.path.dirname(madir)))
        OUT_DIR = "/kaggle/working"
    else:
        SFT_ADAPTER = T.TRACK1["sft_adapter"]
        PREF_DATA = T.TRACK1["pref_data"]
        MATRIX_ADAPTERS = MAC_MATRIX_ADAPTERS
        OUT_DIR = MAC_OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_JSON = os.path.join(OUT_DIR, "val_nf4_98.json")
    print(f"[paths] host={'KAGGLE' if ON_KAGGLE else 'MAC'}")
    print(f"[paths] SFT_ADAPTER     = {SFT_ADAPTER}")
    print(f"[paths] PREF_DATA       = {PREF_DATA}")
    print(f"[paths] MATRIX_ADAPTERS = {MATRIX_ADAPTERS}")
    print(f"[paths] OUT_JSON        = {OUT_JSON}")


def ckpt_dir(seed, ck):
    """Adapter dir for (seed, checkpoint). 'final' is the seed's top-level adapter;
    margin1/3/5 are ckpt_* subdirs (matches kaggle_reward_decomposition_kernel.ckpt_dir)."""
    base = os.path.join(MATRIX_ADAPTERS, f"seed{seed}")
    return os.path.join(base, "final") if ck == "final" else os.path.join(base, f"ckpt_{ck}")


def seed_margin(seed, ck):
    """Smoothed train margin recorded at this checkpoint (for reference/labels)."""
    cj = os.path.join(MATRIX_ADAPTERS, f"seed{seed}", "checkpoints.json")
    if not os.path.exists(cj):
        return None
    m = json.load(open(cj))
    return {c["label"]: c["smoothed_margin"] for c in m["checkpoints"]}.get(ck)


# ------------------------------------------------------------------ pairs
def build_valid_pairs(tok):
    """The 98 held-out pairs validate()/Task A use (mirrors build_pairs/split_pairs,
    DATA_SEED=42) with category/axis. Returns ALL of valid (not just the 20-subset)."""
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
    from collections import Counter
    ax = Counter(p["axis"] for p in valid)
    ok = len(train) == 549 and len(valid) == 98 and ax["style"] == 69 and ax["tool_calling"] == 29
    print(f"  built {len(pairs)} pairs -> train={len(train)} valid={len(valid)} "
          f"axis={dict(ax)} (expect 549/98, 69 style/29 tool: {ok})")
    return valid


# ------------------------------------------------------------------ scoring
def score(model, pairs, device):
    """Per-pair (chosen_logp, rejected_logp) via the training log-prob function."""
    out = {}
    with torch.no_grad():
        for i, p in enumerate(pairs):
            cl = float(PL.compute_log_probs(model, p["chosen_ids"], p["chosen_start"], device))
            rl = float(PL.compute_log_probs(model, p["rejected_ids"], p["rejected_start"], device))
            out[p["pair_id"]] = (cl, rl)
            if torch.cuda.is_available() and (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
    return out


def per_pair_and_axis(pairs, pol, sft):
    """Returns (per_pair rows, per_axis summary). Margin/reward are beta-scaled,
    reward = policy_logp - sft_logp."""
    rows = []
    agg = {"style": {"m": [], "rc": [], "rr": []},
           "tool_calling": {"m": [], "rc": [], "rr": []}}
    for p in pairs:
        pid = p["pair_id"]
        cl, rl = pol[pid]
        rc = cl - sft[pid][0]
        rr = rl - sft[pid][1]
        m = BETA * (rc - rr)
        rows.append({"pair_id": pid, "category": p["category"], "axis": p["axis"],
                     "chosen_logp": cl, "rejected_logp": rl,
                     "reward_chosen": rc, "reward_rejected": rr, "margin": m})
        a = agg[p["axis"]]
        a["m"].append(m); a["rc"].append(rc); a["rr"].append(rr)
    def mean(x):
        return sum(x) / len(x) if x else float("nan")
    summary = {ax: {"n": len(agg[ax]["m"]), "margin": mean(agg[ax]["m"]),
                    "reward_chosen": mean(agg[ax]["rc"]), "reward_rejected": mean(agg[ax]["rr"])}
               for ax in ("style", "tool_calling")}
    all_m = agg["style"]["m"] + agg["tool_calling"]["m"]
    summary["overall"] = {"n": len(all_m), "margin": mean(all_m)}
    summary["tool_gt_style"] = summary["tool_calling"]["margin"] > summary["style"]["margin"]
    return rows, summary


# ------------------------------------------------------------------ model loading
def load_nf4_model():
    """Base in nf4 (EXACTLY the training / reward-decomposition config) + SFT adapter."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    device = torch.device("cuda")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    src, local_only = T.resolve_model_source(T.TRACK1["model_id"])
    print(f"loading base {src} (4bit-nf4-doublequant-fp16compute) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        src, quantization_config=bnb, trust_remote_code=True, device_map={"": 0},
        cache_dir=T.HF_HUB_CACHE, local_files_only=local_only)
    base.eval()
    pm = PeftModel.from_pretrained(base, SFT_ADAPTER, adapter_name="sft")
    pm.eval()
    return pm, device


# ------------------------------------------------------------------ persistence
def load_out():
    if OUT_JSON and os.path.exists(OUT_JSON):
        try:
            return json.load(open(OUT_JSON))
        except Exception:
            pass
    return {"backend": "nf4", "quant": QUANT_TAG, "beta": BETA,
            "reference": f"sft_baseline ({SFT_ADAPTER})",
            "metric": "margin = beta*(reward_chosen - reward_rejected), reward = policy_logp - sft_logp",
            "prediction": "style margin > tool_calling margin (falsified under fp16_mac)",
            "n_pairs": 98, "checkpoints": CKPTS, "sft_logp": None, "by_seed": {}}


def save_out(out):
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"  wrote {os.path.abspath(OUT_JSON)}")


# ------------------------------------------------------------------ analysis print
def print_axis_table(out, seeds):
    print("\n" + "=" * 92)
    print("HELD-OUT MARGIN (98 pairs) PER AXIS — nf4 — "
          "does tool > style hold? (falsification #3 basis)")
    print(f"beta={BETA}  reference=SFT  backend=nf4\n")
    for ck in CKPTS:
        print(f"  [{ck}]  {'seed':>4} {'style_m':>9} {'tool_m':>9} {'tool>style?':>12}")
        sm_all, tm_all = [], []
        for s in seeds:
            d = out["by_seed"].get(str(s), {}).get(ck)
            if not d:
                continue
            sm = d["style"]["margin"]; tm = d["tool_calling"]["margin"]
            sm_all.append(sm); tm_all.append(tm)
            print(f"        {s:>4} {sm:>9.3f} {tm:>9.3f} {str(tm > sm):>12}")
        if sm_all:
            gsm = sum(sm_all) / len(sm_all); gtm = sum(tm_all) / len(tm_all)
            verdict = "tool > style" if gtm > gsm else "style > tool"
            print(f"        {'MEAN':>4} {gsm:>9.3f} {gtm:>9.3f} {verdict:>12}")
        print()
    # headline on the FINAL checkpoint
    fin = [out["by_seed"][str(s)]["final"] for s in seeds if out["by_seed"].get(str(s), {}).get("final")]
    if fin:
        gsm = sum(d["style"]["margin"] for d in fin) / len(fin)
        gtm = sum(d["tool_calling"]["margin"] for d in fin) / len(fin)
        print(f"  FINAL 8-seed mean: style={gsm:.3f}  tool={gtm:.3f}")
        if gtm > gsm:
            print("  => tool > style holds under nf4 (consistent with fp16_mac; falsification #3 stands).")
        else:
            print("  => style > tool under nf4 (OPPOSITE of fp16_mac) — falsification #3 is BACKEND-DEPENDENT.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="all", help="0..7 or 'all'")
    ap.add_argument("--base-model", default=None, help="local Qwen3.5-4B dir (sets DELTA_MODEL_DIR)")
    ap.add_argument("--kaggle", action="store_true", help="force Kaggle path resolution (auto-detected too)")
    args = ap.parse_args()
    resolve_env()
    seeds = list(range(8)) if args.seed == "all" else [int(args.seed)]

    if args.base_model:
        os.environ["DELTA_MODEL_DIR"] = args.base_model

    if not torch.cuda.is_available():
        print("NOT RUN: nf4 (bitsandbytes) requires CUDA and cannot run on MPS/CPU.\n"
              "         Run on the Kaggle T4 backend (same as training / reward decomposition).")
        sys.exit(2)

    print(f"backend=nf4  device=cuda  beta={BETA}  seeds={seeds}  checkpoints={CKPTS}")
    tok = T.build_tokenizer(T.TRACK1)
    pairs = build_valid_pairs(tok)

    pm, device = load_nf4_model()

    def use(path, name):
        if name != "sft" and name not in pm.peft_config:
            pm.load_adapter(path, adapter_name=name)
        pm.set_adapter(name)
        pm.eval()

    out = load_out()

    # SFT reference — scored ONCE on the SAME nf4 model object (adapter 'sft').
    print("scoring SFT reference on the nf4 base (once) ...", flush=True)
    use(SFT_ADAPTER, "sft")
    t0 = time.time()
    sft = score(pm, pairs, device)
    out["sft_logp"] = {str(pid): {"chosen_logp": c, "rejected_logp": r} for pid, (c, r) in sft.items()}
    save_out(out)
    print(f"  SFT reference scored ({time.time()-t0:.0f}s)")

    for s in seeds:
        out["by_seed"].setdefault(str(s), {})
        for ck in CKPTS:
            path = ckpt_dir(s, ck)
            assert os.path.isdir(path), f"missing adapter: {path}"
            name = f"seed{s}_{ck}"
            use(path, name)
            t0 = time.time()
            pol = score(pm, pairs, device)
            rows, summary = per_pair_and_axis(pairs, pol, sft)
            summary["smoothed_train_margin"] = seed_margin(s, ck)
            summary["adapter"] = path
            summary["pairs"] = rows
            out["by_seed"][str(s)][ck] = summary
            print(f"  seed{s} {ck}: style={summary['style']['margin']:.3f} "
                  f"tool={summary['tool_calling']['margin']:.3f} "
                  f"(tool>style={summary['tool_gt_style']})  [{time.time()-t0:.0f}s]", flush=True)
            # free the DPO adapter (keep 'sft' resident) to bound memory
            pm.delete_adapter(name)
            save_out(out)  # incremental

    # cross-seed aggregate per checkpoint per axis
    agg = {}
    for ck in CKPTS:
        recs = [out["by_seed"][str(s)][ck] for s in seeds if out["by_seed"].get(str(s), {}).get(ck)]
        if not recs:
            continue
        def gm(ax, key):
            return sum(r[ax][key] for r in recs) / len(recs)
        agg[ck] = {"style_margin": gm("style", "margin"), "tool_margin": gm("tool_calling", "margin"),
                   "style_reward_chosen": gm("style", "reward_chosen"),
                   "style_reward_rejected": gm("style", "reward_rejected"),
                   "tool_reward_chosen": gm("tool_calling", "reward_chosen"),
                   "tool_reward_rejected": gm("tool_calling", "reward_rejected"),
                   "n_seeds": len(recs)}
        agg[ck]["tool_gt_style"] = agg[ck]["tool_margin"] > agg[ck]["style_margin"]
    out["aggregate_by_checkpoint"] = agg
    save_out(out)
    print_axis_table(out, seeds)


if __name__ == "__main__":
    main()
