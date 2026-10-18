#!/usr/bin/env python
"""kaggle_fp16_train_decomp.py — fp16 reward decomposition on the 549 TRAIN pairs
(+ the 20 val-subset), per-axis, across 8 seeds x 4 checkpoints. Kaggle T4.

Why
---
The nf4 reward decomposition (kaggle_reward_decomposition_kernel.py) found style-axis
margin dominance on the 549 train pairs. This is the fp16 (unquantized) counterpart,
so the train-set decomposition can be compared across backends. It ALSO scores the
same 20 val-subset pairs under fp16 per-axis, which — placed next to val_nf4_check.json
(nf4 val-20) — completes the controlled 2x2 for the style-vs-tool ordering:

                     |   train 549    |   val-20 subset
        -------------+----------------+-----------------
        nf4          |  reward-decomp |  val_nf4_check.json
        fp16         |  THIS (train)  |  THIS (val20)

Method (faithful; mirrors kaggle_reward_decomposition_kernel with fp16 base)
  * pairs   : train_pref.build_pairs + split_pairs -> 549 train (378 style/171 tool)
              and valid[:VAL_SUBSET_N] -> 20 val (15 style/5 tool). DATA_SEED=42.
  * scoring : pref_losses.compute_log_probs (completion-only, length-normalized).
  * reward  : reward = policy_logp - sft_logp   (SFT reference on the SAME fp16 base).
  * margin  : beta * (reward_chosen - reward_rejected),  beta = train_pref.BETA (0.5).
  * quant   : NONE — base loaded torch.float16, no BitsAndBytesConfig, .to('cuda').
  * adapters: matrix_adapters/seed{s}/{final, ckpt_margin1, ckpt_margin3, ckpt_margin5}.

Memory: runs sequentially AFTER kaggle_val_nf4_98.py in the same notebook, but as its
own process it loads a FRESH fp16 model (does NOT reuse the nf4 object). fp16 Qwen3.5-4B
is ~9 GB, fits the 16 GB T4. Needs CUDA.

Usage
  python kaggle_fp16_train_decomp.py                # --seed all (default); resumable
  python kaggle_fp16_train_decomp.py --seed 0       # one seed
Output -> /kaggle/working/fp16_train_decomposition.json  (Mac fallback: analysis_ws/supplementary/)
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
                    "peft", "accelerate"], check=False)   # no bitsandbytes needed for fp16
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
    matrix_adapters/seed{0..7}/ is the canonical per-seed adapter tree for ALL seeds."""
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
    OUT_JSON = os.path.join(OUT_DIR, "fp16_train_decomposition.json")
    print(f"[paths] host={'KAGGLE' if ON_KAGGLE else 'MAC'}")
    print(f"[paths] SFT_ADAPTER     = {SFT_ADAPTER}")
    print(f"[paths] PREF_DATA       = {PREF_DATA}")
    print(f"[paths] MATRIX_ADAPTERS = {MATRIX_ADAPTERS}")
    print(f"[paths] OUT_JSON        = {OUT_JSON}")


def ckpt_dir(seed, ck):
    """'final' is the seed's top-level adapter; margin1/3/5 are ckpt_* subdirs
    (matches kaggle_reward_decomposition_kernel.ckpt_dir)."""
    base = os.path.join(MATRIX_ADAPTERS, f"seed{seed}")
    return os.path.join(base, "final") if ck == "final" else os.path.join(base, f"ckpt_{ck}")


def seed_margin(seed, ck):
    cj = os.path.join(MATRIX_ADAPTERS, f"seed{seed}", "checkpoints.json")
    if not os.path.exists(cj):
        return None
    m = json.load(open(cj))
    return {c["label"]: c["smoothed_margin"] for c in m["checkpoints"]}.get(ck)


# ------------------------------------------------------------------ pairs
def build_pairs(tok):
    """Return (train_pairs[549], val20_pairs[20]) with category/axis, mirroring
    train_pref.build_pairs/split_pairs (DATA_SEED=42). val20 = first VAL_SUBSET_N of valid."""
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
    val20 = valid[:min(T.VAL_SUBSET_N, len(valid))]
    from collections import Counter
    tax, vax = Counter(p["axis"] for p in train), Counter(p["axis"] for p in val20)
    ok = (len(train) == 549 and len(valid) == 98 and tax["style"] == 378
          and tax["tool_calling"] == 171 and len(val20) == 20)
    print(f"  built {len(pairs)} pairs -> train={len(train)} {dict(tax)} | "
          f"val20={len(val20)} {dict(vax)} (expect 549[378/171] + 20[15/5]: {ok})")
    return train, val20


# ------------------------------------------------------------------ scoring
def score(model, pairs, device):
    out = {}
    with torch.no_grad():
        for i, p in enumerate(pairs):
            cl = float(PL.compute_log_probs(model, p["chosen_ids"], p["chosen_start"], device))
            rl = float(PL.compute_log_probs(model, p["rejected_ids"], p["rejected_start"], device))
            out[p["pair_id"]] = (cl, rl)
            if torch.cuda.is_available() and (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
    return out


def per_pair_and_axis(pairs, pol, sft):
    """(rows, summary). reward = policy_logp - sft_logp; margin = beta*(r_chosen - r_rejected)."""
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
def load_fp16_model():
    """Base UNQUANTIZED fp16 (.to cuda, no BitsAndBytesConfig) + SFT adapter. Fresh load."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    device = torch.device("cuda")
    src, local_only = T.resolve_model_source(T.TRACK1["model_id"])
    print(f"loading base {src} (fp16, UNQUANTIZED) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=torch.float16, trust_remote_code=True,
        cache_dir=T.HF_HUB_CACHE, local_files_only=local_only).to(device)
    base.eval()
    pm = PeftModel.from_pretrained(base, SFT_ADAPTER, adapter_name="sft")
    pm.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"  [mem] after base+SFT load: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    return pm, device


# ------------------------------------------------------------------ persistence
def load_out():
    if OUT_JSON and os.path.exists(OUT_JSON):
        try:
            return json.load(open(OUT_JSON))
        except Exception:
            pass
    return {"backend": "fp16", "quant": "fp16-unquantized", "beta": BETA,
            "reference": f"sft_baseline ({SFT_ADAPTER})",
            "metric": "margin = beta*(reward_chosen - reward_rejected), reward = policy_logp - sft_logp",
            "n_train": 549, "n_val_subset": 20, "checkpoints": CKPTS,
            "sft_logp": None, "by_seed": {}}


def save_out(out):
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"  wrote {os.path.abspath(OUT_JSON)}")


# ------------------------------------------------------------------ analysis print
def print_tables(out, seeds):
    for setname in ("train", "val20"):
        print("\n" + "=" * 92)
        print(f"fp16 REWARD DECOMPOSITION — {setname} — style vs tool_calling margin per checkpoint")
        for ck in CKPTS:
            recs = [out["by_seed"][str(s)][ck][setname] for s in seeds
                    if out["by_seed"].get(str(s), {}).get(ck)]
            if not recs:
                continue
            gsm = sum(r["style"]["margin"] for r in recs) / len(recs)
            gtm = sum(r["tool_calling"]["margin"] for r in recs) / len(recs)
            print(f"    {ck:<8} style={gsm:>7.3f}  tool={gtm:>7.3f}  "
                  f"-> {'tool > style' if gtm > gsm else 'style > tool'}  (n_seeds={len(recs)})")
    # 2x2 headline at FINAL
    fin = [out["by_seed"][str(s)]["final"] for s in seeds if out["by_seed"].get(str(s), {}).get("final")]
    if fin:
        print("\n  FINAL 8-seed mean (fp16):")
        for setname in ("train", "val20"):
            gsm = sum(d[setname]["style"]["margin"] for d in fin) / len(fin)
            gtm = sum(d[setname]["tool_calling"]["margin"] for d in fin) / len(fin)
            print(f"    {setname:<6}: style={gsm:.3f}  tool={gtm:.3f}  "
                  f"-> {'tool > style' if gtm > gsm else 'style > tool'}")
        print("  Compare vs nf4: reward_decomposition summary.json (train) and "
              "val_nf4_check.json (val20) to fill the 2x2.")


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
        print("NOT RUN: this scores on a GPU. Run on the Kaggle T4 (fp16 base fits ~9 GB).")
        sys.exit(2)

    out = load_out()

    # Resumable: figure out what still needs scoring before loading the model.
    todo = [(s, ck) for s in seeds for ck in CKPTS
            if not out["by_seed"].get(str(s), {}).get(ck)]
    have_sft = bool(out.get("sft_logp"))
    if not todo and have_sft:
        print("All requested (seed, checkpoint) cells already cached — nothing to score.")
        print_tables(out, seeds)
        return
    print(f"backend=fp16  device=cuda  beta={BETA}  seeds={seeds}  checkpoints={CKPTS}")
    print(f"  to score: {len(todo)} (seed,ckpt) cells; SFT reference "
          f"{'cached' if have_sft else 'to score'}")

    tok = T.build_tokenizer(T.TRACK1)
    train_pairs, val20_pairs = build_pairs(tok)
    all_pairs = train_pairs + val20_pairs  # disjoint; score both sets in one pass

    pm, device = load_fp16_model()  # fresh fp16 model (does NOT reuse any nf4 object)

    def use(path, name):
        if name != "sft" and name not in pm.peft_config:
            pm.load_adapter(path, adapter_name=name)
        pm.set_adapter(name)
        pm.eval()

    # SFT reference under fp16 — score once (or reuse cached fp16 values if resuming).
    if have_sft:
        sft = {int(pid): (v["chosen_logp"], v["rejected_logp"]) for pid, v in out["sft_logp"].items()}
        if not all(p["pair_id"] in sft for p in all_pairs):  # cache incomplete -> rescore
            have_sft = False
    if not have_sft:
        print("scoring SFT reference on the fp16 base (once) ...", flush=True)
        use(SFT_ADAPTER, "sft")
        t0 = time.time()
        sft = score(pm, all_pairs, device)
        out["sft_logp"] = {str(pid): {"chosen_logp": c, "rejected_logp": r} for pid, (c, r) in sft.items()}
        save_out(out)
        print(f"  SFT reference scored ({time.time()-t0:.0f}s)")

    for s in seeds:
        out["by_seed"].setdefault(str(s), {})
        for ck in CKPTS:
            if out["by_seed"][str(s)].get(ck):
                print(f"  seed{s} {ck}: cached -> skip")
                continue
            path = ckpt_dir(s, ck)
            assert os.path.isdir(path), f"missing adapter: {path}"
            name = f"seed{s}_{ck}"
            use(path, name)
            t0 = time.time()
            pol = score(pm, all_pairs, device)
            tr_rows, tr_sum = per_pair_and_axis(train_pairs, pol, sft)
            v_rows, v_sum = per_pair_and_axis(val20_pairs, pol, sft)
            out["by_seed"][str(s)][ck] = {
                "adapter": path, "smoothed_train_margin": seed_margin(s, ck),
                "train": {**tr_sum, "pairs": tr_rows},
                "val20": {**v_sum, "pairs": v_rows}}
            print(f"  seed{s} {ck}: TRAIN style={tr_sum['style']['margin']:.3f} "
                  f"tool={tr_sum['tool_calling']['margin']:.3f} | "
                  f"VAL20 style={v_sum['style']['margin']:.3f} "
                  f"tool={v_sum['tool_calling']['margin']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
            pm.delete_adapter(name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save_out(out)  # incremental / resumable

    # cross-seed aggregate per checkpoint per set per axis
    agg = {}
    for ck in CKPTS:
        recs = [out["by_seed"][str(s)][ck] for s in seeds if out["by_seed"].get(str(s), {}).get(ck)]
        if not recs:
            continue
        agg[ck] = {}
        for setname in ("train", "val20"):
            def gm(ax, key):
                return sum(r[setname][ax][key] for r in recs) / len(recs)
            agg[ck][setname] = {
                "style_margin": gm("style", "margin"), "tool_margin": gm("tool_calling", "margin"),
                "style_reward_chosen": gm("style", "reward_chosen"),
                "style_reward_rejected": gm("style", "reward_rejected"),
                "tool_reward_chosen": gm("tool_calling", "reward_chosen"),
                "tool_reward_rejected": gm("tool_calling", "reward_rejected"),
                "tool_gt_style": gm("tool_calling", "margin") > gm("style", "margin"),
                "n_seeds": len(recs)}
    out["aggregate_by_checkpoint"] = agg
    save_out(out)
    print_tables(out, seeds)


if __name__ == "__main__":
    main()
