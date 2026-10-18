#!/usr/bin/env python
"""Script 4 — two pre-registered secondaries in one Mac run (fp16, MPS by default).

Fills the last two unrun secondaries from notes/2026-06-18_note2_matrix_prereg_FROZEN.md:

  TASK A (#3) — Held-out preference margin on all 98 pairs, per-category (style vs tool_calling).
      The matrix logged only an aggregate val_margin over a 20-pair subset; the prereg asks for
      ALL 98 held-out pairs, split by axis. Teacher-force chosen+rejected for the 98 valid pairs
      through SFT + margin1/3/5/final, all 8 seeds; margin = beta*(reward_chosen - reward_rejected),
      reward = policy_logp - SFT_logp. Pre-registered prediction: style margin > tool_calling margin.

  TASK B (#8) — KL(policy || reference) per checkpoint.
      Proper token-level KL at gold positions (NOT the scalar logp ratio): for each extraction gold
      completion, KL = mean_pos Sigma_v p_policy(v)*(log p_policy(v) - log p_ref(v)), using the frozen
      probe primitives probe_register_shift.forward_full (full [n,V] log-softmax) + kl_over_positions.
      Matches the A1 probe operationalization (extraction, 160 items) but fp16 same-backend, all 8 seeds.

FAITHFUL METHOD — reuses the exact code the matrix/probe used:
  * A pairs/split  : train_pref.split_pairs (DATA_SEED, deterministic 85/15) -> the identical 98 valid
                     pairs; axis = "tool_calling" vs "style" (the binary split mac_reward_decomposition
                     uses). Teacher-forcing: pref_losses.compute_log_probs (completion-only, length-norm).
  * B KL          : probe_register_shift.forward_full + kl_over_positions, verbatim. SFT reference gold
                     distribution computed ONCE per item (160 forwards) and cached in-memory; policy
                     distribution recomputed per checkpoint. One item's [n,~248k] logits live at a time.

  python mac_supplementary.py --task both --seed all --base-model "$PWD/Qwen3.5-4B"
  python mac_supplementary.py --task A --seed 0          # just held-out margin, one seed
  python mac_supplementary.py --task B --analyze-only     # rebuild KL trajectory from cached jsonl
Outputs -> analysis_ws/supplementary/
             A: heldout_sft.jsonl, heldout_seed{S}_{ckpt}.jsonl, heldout_margin_per_category.json
             B: kl_seed{S}_{ckpt}.jsonl, kl_trajectory.json
Forwards: A ~6.5k (98x2x(1+8x4)) ; B ~5.28k (160 ref + 8x4x160 policy).
"""
import argparse
import gc
import json
import os
import time

import torch

import train_pref as T
import pref_losses as PL
import probe_register_shift as P

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "analysis_ws", "supplementary")
SFT_ADAPTER = T.TRACK1["sft_adapter"]                              # adapters/sft_pytorch
PREF_DATA = T.TRACK1["pref_data"]                                  # training_data/dpo_pytorch.jsonl
EXTRACTION_ITEMS = os.path.join(ROOT, "data_eval", "track1_extraction.jsonl")
RUN = "kaggle_result/final/seed{s}/adapters/pref/track1_dpo_seed{s}"   # final = top dir
CKPTS = ["margin1", "margin3", "margin5", "final"]                 # prereg margin checkpoint set
BETA = T.BETA
BACKEND = "fp16_mac"


def ckpt_path(seed, ck):
    base = RUN.format(s=seed)
    return base if ck == "final" else f"{base}/ckpt_{ck}"


def empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def load_base_pm(args, device, dtype):
    """fp16 base (unquantized) + PeftModel holding the SFT adapter; returns (base, pm, model_for)."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    src, local_only = T.resolve_model_source(T.TRACK1["model_id"])
    print(f"  loading base {src} ({args.dtype}, unquantized) ...")
    base = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=dtype, trust_remote_code=True, local_files_only=local_only).to(device)
    base.eval()
    pm = PeftModel.from_pretrained(base, SFT_ADAPTER, adapter_name="sft")
    pm.eval()
    loaded = {"sft": SFT_ADAPTER}

    def model_for(path, name):
        if name not in loaded:
            pm.load_adapter(path, adapter_name=name)
            loaded[name] = path
        pm.set_adapter(name); pm.eval()
        return pm

    return base, pm, model_for


# =====================================================================================
# TASK A — per-category held-out margin (#3)
# =====================================================================================
def build_valid_pairs(tok):
    """The exact 98 HELD-OUT pairs (mirrors mac_reward_decomposition.build_train_pairs but keeps
    `valid` instead of `train`), each carrying category + axis (tool_calling vs style)."""
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
    print(f"  built {len(pairs)} pairs -> train={len(train)} valid={len(valid)} "
          f"(expect 549/98; held-out match: {len(valid) == 98})")
    assert len(valid) == 98, f"expected 98 held-out pairs, got {len(valid)}"
    return valid


def score_pairs(model, pairs, device, label, out_path):
    """Per-pair {pair_id, chosen_logp, rejected_logp, category, axis} for one adapter; resumable."""
    if os.path.exists(out_path) and len([l for l in open(out_path) if l.strip()]) == len(pairs):
        print(f"  [{label}] cached -> {out_path} (skip)"); return
    t0 = time.time()
    with open(out_path, "w") as f:
        with torch.no_grad():
            for i, p in enumerate(pairs):
                cl = float(PL.compute_log_probs(model, p["chosen_ids"], p["chosen_start"], device))
                rl = float(PL.compute_log_probs(model, p["rejected_ids"], p["rejected_start"], device))
                f.write(json.dumps({"pair_id": p["pair_id"], "chosen_logp": cl, "rejected_logp": rl,
                                    "category": p["category"], "axis": p["axis"]}) + "\n")
                if (i + 1) % 25 == 0 or (i + 1) == len(pairs):
                    dt = time.time() - t0
                    eta = dt / (i + 1) * (len(pairs) - i - 1)
                    print(f"    [{label}] {i+1}/{len(pairs)} pairs  {dt:.0f}s  ETA {eta:.0f}s", flush=True)
    print(f"  [{label}] wrote -> {out_path}")


def _load_pairs_file(tag):
    p = os.path.join(OUT_DIR, f"heldout_{tag}.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return {r["pair_id"]: r for r in rows} if len(rows) == 98 else None


def analyze_A():
    import numpy as np
    sft = _load_pairs_file("sft")
    if sft is None:
        print("  analyze A: no complete heldout_sft.jsonl — run task A first."); return
    ids = sorted(sft)
    axis_of = {i: sft[i]["axis"] for i in ids}

    def mean_reward(pol, key, axis):
        v = [pol[i][key] - sft[i][key] for i in ids if axis_of[i] == axis]
        return float(np.mean(v)) if v else float("nan")

    seeds = [s for s in range(8) if all(_load_pairs_file(f"seed{s}_{ck}") for ck in CKPTS)]
    summary = {"backend": BACKEND, "beta": BETA, "reference": "sft_baseline",
               "metric": "margin = beta*(reward_chosen - reward_rejected), reward = policy_logp - sft_logp",
               "prediction": "style margin > tool_calling margin",
               "n_pairs": 98, "seeds": seeds, "by_seed": {}, "by_checkpoint": {}}
    print("\n" + "=" * 86)
    print("TASK A — HELD-OUT PREFERENCE MARGIN per category (style vs tool_calling)")
    print(f"backend={BACKEND}  beta={BETA}  ref=SFT  n_pairs=98  seeds={seeds}\n")

    agg = {ck: {"style": [], "tool_calling": []} for ck in CKPTS}
    for s in seeds:
        summary["by_seed"][s] = {}
        for ck in CKPTS:
            pol = _load_pairs_file(f"seed{s}_{ck}")
            d = {}
            for ax in ("style", "tool_calling"):
                rc = mean_reward(pol, "chosen_logp", ax)
                rr = mean_reward(pol, "rejected_logp", ax)
                m = BETA * (rc - rr)
                d[ax] = {"margin": m, "reward_chosen": rc, "reward_rejected": rr}
                agg[ck][ax].append(m)
            summary["by_seed"][s][ck] = d

    print(f"  {'ckpt':<9}{'style margin':>14}{'tool margin':>13}   prediction (style>tool)?")
    for ck in CKPTS:
        sm = float(np.mean(agg[ck]["style"])) if agg[ck]["style"] else float("nan")
        tm = float(np.mean(agg[ck]["tool_calling"])) if agg[ck]["tool_calling"] else float("nan")
        ok = "YES" if sm > tm else "no"
        summary["by_checkpoint"][ck] = {"style_margin": sm, "tool_margin": tm,
                                        "style_gt_tool": bool(sm > tm),
                                        "n_seeds": len(agg[ck]["style"])}
        print(f"  {ck:<9}{sm:>+14.3f}{tm:>+13.3f}   {ok}")
    held = [ck for ck in CKPTS if summary["by_checkpoint"][ck]["style_gt_tool"]]
    print(f"\n  pre-registered prediction holds at {len(held)}/{len(CKPTS)} checkpoints: {held}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "heldout_margin_per_category.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"  wrote {out}")


def run_task_A(args, device, dtype, seeds):
    print("\n##### TASK A — per-category held-out margin (#3) #####")
    tok = T.build_tokenizer(T.TRACK1)
    pairs = build_valid_pairs(tok)
    base, pm, model_for = load_base_pm(args, device, dtype)
    try:
        # SFT reference on the 98 pairs (computed/cached once)
        model_for(SFT_ADAPTER, "sft")
        score_pairs(pm, pairs, device, "sft", os.path.join(OUT_DIR, "heldout_sft.jsonl"))
        for s in seeds:
            for ck in CKPTS:
                path = ckpt_path(s, ck)
                assert os.path.isdir(path), f"missing adapter: {path}"
                model_for(path, f"seed{s}_{ck}")
                score_pairs(pm, pairs, device, f"seed{s}_{ck}",
                            os.path.join(OUT_DIR, f"heldout_seed{s}_{ck}.jsonl"))
    finally:
        del pm, base
        gc.collect(); empty_cache(device)
    analyze_A()


# =====================================================================================
# TASK B — KL(policy || reference) per checkpoint (#8)
# =====================================================================================
def _kl_complete(path):
    if not os.path.exists(path):
        return False
    return len([l for l in open(path) if l.strip()]) == 160


def build_extraction_renders(tok, items):
    """(item_id, ids, cstart) for each extraction GOLD completion — same render as the A1 probe."""
    out = []
    for it in items:
        g = it["gold"]
        ids, cstart = P.render_json(tok, it["prompt"], g["ticker"], g["section"], g["filing_date"])
        out.append((it["item_id"], ids, cstart))
    return out


def analyze_B():
    import numpy as np
    rows_by = {}
    for s in range(8):
        for ck in CKPTS:
            p = os.path.join(OUT_DIR, f"kl_seed{s}_{ck}.jsonl")
            if _kl_complete(p):
                rows_by[(s, ck)] = [json.loads(l) for l in open(p) if l.strip()]
    if not rows_by:
        print("  analyze B: no complete kl_seed*.jsonl — run task B first."); return
    summary = {"backend": BACKEND, "data": "extraction gold (160 items)",
               "metric": "mean_pos Sigma_v p_pol*(log p_pol - log p_ref); ref=SFT_baseline",
               "by_seed": {}, "trajectory": {}}
    print("\n" + "=" * 74)
    print("TASK B — KL(policy || reference) per checkpoint (extraction gold, 160 items)")
    print(f"backend={BACKEND}  ref=SFT_baseline\n")
    per_ck = {ck: [] for ck in CKPTS}
    for s in range(8):
        block = {}
        for ck in CKPTS:
            if (s, ck) in rows_by:
                m = float(np.mean([r["kl"] for r in rows_by[(s, ck)]]))
                block[ck] = m
                per_ck[ck].append(m)
        if block:
            summary["by_seed"][s] = block
    print(f"  {'ckpt':<9}{'mean KL':>12}{'std (8 seeds)':>16}{'n_seeds':>9}")
    for ck in CKPTS:
        if per_ck[ck]:
            mean = float(np.mean(per_ck[ck])); std = float(np.std(per_ck[ck]))
            summary["trajectory"][ck] = {"mean_kl": mean, "std_kl": std, "n_seeds": len(per_ck[ck])}
            print(f"  {ck:<9}{mean:>12.4f}{std:>16.4f}{len(per_ck[ck]):>9}")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "kl_trajectory.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"  wrote {out}")


def run_task_B(args, device, dtype, seeds):
    print("\n##### TASK B — KL(policy || reference) per checkpoint (#8) #####")
    # what work remains?
    todo = [(s, ck) for s in seeds for ck in CKPTS
            if not _kl_complete(os.path.join(OUT_DIR, f"kl_seed{s}_{ck}.jsonl"))]
    if not todo:
        print("  all requested seed/checkpoint KL files already complete (skip compute).")
        analyze_B(); return

    items = [json.loads(l) for l in open(EXTRACTION_ITEMS) if l.strip()]
    if args.n:
        items = items[:args.n]
    tok = T.build_tokenizer(T.TRACK1)
    renders = build_extraction_renders(tok, items)
    base, pm, model_for = load_base_pm(args, device, dtype)
    try:
        # --- SFT reference gold distribution: ONCE per item, cached in memory (fp16, CPU) ---
        # One [n,~248k] tensor at a time during this pass; we keep only the compact fp16 copy.
        print(f"  [ref] computing SFT gold distribution for {len(renders)} items (once) ...")
        model_for(SFT_ADAPTER, "sft")
        ref_dist = {}
        t0 = time.time()
        with torch.no_grad():
            for i, (iid, ids, cstart) in enumerate(renders):
                _, _, logsm, _ = P.forward_full(pm, ids, cstart, device)     # [n,V] fp32 on device
                ref_dist[iid] = logsm.to("cpu", dtype=torch.float16)         # compact cache
                del logsm
                if (i + 1) % 40 == 0 or (i + 1) == len(renders):
                    dt = time.time() - t0
                    print(f"    [ref] {i+1}/{len(renders)}  {dt:.0f}s", flush=True)
        empty_cache(device)

        # --- policy distribution per (seed, checkpoint); KL vs cached ref; one item at a time ---
        for s in seeds:
            for ck in CKPTS:
                out_path = os.path.join(OUT_DIR, f"kl_seed{s}_{ck}.jsonl")
                if _kl_complete(out_path):
                    print(f"  [seed{s}/{ck}] cached -> {out_path} (skip)"); continue
                path = ckpt_path(s, ck)
                assert os.path.isdir(path), f"missing adapter: {path}"
                model_for(path, f"seed{s}_{ck}")
                t0 = time.time()
                with open(out_path, "w") as f:
                    with torch.no_grad():
                        for i, (iid, ids, cstart) in enumerate(renders):
                            _, _, pol_logsm, _ = P.forward_full(pm, ids, cstart, device)   # [n,V] fp32
                            ref = ref_dist[iid].to(device, dtype=torch.float32)
                            kl = P.kl_over_positions(pol_logsm, ref)
                            f.write(json.dumps({"item_id": iid, "kl": kl}) + "\n")
                            del pol_logsm, ref                                    # discard logits now
                            if (i + 1) % 40 == 0 or (i + 1) == len(renders):
                                dt = time.time() - t0
                                eta = dt / (i + 1) * (len(renders) - i - 1)
                                print(f"    [seed{s}/{ck}] {i+1}/{len(renders)}  {dt:.0f}s  ETA {eta:.0f}s", flush=True)
                empty_cache(device)
                print(f"  [seed{s}/{ck}] wrote -> {out_path}")
    finally:
        del pm, base
        gc.collect(); empty_cache(device)
    analyze_B()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both", choices=["A", "B", "both"])
    ap.add_argument("--seed", default="all", help="0..7 or 'all'")
    ap.add_argument("--base-model", default=None, help="local Qwen3.5-4B dir (sets DELTA_MODEL_DIR)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--device", default="mps", help="mps (default) / cpu / cuda")
    ap.add_argument("--n", type=int, default=None, help="limit items/pairs (debug)")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = list(range(8)) if args.seed == "all" else [int(args.seed)]

    if args.analyze_only:
        if args.task in ("A", "both"):
            analyze_A()
        if args.task in ("B", "both"):
            analyze_B()
        return

    if args.base_model:
        os.environ["DELTA_MODEL_DIR"] = args.base_model
    req = args.device or os.environ.get("DELTA_DEVICE") or "mps"
    if req == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("  [warn] MPS unavailable -> falling back to CPU")
        req = "cpu"
    device = torch.device(req)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"device={device} dtype={args.dtype} backend={BACKEND}  task={args.task}  seeds={seeds}")

    if args.task in ("A", "both"):
        run_task_A(args, device, dtype, seeds)
    if args.task in ("B", "both"):
        run_task_B(args, device, dtype, seeds)


if __name__ == "__main__":
    main()
