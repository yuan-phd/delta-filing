#!/usr/bin/env python
"""Script 2 — chosen/rejected reward decomposition across checkpoints, on Mac (fp16/bf16).

Closes the #3 gap: the matrix never logged chosen/rejected rewards, so we could not tell
whether the DPO margin grows by chosen-logp rising or rejected-logp falling. This teacher-
forces the chosen and rejected completion of each of the 549 TRAIN pairs (the exact 85%
split training used, DATA_SEED=42) through every checkpoint and logs per-token length-
normalized log-probs, then decomposes the margin.

FAITHFUL METHOD:
  * pairs/split    : mirrors train_pref.build_pairs + split_pairs (same tokenization, same
                     system-prompt-by-category, same DATA_SEED) -> the identical 549 pairs.
  * teacher-forcing: pref_losses.compute_log_probs (completion-only, length-normalized) —
                     the SAME function training uses for reference log-probs.
  * reward         : reward = policy_logp - reference_logp (reference = SFT adapter).
                     margin = beta * (reward_chosen - reward_rejected),  beta = train_pref.BETA.
  * axis           : 'tool_calling' (category==tool_calling) vs 'style' (all other categories)
                     — the same binary split training uses to pick the system prompt.

  python mac_reward_decomposition.py --seed all       # all 8 seeds x 4 ckpts (hours)
  python mac_reward_decomposition.py --seed 0         # one seed (reuses cached SFT reference)
  python mac_reward_decomposition.py --analyze-only   # rebuild tables/figures from cached jsonl
Outputs -> analysis_ws/reward_decomposition/{sft.jsonl, seed{S}_{ckpt}.jsonl, summary.json}
           figures/reward_decomposition_*.png
"""
import argparse
import json
import os
import time

import torch

import train_pref as T
import pref_losses as PL

ROOT = os.path.dirname(os.path.abspath(__file__))
SFT_ADAPTER = T.TRACK1["sft_adapter"]
PREF_DATA = T.TRACK1["pref_data"]
OUT_DIR = os.path.join(ROOT, "analysis_ws", "reward_decomposition")
FIG_DIR = os.path.join(ROOT, "figures")
CKPTS = ["margin1", "margin3", "margin5", "final"]
RUN = "kaggle_result/final/seed{s}/adapters/pref/track1_dpo_seed{s}"
BETA = T.BETA


def ckpt_path(seed, ck):
    base = RUN.format(s=seed)
    return base if ck == "final" else f"{base}/ckpt_{ck}"


def seed_margin(seed, ck):
    m = json.load(open(f"{RUN.format(s=seed)}/checkpoints.json"))
    return {c["label"]: c["smoothed_margin"] for c in m["checkpoints"]}.get(ck)


def build_train_pairs(tok):
    """The exact 549 train pairs (mirrors train_pref.build_pairs/split_pairs) + category/axis."""
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
          f"(expect 549/98; header match: {len(train)==549 and len(valid)==98})")
    return train


def score_checkpoint(model, pairs, device, label, out_path):
    if os.path.exists(out_path) and len(open(out_path).readlines()) == len(pairs):
        print(f"  [{label}] cached -> {out_path} (skip)"); return
    t0 = time.time()
    with open(out_path, "w") as f:
        with torch.no_grad():
            for i, p in enumerate(pairs):
                cl = float(PL.compute_log_probs(model, p["chosen_ids"], p["chosen_start"], device))
                rl = float(PL.compute_log_probs(model, p["rejected_ids"], p["rejected_start"], device))
                f.write(json.dumps({"pair_id": p["pair_id"], "chosen_logp": cl, "rejected_logp": rl,
                                    "category": p["category"], "axis": p["axis"]}) + "\n")
                if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
                    dt = time.time() - t0
                    eta = dt / (i + 1) * (len(pairs) - i - 1)
                    print(f"    [{label}] {i+1}/{len(pairs)} pairs  {dt:.0f}s  ETA {eta:.0f}s", flush=True)
    print(f"  [{label}] wrote -> {out_path}")


# ---------------- analysis ----------------
def _load(tag):
    p = os.path.join(OUT_DIR, f"{tag}.jsonl")
    return {r["pair_id"]: r for r in (json.loads(l) for l in open(p))} if os.path.exists(p) else None


def analyze():
    import numpy as np
    sft = _load("sft")
    if sft is None:
        print("  analyze: no cached sft.jsonl — run a seed first."); return
    seeds = [s for s in range(8) if all(_load(f"seed{s}_{ck}") for ck in CKPTS)]
    if not seeds:
        print("  analyze: no seed fully cached yet."); return
    ids = sorted(sft)

    def stats(rows, key, axis=None):
        v = [rows[i][key] for i in ids if (axis is None or sft[i]["axis"] == axis)]
        return float(np.mean(v))

    summary = {"beta": BETA, "reference": "sft_baseline", "seeds": seeds, "by_seed": {}}
    print("\n" + "=" * 88)
    print("REWARD DECOMPOSITION  (reward = policy_logp - sft_logp; margin = beta*(r_chosen - r_rejected))")
    print(f"beta={BETA}  reference=SFT  seeds={seeds}\n")
    # aggregate over seeds: per checkpoint, mean reward_chosen / reward_rejected / margin (+ per axis)
    agg = {ck: {"r_chosen": [], "r_rejected": [], "margin": [],
                "style_margin": [], "tool_margin": [],
                "style_rc": [], "style_rr": [], "tool_rc": [], "tool_rr": []} for ck in CKPTS}
    for s in seeds:
        summary["by_seed"][s] = {}
        for ck in CKPTS:
            pol = _load(f"seed{s}_{ck}")
            rc = stats(pol, "chosen_logp") - stats(sft, "chosen_logp")
            rr = stats(pol, "rejected_logp") - stats(sft, "rejected_logp")
            margin = BETA * (rc - rr)
            d = {"reward_chosen": rc, "reward_rejected": rr, "margin": BETA * (rc - rr),
                 "mean_chosen_logp": stats(pol, "chosen_logp"), "mean_rejected_logp": stats(pol, "rejected_logp"),
                 "smoothed_margin_log": seed_margin(s, ck)}
            for ax, tag in [("style", "style"), ("tool_calling", "tool")]:
                arc = stats(pol, "chosen_logp", ax) - stats(sft, "chosen_logp", ax)
                arr = stats(pol, "rejected_logp", ax) - stats(sft, "rejected_logp", ax)
                d[f"{tag}_reward_chosen"] = arc; d[f"{tag}_reward_rejected"] = arr
                d[f"{tag}_margin"] = BETA * (arc - arr)
                agg[ck][f"{tag}_margin"].append(BETA * (arc - arr))
                agg[ck][f"{tag}_rc"].append(arc); agg[ck][f"{tag}_rr"].append(arr)
            agg[ck]["r_chosen"].append(rc); agg[ck]["r_rejected"].append(rr); agg[ck]["margin"].append(margin)
            summary["by_seed"][s][ck] = d

    print(f"  {'ckpt':<9}{'r_chosen':>10}{'r_rejected':>12}{'margin':>9}  driver")
    for ck in CKPTS:
        rc = float(np.mean(agg[ck]["r_chosen"])); rr = float(np.mean(agg[ck]["r_rejected"]))
        mg = float(np.mean(agg[ck]["margin"]))
        driver = ("rejected falls" if abs(rr) > abs(rc) else "chosen rises") if mg else "-"
        print(f"  {ck:<9}{rc:>+10.3f}{rr:>+12.3f}{mg:>+9.3f}  {driver} (|Δrej|={abs(rr):.3f} vs |Δcho|={abs(rc):.3f})")
    print("\n  By axis at FINAL (mean over seeds):")
    for tag, nm in [("style", "style"), ("tool", "tool_calling")]:
        rc = float(np.mean(agg["final"][f"{tag}_rc"])); rr = float(np.mean(agg["final"][f"{tag}_rr"]))
        mg = float(np.mean(agg["final"][f"{tag}_margin"]))
        print(f"    {nm:<13} margin={mg:+.3f}  (reward_chosen={rc:+.3f}  reward_rejected={rr:+.3f})")
    sm = float(np.mean(agg["final"]["style_margin"])); tm = float(np.mean(agg["final"]["tool_margin"]))
    print(f"    -> style axis {'DOMINATES' if sm>tm else 'does NOT dominate'} "
          f"(style margin {sm:+.3f} vs tool {tm:+.3f})")
    summary["aggregate_final"] = {"style_margin": sm, "tool_margin": tm}
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    print(f"  wrote {OUT_DIR}/summary.json")
    make_figures(summary, seeds, agg)


def make_figures(summary, seeds, agg):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    xs = list(range(len(CKPTS)))

    # Fig 1: chosen vs rejected REWARD trajectory (mean over seeds) — the displacement test
    fig, ax = plt.subplots(figsize=(7, 4.6))
    rc = [np.mean(agg[ck]["r_chosen"]) for ck in CKPTS]
    rr = [np.mean(agg[ck]["r_rejected"]) for ck in CKPTS]
    ax.plot(xs, rc, "o-", color="tab:green", label="reward_chosen (chosen logp - sft)")
    ax.plot(xs, rr, "o-", color="tab:red", label="reward_rejected (rejected logp - sft)")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set(xticks=xs, xlabel="checkpoint", ylabel="reward (policy - reference logp)",
           title="Margin decomposition: chosen↑ vs rejected↓ (mean over seeds)")
    ax.set_xticklabels(CKPTS); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/reward_decomposition_chosen_vs_rejected.png", dpi=130)
    plt.close(fig)

    # Fig 2: per-seed margin + per-axis margin trajectory
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for s in seeds:
        axes[0].plot(xs, [summary["by_seed"][s][ck]["margin"] for ck in CKPTS], "o-", alpha=0.6)
    axes[0].set(xticks=xs, title="reconstructed margin per seed", xlabel="checkpoint",
                ylabel="beta*(r_chosen - r_rejected)"); axes[0].set_xticklabels(CKPTS)
    for tag, nm, col in [("style", "style", "tab:blue"), ("tool", "tool_calling", "tab:orange")]:
        m = [np.mean(agg[ck][f"{tag}_margin"]) for ck in CKPTS]
        axes[1].plot(xs, m, "o-", color=col, label=nm)
    axes[1].set(xticks=xs, title="margin by axis (style vs tool_calling)", xlabel="checkpoint",
                ylabel="margin"); axes[1].set_xticklabels(CKPTS); axes[1].legend()
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/reward_decomposition_by_axis.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {FIG_DIR}/reward_decomposition_chosen_vs_rejected.png + _by_axis.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="all", help="0..7 or 'all'")
    ap.add_argument("--base-model", default=None, help="local Qwen3.5-4B dir (sets DELTA_MODEL_DIR)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--device", default=None, help="mps (default on M1) / cpu")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = list(range(8)) if args.seed == "all" else [int(args.seed)]

    if args.analyze_only:
        analyze(); return

    if args.base_model:
        os.environ["DELTA_MODEL_DIR"] = args.base_model
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"device={device} dtype={args.dtype}  seeds={seeds}")

    tok = T.build_tokenizer(T.TRACK1)
    pairs = build_train_pairs(tok)

    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    src, local_only = T.resolve_model_source(T.TRACK1["model_id"])
    print(f"loading base {src} ({args.dtype}, unquantized) ...")
    base = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=dtype, trust_remote_code=True, local_files_only=local_only).to(device)
    base.eval()
    pm = PeftModel.from_pretrained(base, SFT_ADAPTER, adapter_name="sft")
    pm.eval()
    loaded = {"sft": SFT_ADAPTER}

    def model_for(path, name):
        if name not in loaded:
            pm.load_adapter(path, adapter_name=name); loaded[name] = path
        pm.set_adapter(name); pm.eval()
        return pm

    # SFT reference (shared, cached once)
    model_for(SFT_ADAPTER, "sft")
    score_checkpoint(pm, pairs, device, "sft", os.path.join(OUT_DIR, "sft.jsonl"))
    # each seed x 4 margin checkpoints
    for s in seeds:
        for ck in CKPTS:
            path = ckpt_path(s, ck)
            assert os.path.isdir(path), f"missing adapter: {path}"
            model_for(path, f"seed{s}_{ck}")
            score_checkpoint(pm, pairs, device, f"seed{s}_{ck}",
                             os.path.join(OUT_DIR, f"seed{s}_{ck}.jsonl"))

    analyze()


if __name__ == "__main__":
    main()
