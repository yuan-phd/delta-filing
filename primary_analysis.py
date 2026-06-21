#!/usr/bin/env python
"""Pre-registered PRIMARY endpoint — Note 2 (FROZEN 2026-06-18).

Per-task SEMANTIC accuracy at the FINAL fixed step (180) vs baseline_v2,
TOST equivalence ±5 pts (90% CI), over 8 seeds. Equivalence on a task requires
ALL of:
  (1) >=7/8 individual seeds within +-5 pts of baseline_v2,
  (2) seed-level mean-diff 90% CI within +-5 pts  [t-CI, df=7],
  (3) item-level pooled 90% CI within +-5 pts     [bootstrap over items, seeds pooled].
Headline CLEAN-NULL iff all three included tasks pass; else TASK-SELECTIVE branch.
Aborted seeds are excluded (none here). Reads the reorganized per-seed result JSONs
(aggregate semantic_accuracy) + generations (per-item semantic_pass) + baseline_v2.
No GPU, no model, no re-scoring (uses the frozen scorer-v2 flags saved at eval time).

  python primary_analysis.py
"""
import glob
import json
import os
import numpy as np
from scipy import stats

ROOT = "kaggle_result/final"
BASE = "kaggle_result/A2/results"
BASE_GEN = "kaggle_result/A2/generations"
SEEDS = list(range(8))
TASKS = ["json", "router", "extraction"]
BOUND = 5.0           # +-5 pts equivalence margin (FROZEN)
T95_DF7 = stats.t.ppf(0.95, 7)   # one-sided 0.95 -> two-sided 90% CI
N_BOOT = 10000
BOOT_SEED = 12345
CEILING_FLAGGED = {"json"}   # baseline >0.98 (per Note 2; included with caveat)


def per_item_pass(gen_path):
    """item_id -> semantic_pass (0/1), from frozen scorer-v2 flags saved at eval."""
    out = {}
    for line in open(gen_path):
        r = json.loads(line)
        if r.get("_header"):
            continue
        out[r["item_id"]] = int(bool(r["semantic_pass"]))
    return out


def load():
    """task -> {baseline_acc, baseline_items{id:0/1}, seed_acc[8], seed_items[8]{id:0/1}}."""
    data = {}
    for task in TASKS:
        b = json.load(open(f"{BASE}/track1_baseline_v2_{task}.json"))
        bitems = per_item_pass(f"{BASE_GEN}/track1_baseline_v2_{task}.jsonl")
        seed_acc, seed_items = [], []
        for s in SEEDS:
            rj = json.load(open(
                f"{ROOT}/seed{s}/results/track1_dpo_final_seed{s}_{task}.json"))
            gi = per_item_pass(
                f"{ROOT}/seed{s}/generations/track1_dpo_final_seed{s}_{task}.jsonl")
            seed_acc.append(rj["semantic_accuracy"])
            seed_items.append(gi)
        data[task] = dict(baseline_acc=b["semantic_accuracy"], baseline_items=bitems,
                          seed_acc=seed_acc, seed_items=seed_items, n=b["n"])
    return data


def seed_level(seed_acc, base_acc):
    """Mean per-seed diff (pts) + 90% t-CI (df=7) + per-seed within-bound count."""
    diffs = (np.array(seed_acc) - base_acc) * 100.0
    mean = diffs.mean()
    sd = diffs.std(ddof=1)
    se = sd / np.sqrt(len(diffs))
    half = T95_DF7 * se
    lo, hi = mean - half, mean + half
    within = int(np.sum(np.abs(diffs) <= BOUND))
    # TOST two one-sided p-values (H0: |mean|>=BOUND)
    t_lower = (mean - (-BOUND)) / se        # vs -BOUND
    t_upper = (mean - (+BOUND)) / se        # vs +BOUND
    p_lower = stats.t.sf(t_lower, 7)        # P(T > t_lower); reject H0:mean<=-B if small
    p_upper = stats.t.cdf(t_upper, 7)       # P(T < t_upper); reject H0:mean>=+B if small
    tost_p = max(p_lower, p_upper)
    return dict(diffs=diffs.tolist(), mean=mean, sd=sd, lo=lo, hi=hi,
                within=within, tost_p=tost_p,
                ci_in_bound=(lo > -BOUND and hi < BOUND))


def item_level(seed_items, base_items):
    """Item-level pooled (seeds pooled) bootstrap 90% CI of mean diff (pts).
    Cluster bootstrap over the shared item set; baseline aligned by item_id."""
    ids = sorted(base_items)                      # shared 160-item suite
    b = np.array([base_items[i] for i in ids], float)            # (n,)
    M = np.array([[si[i] for i in ids] for si in seed_items], float)  # (8, n)
    n = len(ids)
    point = (M.mean(axis=0) - b).mean() * 100.0   # = mean_seed_acc - base_acc
    rng = np.random.default_rng(BOOT_SEED)
    per_item = M.mean(axis=0) - b                 # (n,) seed-averaged item diff
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = per_item[idx].mean(axis=1) * 100.0
    lo, hi = np.percentile(boot, [5, 95])
    return dict(point=point, lo=lo, hi=hi, ci_in_bound=(lo > -BOUND and hi < BOUND))


def main():
    data = load()
    lines = ["# PRIMARY ENDPOINT — semantic accuracy at FINAL (step 180) vs baseline_v2",
             "# TOST equivalence +-5 pts (90% CI). Note 2 FROZEN 2026-06-18. n=8 seeds, none aborted.",
             ""]
    verdicts = {}
    for task in TASKS:
        d = data[task]
        base = d["baseline_acc"] * 100
        sl = seed_level(d["seed_acc"], d["baseline_acc"])
        il = item_level(d["seed_items"], d["baseline_items"])
        seed_ok = sl["within"] >= 7
        passed = seed_ok and sl["ci_in_bound"] and il["ci_in_bound"]
        verdicts[task] = passed
        mean_final = np.mean(d["seed_acc"]) * 100
        caveat = "  [CEILING-FLAGGED: baseline>98, equivalence test detects DEGRADATION not improvement]" \
            if task in CEILING_FLAGGED else ""
        lines += [
            f"## {task}{caveat}",
            f"  baseline_v2 semantic acc : {base:6.2f} pts",
            f"  mean final (8 seeds)     : {mean_final:6.2f} pts   (per-seed: "
            + " ".join(f"{a*100:.1f}" for a in d['seed_acc']) + ")",
            f"  per-seed diff vs base    : " + " ".join(f"{x:+.1f}" for x in sl["diffs"]),
            f"  (1) seeds within +-5 pts : {sl['within']}/8   -> {'PASS' if seed_ok else 'FAIL'} (need >=7)",
            f"  (2) seed-level mean diff : {sl['mean']:+.2f} pts  90% t-CI[df=7] "
            f"[{sl['lo']:+.2f}, {sl['hi']:+.2f}]  -> {'within' if sl['ci_in_bound'] else 'OUTSIDE'} +-5"
            f"   (TOST p={sl['tost_p']:.3g})",
            f"  (3) item-level pooled    : {il['point']:+.2f} pts  90% boot-CI "
            f"[{il['lo']:+.2f}, {il['hi']:+.2f}]  -> {'within' if il['ci_in_bound'] else 'OUTSIDE'} +-5",
            f"  VERDICT: {'EQUIVALENT (preserved)' if passed else 'NOT EQUIVALENT (degraded beyond bound)'}",
            "",
        ]
    allpass = all(verdicts.values())
    fails = [t for t, ok in verdicts.items() if not ok]
    lines += ["=" * 70]
    if allpass:
        lines.append("HEADLINE: CLEAN-NULL — free-generation capability PRESERVED within +-5 pts "
                     "on all three tasks at the 180-step margin endpoint.")
    else:
        lines.append(f"HEADLINE: TASK-SELECTIVE — equivalence holds on {sorted(set(TASKS)-set(fails))}; "
                     f"FAILS (degrades beyond +-5 pts) on {fails}.")
        lines.append("Per the pre-registered falsifier, the bounded-null headline is DROPPED; "
                     "the task-selective branch governs.")
    out = "\n".join(lines)
    os.makedirs("analysis_ws", exist_ok=True)
    open("analysis_ws/PRIMARY_TOST.md", "w").write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
