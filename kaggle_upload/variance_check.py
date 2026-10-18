#!/usr/bin/env python
"""Seed-to-seed variance check for the 2-seed DPO control test.

Reports, for DPO seed0 vs seed1: per-task final-checkpoint accuracy spread AND the
final training margin reached. Two questions to answer: did margin reach ~5-6+, and
did seed-to-seed accuracy variance drop vs the old (batch=1) high-variance pair?

  python variance_check.py --results-dir results --logs-dir logs
  python variance_check.py --results-dir results --logs-dir logs --old-dir old/results
"""
import argparse
import json
import os

TASKS = ["json", "router", "extraction"]


def final_acc(results_dir, seed, task, cond="dpo"):
    p = os.path.join(results_dir, f"track1_{cond}_final_seed{seed}_{task}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("accuracy")


def final_margin(logs_dir, seed, cond="dpo"):
    """Last logged train_margin from the margin trajectory (= final margin)."""
    p = os.path.join(logs_dir, f"track1_{cond}_seed{seed}_margin.jsonl")
    if not os.path.exists(p):
        return None
    last = None
    for line in open(p):
        r = json.loads(line)
        if "train_margin" in r:
            last = r["train_margin"]
    return last


def spreads(results_dir):
    out = {}
    for t in TASKS:
        a0, a1 = final_acc(results_dir, 0, t), final_acc(results_dir, 1, t)
        out[t] = (a0, a1, (abs(a0 - a1) if (a0 is not None and a1 is not None) else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--old-dir", default=None, help="old (batch=1) results for comparison")
    ap.add_argument("--old-logs-dir", default=None)
    args = ap.parse_args()

    new = spreads(args.results_dir)
    old = spreads(args.old_dir) if args.old_dir else None

    # ---- final margins (did we reach the high-margin regime?) ----
    m0, m1 = final_margin(args.logs_dir, 0), final_margin(args.logs_dir, 1)
    print("=== final DPO margin reached ===")
    print(f"  seed0={('n/a' if m0 is None else f'{m0:.2f}')}  "
          f"seed1={('n/a' if m1 is None else f'{m1:.2f}')}")
    if args.old_logs_dir:
        om0 = final_margin(args.old_logs_dir, 0); om1 = final_margin(args.old_logs_dir, 1)
        print(f"  (old batch=1: seed0={om0}, seed1={om1})")
    margins = [m for m in (m0, m1) if m is not None]
    high_margin = bool(margins) and min(margins) >= 5.0
    print(f"  margin >= 5 on both seeds? {high_margin}\n")

    # ---- accuracy variance ----
    print(f"{'task':<12}{'seed0':<9}{'seed1':<9}{'|spread|':<10}" +
          ("  | old_spread" if old else ""))
    print("-" * (40 + (14 if old else 0)))
    new_spreads, old_spreads = [], []
    for t in TASKS:
        a0, a1, sp = new[t]
        line = (f"{t:<12}{('n/a' if a0 is None else f'{a0:.3f}'):<9}"
                f"{('n/a' if a1 is None else f'{a1:.3f}'):<9}"
                f"{('n/a' if sp is None else f'{sp:.3f}'):<10}")
        if sp is not None:
            new_spreads.append(sp)
        if old:
            osp = old[t][2]
            line += f"  | {('n/a' if osp is None else f'{osp:.3f}')}"
            if osp is not None:
                old_spreads.append(osp)
        print(line)

    if new_spreads:
        print(f"\nnew mean |spread| = {sum(new_spreads)/len(new_spreads):.3f}  "
              f"max = {max(new_spreads):.3f}")
    variance_dropped = None
    if old_spreads:
        print(f"old mean |spread| = {sum(old_spreads)/len(old_spreads):.3f}  "
              f"max = {max(old_spreads):.3f}")
        if new_spreads:
            variance_dropped = max(new_spreads) < max(old_spreads)

    # ---- combined verdict (the two things we need) ----
    print("\n=== VERDICT ===")
    print(f"  high margin (>=5 both seeds): {high_margin}")
    print(f"  variance dropped vs old:      "
          f"{'n/a (no --old-dir)' if variance_dropped is None else variance_dropped}")
    if high_margin and variance_dropped:
        print("  -> BOTH satisfied: proceed to the full matrix.")
    else:
        print("  -> NOT both satisfied: stop and pick another lever "
              "(epochs, more seeds, or report as a variance finding).")


if __name__ == "__main__":
    main()
