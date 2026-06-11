#!/usr/bin/env python
"""Baseline re-measurement (Item 3) — step-0 / untrained-SFT checkpoint evaluated on
the SAME 160-item suite, in the IDENTICAL generation pipeline as every other
checkpoint, scored with scorer v2 (three-tier). This becomes the on-disk reference
point for the equivalence test (replacing remembered baseline numbers).

Outputs per task: three-tier counts + semantic_accuracy (primary) + strict_accuracy
(secondary), plus every generation with its tier.

Requires a CUDA GPU (Kaggle). Stage only; run after quota reset.

  python run_baseline_v2.py --sft-adapter <path>
"""
import argparse
import json
import os

import eval_pref as E
import scorer_v2 as V

TRACK = "track1"
TASKS = ["json", "router", "extraction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()
    for d in ("generations", "results"):
        os.makedirs(d, exist_ok=True)

    session = E.EvalSession(TRACK)            # base loaded once; identical pipeline
    print("=" * 92)
    print(f"{'task':<12}{'n':<5}{'sem_acc':<9}{'strict_acc':<12}{'PASS':<6}{'SURF':<6}"
          f"{'SEM':<5}{'FB':<4}")
    print("-" * 92)
    summary = {}
    for task in TASKS:
        items = E.load_eval_items(TRACK, task, args.max_items, smoke=False)
        gen_items, tok = session.generate(task, items, args.sft_adapter)
        rows, fields_list = [], []
        for it in gen_items:
            tier, f = V.score_v2(task, it["raw_output"], it["gold"],
                                 prompt=it.get("prompt"), track=TRACK)
            fields_list.append(f)
            rows.append({"item_id": it["item_id"], "prompt": it["prompt"], "gold": it["gold"],
                         "raw_output": it["raw_output"],
                         "output_token_len": len(tok.encode(it["raw_output"])),
                         "tier": tier, "strict_pass": f["strict_pass"],
                         "semantic_pass": f["semantic_pass"],
                         "format_pass": f["format_pass"]})
        gp = f"generations/{TRACK}_baseline_v2_{task}.jsonl"
        with open(gp, "w") as fp:
            fp.write(json.dumps({"_header": True, "track": TRACK, "checkpoint": "baseline_sft",
                                 "task": task, "scorer": "v2", "canon_rules": V.CANON_RULES}) + "\n")
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        agg = V.aggregate(fields_list)
        summary[task] = agg
        rp = f"results/{TRACK}_baseline_v2_{task}.json"
        json.dump({**agg, "checkpoint": "baseline_sft", "scorer": "v2",
                   "canon_rules": V.CANON_RULES, "generations_file": gp}, open(rp, "w"), indent=2)
        print(f"{task:<12}{agg['n']:<5}{agg['semantic_accuracy']:<9.3f}{agg['strict_accuracy']:<12.3f}"
              f"{agg['pass']:<6}{agg['surface_form']:<6}{agg['semantic_error']:<5}{agg['format_break']:<4}")
    json.dump(summary, open("results/baseline_v2_summary.json", "w"), indent=2)
    print("\nwrote results/baseline_v2_summary.json (the on-disk reference for the equivalence test)")


if __name__ == "__main__":
    main()
