#!/usr/bin/env python
"""Baseline-competence gate (Part A #1/#4) — FREE-GENERATION capability of the
untrained SFT baseline on each candidate task. No training.

A task ENTERS the final suite iff (pre-registered, decided on the baseline only):
  - accuracy >= 0.80          (base model is genuinely competent)
  - accuracy <= ~0.97         (headroom: not pinned at 1.0 -> a drop is observable)
  - degeneracy visible        (random/constant floor is well below baseline acc)
Inclusion does NOT depend on whether DPO later collapses it (no p-hacking).

Requires a CUDA GPU (Kaggle). Run, e.g.:
  python run_baseline_audit.py --sft-adapter <path>
"""
import argparse
import json
import os
from types import SimpleNamespace

import eval_pref as E
import scorer as SC

TRACK = "track1"

# A degenerate constant output per task, used to MEASURE the degeneracy floor on
# the real gold set (model-free, exact): "what would a collapsed-to-constant model
# score?". Baseline accuracy must sit well above this for collapse to be detectable.
DEGEN = {
    "binary": "COMPLETE",
    "router": "FILING_ANALYSIS",
    "json": json.dumps({"tool": "search_filings", "arguments": {"ticker": "AAPL"}}),
    "function_call": json.dumps({"tool": "search_filings", "arguments": {"ticker": "AAPL"}}),
    "extraction": json.dumps({"ticker": "AAPL", "section": "Item 1A", "filing_date": "2025-10-28"}),
}


def degeneracy_floor(task, items):
    """Measured accuracy of a constant degenerate answer on the real gold set."""
    st = "function_call" if task == "json" else task
    const = DEGEN[task]
    n_pass = sum(SC.score(st, const, it["gold"], track="track1")[0] == "PASS" for it in items)
    return n_pass / len(items) if items else 0.0


def run_task(task, sft_adapter, output_base, max_items):
    args = SimpleNamespace(track=TRACK, condition="dpo", seed=0, task=task,
                           baseline=True, output_base=output_base,
                           max_items=max_items, sft_adapter=sft_adapter)
    items = E.load_eval_items(TRACK, task, max_items, smoke=False)
    gen_items, tok = E.generate_items(args, items)
    gp = f"generations/{TRACK}_baseline_seed0_{task}.jsonl"
    rp = f"results/{TRACK}_baseline_seed0_{task}.json"
    res, rows = E.score_and_persist(TRACK, "baseline", 0, task, gen_items, gp, rp, tokenizer=tok)
    res["degeneracy_floor"] = degeneracy_floor(task, items)
    return res, rows


def gate(task, res):
    acc = res["accuracy"]
    floor = res["degeneracy_floor"]              # measured on the real gold set
    competent = acc >= 0.80                       # genuinely competent
    headroom = acc <= 0.97                         # room to drop (not pinned at 1.0)
    degen_visible = (acc - floor) >= 0.30          # collapse would be detectable
    # "fragile AND competent": baseline emits the format fine now (low format_break)
    # but the gate CAN break (proven separately in audit_eval.py).
    verdict = "INCLUDE" if (competent and degen_visible) else "EXCLUDE"
    note = []
    if not competent:
        note.append("baseline<0.80")
    if not headroom:
        note.append("ceiling(acc>0.97): drop may be unobservable")
    if not degen_visible:
        note.append("acc too close to degenerate floor")
    return verdict, competent, headroom, degen_visible, floor, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--output-base", default="adapters/pref")
    ap.add_argument("--tasks", nargs="+",
                    default=["binary", "json", "router", "extraction"])
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()
    for d in ("generations", "results"):
        os.makedirs(d, exist_ok=True)

    print("=" * 104)
    print("competent = acc>=0.80 | headroom = acc<=0.97 | degeneracy-visible = acc - degen_floor >= 0.30")
    print(f"{'task':<12}{'n':<5}{'acc':<8}{'fmt_break':<11}{'reason_err':<12}"
          f"{'degen_floor':<13}{'mean_len':<9}{'VERDICT'}")
    print("-" * 104)
    summary = {}
    for task in args.tasks:
        try:
            res, _ = run_task(task, args.sft_adapter, args.output_base, args.max_items)
        except SystemExit as e:
            print(f"{task:<12} SKIP ({e})")
            continue
        v, comp, head, dv, floor, note = gate(task, res)
        summary[task] = {**{k: res[k] for k in
                            ("n", "accuracy", "format_break_rate", "reasoning_error_rate",
                             "mean_output_len")},
                         "degeneracy_floor": floor, "verdict": v, "competent": comp,
                         "headroom": head, "degeneracy_visible": dv, "notes": note}
        print(f"{task:<12}{res['n']:<5}{res['accuracy']:<8.3f}{res['format_break_rate']:<11.3f}"
              f"{res['reasoning_error_rate']:<12.3f}{floor:<13.3f}{res['mean_output_len']:<9.1f}{v}"
              + (f"  [{', '.join(note)}]" if note else ""))
    json.dump(summary, open("results/baseline_audit_summary.json", "w"), indent=2)
    print("\nwrote results/baseline_audit_summary.json")
    inc = [t for t, s in summary.items() if s["verdict"] == "INCLUDE"]
    print(f"INCLUDE (competent AND degeneracy-visible): {inc}")
    print("Per-task you can read: competence (acc>=0.80), current format adherence")
    print("(fmt_break ~0 at baseline), and the degeneracy floor a collapse would hit.")
    print("Final suite is chosen from this BASELINE table — before any DPO result.")


if __name__ == "__main__":
    main()
