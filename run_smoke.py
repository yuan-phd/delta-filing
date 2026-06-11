#!/usr/bin/env python
"""Smoke test as a GATE (claude_code_spec.md §9 step 3) — NOT a smoke check.

Order (hard-stops if a gate fails; never launches the full matrix):
  1. Baseline (SFT) eval FIRST on binary + json.
     GATE 1: baseline strict format_break_rate must be near zero on BOTH tasks.
             Substantial break => the strict gate is mismatched to the trained
             "label: reason" format => STOP.
  2. DPO smoke train (~30 steps) under completion-only log-probs.
     GATE 2: the DPO train_margin must climb (final > 0 and increasing) over the
             smoke steps. Confirms the unbounded-growth phenomenon survives the
             completion-only rescale before we build 20 runs on it.
  3. DPO eval on binary + json; print real generations + verdicts + scorer fields.

Requires a CUDA GPU (Kaggle). Run, e.g.:
  python run_smoke.py --sft-adapter /kaggle/input/.../sft \
                      --pref-data /kaggle/input/.../dpo_pytorch.jsonl
"""
import argparse
import json
import os
from types import SimpleNamespace

import eval_pref as E
import train_pref as T

TRACK = "track1"
SEED = 0
TASKS = ["binary", "json"]


def hr(t):
    print("\n" + "#" * 72 + f"\n# {t}\n" + "#" * 72)


def run_eval(condition, baseline, sft_adapter, output_base, max_items):
    out = {}
    for task in TASKS:
        args = SimpleNamespace(track=TRACK, condition=condition, seed=SEED, task=task,
                               baseline=baseline, output_base=output_base,
                               max_items=max_items, sft_adapter=sft_adapter)
        items = E.load_eval_items(TRACK, task, max_items, smoke=True)
        gen_items, tok = E.generate_items(args, items)
        cond = "baseline" if baseline else condition
        gen_path = f"generations/{TRACK}_{cond}_seed{SEED}_{task}.jsonl"
        res_path = f"results/{TRACK}_{cond}_seed{SEED}_{task}.json"
        results, rows = E.score_and_persist(TRACK, cond, SEED, task, gen_items,
                                            gen_path, res_path, tokenizer=tok)
        out[task] = (results, rows)
    return out


def read_margin_trajectory(condition):
    path = f"logs/{TRACK}_{condition}_seed{SEED}_margin.jsonl"
    traj = []
    for line in open(path):
        rec = json.loads(line)
        if rec.get("_header"):
            continue
        if "train_margin" in rec:
            traj.append((rec["step"], rec["train_margin"], rec.get("pref_loss")))
    return traj


def gate1_baseline(base_out, break_threshold):
    hr("GATE 1 — baseline (SFT) strict format_break on binary + json")
    passed = True
    for task in TASKS:
        res, rows = base_out[task]
        fbr = res["format_break_rate"]
        fbr_len = res["format_break_rate_lenient"]
        ok = fbr <= break_threshold
        passed = passed and ok
        print(f"  {task:<6} n={res['n']}  strict_format_break={fbr:.2f}  "
              f"lenient_format_break={fbr_len:.2f}  accuracy={res['accuracy']:.2f}  "
              f"-> {'OK' if ok else 'FAIL'}")
        for r in rows:
            if r["verdict"] == "FORMAT_BREAK":
                print(f"      break: id={r['item_id']} leading_prose={r['leading_prose']} "
                      f"raw={r['raw_output'][:80]!r}")
    print(f"  GATE 1: {'PASS' if passed else 'FAIL'} "
          f"(threshold strict_format_break <= {break_threshold})")
    return passed


def gate2_margin(traj):
    hr("GATE 2 — DPO smoke margin trajectory (completion-only log-probs)")
    print("  step   train_margin   pref_loss")
    for step, mgn, pl in traj:
        print(f"  {step:>4}   {mgn:>+12.4f}   {pl if pl is None else f'{pl:.4f}'}")
    if len(traj) < 2:
        print("  GATE 2: FAIL (insufficient trajectory points)")
        return False
    margins = [m for _, m, _ in traj]
    k = max(1, len(margins) // 3)
    first, last = sum(margins[:k]) / k, sum(margins[-k:]) / k
    climbs = last > first
    positive = margins[-1] > 0
    passed = climbs and positive
    print(f"  first-third mean={first:+.4f}  last-third mean={last:+.4f}  "
          f"final={margins[-1]:+.4f}")
    print(f"  climbing={climbs}  final_positive={positive}")
    print(f"  GATE 2: {'PASS' if passed else 'FAIL'} (need increasing AND final > 0)")
    return passed


def show_generations(dpo_out, k):
    hr("DPO smoke generations (real) + verdicts + scorer intermediates")
    for task in TASKS:
        _, rows = dpo_out[task]
        print(f"\n=== task={task} ===")
        for r in rows[:k]:
            print(f"  [{r['item_id']}] verdict={r['verdict']}")
            print(f"     gold       = {r['gold']}")
            print(f"     raw_output = {r['raw_output'][:160]!r}")
            print(f"     format_pass={r['format_pass']}  lenient={r['lenient_format_pass']}  "
                  f"leading_prose={r['leading_prose']}  norm={r['normalization_applied']}  "
                  f"content_correct={r['content_correct']}  out_len={r['output_token_len']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", required=True, help="path to track1 SFT adapter (baseline + init)")
    ap.add_argument("--pref-data", default=None, help="override preference data path")
    ap.add_argument("--output-base", default="adapters/pref")
    ap.add_argument("--max-items", type=int, default=5)
    ap.add_argument("--smoke-n", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--log-every", type=int, default=2)
    ap.add_argument("--break-threshold", type=float, default=0.20,
                    help="GATE 1 max baseline strict format_break_rate")
    args = ap.parse_args()

    for d in ("logs", "generations", "results"):
        os.makedirs(d, exist_ok=True)

    # ---- step 1: baseline FIRST ----
    hr("STEP 1 — baseline (SFT) eval")
    base_out = run_eval("dpo", baseline=True, sft_adapter=args.sft_adapter,
                        output_base=args.output_base, max_items=args.max_items)
    if not gate1_baseline(base_out, args.break_threshold):
        print("\nSTOP: GATE 1 failed. Gate mismatched to trained format — "
              "NOT launching the matrix. Report to researcher.")
        return 2

    # ---- step 2: DPO smoke train ----
    hr("STEP 2 — DPO smoke train")
    train_args = SimpleNamespace(
        track=TRACK, condition="dpo", seed=SEED, output_base=args.output_base,
        sft_adapter=args.sft_adapter, pref_data=args.pref_data, smoke=True,
        smoke_n=args.smoke_n, max_steps=args.max_steps, log_every=args.log_every,
        eval_every=10_000)  # skip val in smoke; we only need the train-margin climb
    T.train(train_args)
    traj = read_margin_trajectory("dpo")
    if not gate2_margin(traj):
        print("\nSTOP: GATE 2 failed. DPO margin did not climb under completion-only "
              "log-probs — NOT launching the matrix. Report to researcher.")
        return 3

    # ---- step 3: DPO eval ----
    hr("STEP 3 — DPO smoke eval")
    dpo_out = run_eval("dpo", baseline=False, sft_adapter=args.sft_adapter,
                       output_base=args.output_base, max_items=args.max_items)
    show_generations(dpo_out, args.max_items)

    hr("SMOKE GATE RESULT: BOTH GATES PASSED")
    print("Both gates passed. Awaiting researcher sign-off before the full matrix "
          "(do NOT auto-launch).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
