#!/usr/bin/env python
"""Track-1 training+eval matrix with margin-trajectory checkpoints.

For each (condition, seed): train once (saving adapter checkpoints at smoothed
margin ~= 1, 3, 5, final), then eval capability at EVERY checkpoint on the final
task suite. This yields the capability-vs-margin curve the decoupling claim needs.

Base model loaded from cache/mount (download once). Does NOT run until invoked.
Use matrix_kernel.py per-condition (DELTA_CONDITION) so each kernel is one
condition x 5 seeds. Outputs are no-overwrite and merge cleanly.

  python run_matrix.py --sft-adapter <path> --pref-data <path>
  python run_matrix.py --conditions dpo --seeds 0 1 --tasks json router extraction
"""
import argparse
import gc
import json
import os
import statistics
from types import SimpleNamespace

import eval_pref as E
import train_pref as T

try:
    import torch
except Exception:
    torch = None

TRACK = "track1"
CONDITIONS = ["dpo", "dpo_clamp", "ipo", "cdpo"]
SEEDS = [0, 1, 2, 3, 4]
# Final suite (binary dropped). Override after the baseline-competence gate.
TASKS = ["json", "router", "extraction"]


def _free():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def eval_adapter(session, tag, seed, task, adapter_path, max_items=None):
    """Eval one adapter (via the shared session) on one task; persist by `tag`.
    Also logs the secondary gold per-token log-prob (all 3 tasks)."""
    rp = f"results/{TRACK}_{tag}_seed{seed}_{task}.json"
    items = E.load_eval_items(TRACK, task, max_items, smoke=False)
    if not os.path.exists(rp):                   # no recompute of the behavioral eval
        gen_items, tok = session.generate(task, items, adapter_path)
        gp = f"generations/{TRACK}_{tag}_seed{seed}_{task}.jsonl"
        # FROZEN scorer v2 (three-tier; semantic accuracy = primary) — same scorer as
        # baseline_v2, so matrix-vs-baseline_v2 is like-for-like and the extraction
        # register-shift artifact is canonicalized into SURFACE_FORM, not REASONING_ERROR.
        res, _ = E.score_and_persist_v2(TRACK, tag, seed, task, gen_items, gp, rp, tokenizer=tok)
    else:
        res = json.load(open(rp))
    gold_logp_log(session, tag, seed, task, adapter_path, items)   # secondary, near-free
    return res


def gold_logp_log(session, tag, seed, task, adapter_path, items):
    """SECONDARY: teacher-forced gold per-token log-prob per item (all 3 tasks),
    so 'extraction deflation' can be tested across json/router/extraction."""
    lp = f"results/goldlogp_{TRACK}_{tag}_seed{seed}_{task}.jsonl"
    if os.path.exists(lp):
        return
    recs = session.gold_logprob(task, items, adapter_path)
    with open(lp, "w") as f:
        f.write(json.dumps({"_header": True, "track": TRACK, "tag": tag, "seed": seed,
                            "task": task, "metric": "gold_per_token_logp",
                            "n": len(recs)}) + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--pref-data", default=None)
    ap.add_argument("--output-base", default="adapters/pref")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="reuse baseline results from the baseline audit (per-seed kernels)")
    # smoke knobs (exercise the full path on a small slice)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=2)
    ap.add_argument("--max-items", type=int, default=None, help="eval item cap (smoke)")
    ap.add_argument("--margin-checkpoints", nargs="+", type=float, default=None,
                    help="override margin checkpoint levels (smoke uses low ones)")
    ap.add_argument("--step-checkpoints", nargs="+", type=int, default=None,
                    help="override fixed-step checkpoint(s) (smoke uses a small step)")
    args = ap.parse_args()
    for d in ("logs", "generations", "results", args.output_base):
        os.makedirs(d, exist_ok=True)

    # baseline (skip in per-seed matrix kernels; reuse the baseline-audit results).
    # Uses its own short-lived session so the base isn't resident during training.
    if not args.skip_baseline:
        print("=== baseline eval (SFT) ===")
        sess = E.EvalSession(TRACK)
        for task in args.tasks:
            eval_adapter(sess, "baseline", 0, task, args.sft_adapter, max_items=args.max_items)
        del sess
        _free()

    # (cond, ckpt_label, task) -> list over seeds of (acc, fmt_break, smoothed_margin)
    curve = {}
    for cond in args.conditions:
        for seed in args.seeds:
            print(f"\n=== TRAIN {cond} seed{seed} ===")
            tr = SimpleNamespace(track=TRACK, condition=cond, seed=seed,
                                 output_base=args.output_base, sft_adapter=args.sft_adapter,
                                 pref_data=args.pref_data, smoke=args.smoke,
                                 smoke_n=args.smoke_n, max_steps=args.max_steps,
                                 log_every=(args.log_every if args.smoke else None),
                                 eval_every=(10_000 if args.smoke else None),
                                 margin_checkpoints=args.margin_checkpoints,
                                 step_checkpoints=args.step_checkpoints)
            T.train(tr)                      # frees its own models on return
            _free()
            run_dir = os.path.join(args.output_base, f"{TRACK}_{cond}_seed{seed}")
            ckpts = json.load(open(os.path.join(run_dir, "checkpoints.json")))["checkpoints"]
            print(f"=== EVAL {cond} seed{seed}: {[c['label'] for c in ckpts]} ===")
            # one base load for ALL checkpoints x tasks of this run; freed after
            sess = E.EvalSession(TRACK)
            for ck in ckpts:
                tag = f"{cond}_{ck['label']}"
                for task in args.tasks:
                    r = eval_adapter(sess, tag, seed, task, ck["dir"], max_items=args.max_items)
                    curve.setdefault((cond, ck["label"], task), []).append(
                        (r["accuracy"], r["format_break_rate"], ck["smoothed_margin"]))
            del sess
            _free()

    # summary: capability-vs-margin curve (mean across seeds)
    print("\n" + "=" * 92)
    print(f"{'condition':<11}{'ckpt':<9}{'task':<11}{'margin':<9}{'acc(m±s)':<18}{'fmt_break(m±s)'}")
    out = {}
    for (cond, lbl, task), rows in sorted(curve.items()):
        accs = [x[0] for x in rows]; fbs = [x[1] for x in rows]; mgs = [x[2] for x in rows]
        def ms(v): return (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0)
        am, asd = ms(accs); fm, fsd = ms(fbs); mm = statistics.mean(mgs)
        out[f"{cond}/{lbl}/{task}"] = {"margin_mean": mm, "accuracy_mean": am,
                                       "accuracy_std": asd, "format_break_mean": fm,
                                       "format_break_std": fsd, "n_seeds": len(rows)}
        print(f"{cond:<11}{lbl:<9}{task:<11}{mm:<9.2f}{f'{am:.3f}±{asd:.3f}':<18}{f'{fm:.3f}±{fsd:.3f}'}")
    json.dump(out, open("results/track1_matrix_summary.json", "w"), indent=2)
    print("\nwrote results/track1_matrix_summary.json  (capability-vs-margin curve)")
    print("Run analyze.py (§7) next.")


if __name__ == "__main__":
    main()
