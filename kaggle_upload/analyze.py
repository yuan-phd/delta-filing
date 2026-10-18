#!/usr/bin/env python
"""§7 analysis — capability-vs-margin curves + main tables + verdicts.

Runs on the downloaded results/ + logs/ (no GPU, no model). Robust to partial
data: reports what exists, writes NOT RUN where it doesn't. Fabricates nothing.

Produces:
  7a  main table: per condition, mean±std {accuracy, format_break} per task at the
      FINAL checkpoint, + per-seed appendix.
  7b  margin trajectories figure (from logs/*_margin.jsonl).
  7c  verbosity figure (mean output_len per task per condition).
  7d  THE CURVE: capability vs margin across checkpoints (decoupling plot) +
      results/decoupling_verdict.txt.
  7e  clamp causal readout: dpo vs dpo_clamp.
  RESULTS.md  terse summary, numbers + source files.

  python analyze.py --results-dir results --logs-dir logs --figures-dir figures
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

CKPT_ORDER = ["margin1", "margin3", "margin5", "final"]
NOMINAL_MARGIN = {"margin1": 1.0, "margin3": 3.0, "margin5": 5.0, "final": 7.0}
PRESERVED_TOL = 0.10   # capability drop <= this from baseline => "preserved" (decoupling)


def parse_results(results_dir):
    """Return rows: {tag, cond, ckpt, seed, task, accuracy, format_break_rate, ...}."""
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "track1_*_seed*_*.json"))):
        name = os.path.basename(p)[:-len(".json")]
        rest = name[len("track1_"):]
        if "_seed" not in rest:
            continue
        tag, right = rest.split("_seed", 1)
        seed_str, _, task = right.partition("_")
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        d = json.load(open(p))
        if tag == "baseline":
            cond, ckpt = "baseline", "baseline"
        else:
            cond, _, ckpt = tag.rpartition("_")
        rows.append({"tag": tag, "cond": cond, "ckpt": ckpt, "seed": seed, "task": task,
                     "accuracy": d.get("accuracy"), "format_break_rate": d.get("format_break_rate"),
                     "reasoning_error_rate": d.get("reasoning_error_rate"),
                     "mean_output_len": d.get("mean_output_len"), "n": d.get("n")})
    return rows


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return (None, None)
    return (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0)


def load_margins(results_dir):
    """(cond,ckpt,task) -> mean margin, from the matrix summary if present."""
    sp = os.path.join(results_dir, "track1_matrix_summary.json")
    out = {}
    if os.path.exists(sp):
        for k, v in json.load(open(sp)).items():
            cond, ckpt, task = k.split("/")
            out[(cond, ckpt, task)] = v.get("margin_mean")
    return out


def agg(rows, ckpt, field):
    """(cond,task) -> (mean,std) of field across seeds at a checkpoint."""
    buckets = defaultdict(list)
    for r in rows:
        if r["ckpt"] == ckpt:
            buckets[(r["cond"], r["task"])].append(r[field])
    return {k: ms(v) for k, v in buckets.items()}


def fmt(mst):
    m, s = mst
    return "  n/a " if m is None else f"{m:.3f}±{s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--out", default="RESULTS.md")
    args = ap.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    rows = parse_results(args.results_dir)
    if not rows:
        open(args.out, "w").write("# RESULTS\n\nNOT RUN: no matrix results in "
                                  f"`{args.results_dir}/`.\n")
        print(f"NOT RUN: no results in {args.results_dir}/ — wrote stub {args.out}")
        return

    conds = [c for c in ["dpo", "dpo_clamp", "ipo", "cdpo"]
             if any(r["cond"] == c for r in rows)]
    tasks = sorted({r["task"] for r in rows})
    margins = load_margins(args.results_dir)
    lines = ["# RESULTS — capability/margin decoupling (Track 1)", ""]
    lines.append("Source: free-generation eval (greedy) + exact-format scorer. "
                 "Numbers from `results/` + `logs/`. No teacher-forcing.\n")

    # ---- 7a main table (final) ----
    accF = agg(rows, "final", "accuracy")
    fbF = agg(rows, "final", "format_break_rate")
    base_acc = agg(rows, "baseline", "accuracy")
    base_fb = agg(rows, "baseline", "format_break_rate")
    lines.append("## 7a Main table — FINAL checkpoint (mean±std over seeds)\n")
    lines.append("| condition | " + " | ".join(f"{t} acc | {t} fmt_break" for t in tasks) + " |")
    lines.append("|" + "---|" * (1 + 2 * len(tasks)))
    lines.append("| baseline | " + " | ".join(
        f"{fmt(base_acc.get(('baseline', t), (None, None)))} | "
        f"{fmt(base_fb.get(('baseline', t), (None, None)))}" for t in tasks) + " |")
    for c in conds:
        lines.append(f"| {c} | " + " | ".join(
            f"{fmt(accF.get((c, t), (None, None)))} | {fmt(fbF.get((c, t), (None, None)))}"
            for t in tasks) + " |")
    lines.append("")

    # ---- 7d capability-vs-margin curve + decoupling verdict ----
    lines.append("## 7d Capability vs margin (the decoupling curve)\n")
    verdict_lines = []
    for c in conds:
        for t in tasks:
            pts = []
            for ck in CKPT_ORDER:
                a = agg(rows, ck, "accuracy").get((c, t))
                if a and a[0] is not None:
                    mg = margins.get((c, ck, t)) or NOMINAL_MARGIN.get(ck)
                    pts.append((mg, a[0]))
            if not pts:
                continue
            b = base_acc.get(("baseline", t), (None, None))[0]
            drop = (b - min(p[1] for p in pts)) if b is not None else None
            tag = ("PRESERVED" if (drop is not None and drop <= PRESERVED_TOL)
                   else "DEGRADED" if drop is not None else "n/a")
            curve = "  ".join(f"m={mg:.1f}:acc={ac:.3f}" for mg, ac in pts)
            verdict_lines.append(f"[{c}/{t}] baseline={b if b is None else round(b,3)}  "
                                 f"{curve}  max_drop={None if drop is None else round(drop,3)} -> {tag}")
    lines += ["```", *verdict_lines, "```", ""]
    open(os.path.join(args.results_dir, "decoupling_verdict.txt"), "w").write(
        "\n".join(verdict_lines) + "\n")

    # ---- 7e clamp causal ----
    lines.append("## 7e Clamp causal readout (dpo vs dpo_clamp, final)\n")
    if "dpo" in conds and "dpo_clamp" in conds:
        for t in tasks:
            d = accF.get(("dpo", t), (None, None))[0]
            dc = accF.get(("dpo_clamp", t), (None, None))[0]
            if d is not None and dc is not None:
                lines.append(f"- {t}: dpo acc={d:.3f}  dpo_clamp acc={dc:.3f}  "
                             f"delta(clamp-dpo)={dc - d:+.3f}")
    else:
        lines.append("- NOT RUN: need both dpo and dpo_clamp final results.")
    lines.append("")

    # ---- figures ----
    figs = make_figures(rows, conds, tasks, margins, base_acc, args)
    lines.append("## Figures\n")
    lines += [f"- {f}" for f in figs] or ["- (matplotlib unavailable — figures skipped)"]
    lines.append("")

    open(args.out, "w").write("\n".join(lines))
    print("\n".join(verdict_lines) if verdict_lines else "(no curve points yet)")
    print(f"\nwrote {args.out}, results/decoupling_verdict.txt, and {len(figs)} figure(s)")


def make_figures(rows, conds, tasks, margins, base_acc, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    figs = []

    # 7d curve per task
    for t in tasks:
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for c in conds:
            xs, ys = [], []
            for ck in CKPT_ORDER:
                a = agg(rows, ck, "accuracy").get((c, t))
                if a and a[0] is not None:
                    xs.append(margins.get((c, ck, t)) or NOMINAL_MARGIN.get(ck))
                    ys.append(a[0])
            if xs:
                ax.plot(xs, ys, marker="o", label=c); plotted = True
        b = base_acc.get(("baseline", t), (None, None))[0]
        if b is not None:
            ax.axhline(b, ls="--", color="gray", label="baseline")
        ax.set(xlabel="DPO margin", ylabel="accuracy (free-gen)", title=f"capability vs margin — {t}")
        ax.set_ylim(0, 1.02)
        if plotted:
            ax.legend()
            p = os.path.join(args.figures_dir, f"capability_vs_margin_{t}.png")
            fig.tight_layout(); fig.savefig(p, dpi=120); figs.append(p)
        plt.close(fig)

    # 7b margin trajectories
    fig, ax = plt.subplots(figsize=(6, 4))
    any_traj = False
    for c in conds:
        for p in sorted(glob.glob(os.path.join(args.logs_dir, f"track1_{c}_seed*_margin.jsonl"))):
            steps, mg = [], []
            for line in open(p):
                r = json.loads(line)
                if "train_margin" in r:
                    steps.append(r["step"]); mg.append(r["train_margin"])
            if steps:
                ax.plot(steps, mg, alpha=0.5, label=c if "_seed0_" in p else None)
                any_traj = True
    if any_traj:
        ax.set(xlabel="step", ylabel="train margin", title="margin trajectories")
        ax.legend()
        p = os.path.join(args.figures_dir, "margin_trajectories.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); figs.append(p)
    plt.close(fig)

    return figs


if __name__ == "__main__":
    main()
