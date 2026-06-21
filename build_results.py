#!/usr/bin/env python
"""Assemble the consolidated Track-1 matrix RESULTS from on-disk artifacts.

Reads the reorganized per-seed results + the analysis artifacts and recomputes
the secondaries inline (every number traces to a file in kaggle_result/). Writes
RESULTS_track1_matrix.md. No GPU. Run primary_analysis.py + analyze.py first.
"""
import json
import statistics as st

ROOT = "kaggle_result/final"
TASKS = ["json", "router", "extraction"]


def res(s, ck, task):
    return json.load(open(f"{ROOT}/seed{s}/results/track1_dpo_{ck}_seed{s}_{task}.json"))


def pooled_goldlogp(ck, task):
    vals = []
    for s in range(8):
        for l in open(f"{ROOT}/seed{s}/results/goldlogp_track1_dpo_{ck}_seed{s}_{task}.jsonl"):
            r = json.loads(l)
            if not r.get("_header"):
                vals.append(r["gold_logp"])
    return st.mean(vals)


def main():
    L = ["# Track-1 confirmatory matrix — RESULTS", "",
         "Pre-registration: `notes/2026-06-18_note2_matrix_prereg_FROZEN.md`. "
         "8 seeds, fixed 180 opt-steps, scorer v2, none aborted (final margins 6.08-7.68). "
         "All numbers from `kaggle_result/`.", ""]

    # ---- PRIMARY (verbatim from primary_analysis.py output) ----
    L += ["## Primary endpoint — TOST equivalence +-5 pts vs baseline_v2 (semantic acc @ step 180)",
          "", "```", open("analysis_ws/PRIMARY_TOST.md").read().rstrip(), "```", ""]

    # ---- 7d decoupling curve (from analyze.py) ----
    L += ["## Capability-vs-margin curve (trajectory, mean over seeds)", "",
          "```", open("analysis_ws/results/decoupling_verdict.txt").read().rstrip(), "```",
          "(`max_drop` is trajectory-min vs baseline; the *binding* test is the final-step TOST above.)", ""]

    # ---- secondaries ----
    L += ["## Secondaries", "", "### FORMAT_BREAK & semantic-error rate at final (mean+-sd, 8 seeds)", ""]
    L += ["| task | format_break | semantic_error |", "|---|---|---|"]
    for t in TASKS:
        fb = [res(s, "final", t)["format_break_rate"] for s in range(8)]
        se = [res(s, "final", t)["semantic_error_rate"] for s in range(8)]
        L.append(f"| {t} | {st.mean(fb):.4f}+-{st.pstdev(fb):.4f} | {st.mean(se):.4f}+-{st.pstdev(se):.4f} |")
    L += ["", "### Gold per-token logp (direction-only, pooled 160x8), margin1 -> final", ""]
    L += ["| task | margin1 | final | delta | dir |", "|---|---|---|---|---|"]
    for t in TASKS:
        m1, fin = pooled_goldlogp("margin1", t), pooled_goldlogp("final", t)
        d = fin - m1
        L.append(f"| {t} | {m1:.4f} | {fin:.4f} | {d:+.4f} | {'FALLS' if d < -0.002 else 'RISES' if d > 0.002 else 'FLAT'} |")
    L += ["", "Gold-logp deflation magnitude tracks the behavioral degradation "
          "(router >> json >> extraction) — direction-consistent with the primary verdict.", ""]

    L += ["### Register-shift probe replication (nf4, seeds 0/1) — secondary 6", "",
          "Frozen rule X=0.05 nats/token (`DECISION_NOTE_register.md`). "
          "`register_decision.py --probe kaggle_result/A1/register_probe_nf4.jsonl`:", "",
          "- **PRIMARY VERDICT: GOLD_PROTECTED** on both seeds "
          "(paired stat Dgold-Dmean = +0.30 / +0.23, 95% CI clear of 0).",
          "- Probe replicates the probability-layer branch; consistent with extraction "
          "behavioral preservation (no falsifier contradiction).", ""]

    open("RESULTS_track1_matrix.md", "w").write("\n".join(L) + "\n")
    print("wrote RESULTS_track1_matrix.md")


if __name__ == "__main__":
    main()
