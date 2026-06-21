# Track-1 confirmatory matrix — RESULTS

Pre-registration: `notes/2026-06-18_note2_matrix_prereg_FROZEN.md`. 8 seeds, fixed 180 opt-steps, scorer v2, none aborted (final margins 6.08-7.68). All numbers from `kaggle_result/`.

## Primary endpoint — TOST equivalence +-5 pts vs baseline_v2 (semantic acc @ step 180)

```
# PRIMARY ENDPOINT — semantic accuracy at FINAL (step 180) vs baseline_v2
# TOST equivalence +-5 pts (90% CI). Note 2 FROZEN 2026-06-18. n=8 seeds, none aborted.

## json  [CEILING-FLAGGED: baseline>98, equivalence test detects DEGRADATION not improvement]
  baseline_v2 semantic acc :  98.75 pts
  mean final (8 seeds)     :  95.23 pts   (per-seed: 92.5 96.9 95.6 96.2 93.8 96.9 94.4 95.6)
  per-seed diff vs base    : -6.2 -1.9 -3.1 -2.5 -5.0 -1.9 -4.4 -3.1
  (1) seeds within +-5 pts : 6/8   -> FAIL (need >=7)
  (2) seed-level mean diff : -3.52 pts  90% t-CI[df=7] [-4.56, -2.47]  -> within +-5   (TOST p=0.0157)
  (3) item-level pooled    : -3.52 pts  90% boot-CI [-5.31, -1.95]  -> OUTSIDE +-5
  VERDICT: NOT EQUIVALENT (degraded beyond bound)

## router
  baseline_v2 semantic acc :  93.75 pts
  mean final (8 seeds)     :  63.59 pts   (per-seed: 64.4 76.9 58.1 63.1 61.9 48.8 69.4 66.2)
  per-seed diff vs base    : -29.4 -16.9 -35.6 -30.6 -31.9 -45.0 -24.4 -27.5
  (1) seeds within +-5 pts : 0/8   -> FAIL (need >=7)
  (2) seed-level mean diff : -30.16 pts  90% t-CI[df=7] [-35.65, -24.66]  -> OUTSIDE +-5   (TOST p=1)
  (3) item-level pooled    : -30.16 pts  90% boot-CI [-34.92, -25.47]  -> OUTSIDE +-5
  VERDICT: NOT EQUIVALENT (degraded beyond bound)

## extraction
  baseline_v2 semantic acc :  97.50 pts
  mean final (8 seeds)     :  96.95 pts   (per-seed: 97.5 94.4 98.1 98.1 98.8 95.0 93.8 100.0)
  per-seed diff vs base    : +0.0 -3.1 +0.6 +0.6 +1.3 -2.5 -3.7 +2.5
  (1) seeds within +-5 pts : 8/8   -> PASS (need >=7)
  (2) seed-level mean diff : -0.55 pts  90% t-CI[df=7] [-2.07, +0.98]  -> within +-5   (TOST p=0.000438)
  (3) item-level pooled    : -0.55 pts  90% boot-CI [-1.80, +0.86]  -> within +-5
  VERDICT: EQUIVALENT (preserved)

======================================================================
HEADLINE: TASK-SELECTIVE — equivalence holds on ['extraction']; FAILS (degrades beyond +-5 pts) on ['json', 'router'].
Per the pre-registered falsifier, the bounded-null headline is DROPPED; the task-selective branch governs.
```

## Capability-vs-margin curve (trajectory, mean over seeds)

```
[dpo/extraction] baseline=0.975  m=1.1:acc=0.995  m=3.1:acc=0.988  m=5.1:acc=0.979  m=6.7:acc=0.970  max_drop=0.005 -> PRESERVED
[dpo/json] baseline=0.988  m=1.1:acc=0.939  m=3.1:acc=0.891  m=5.1:acc=0.930  m=6.7:acc=0.952  max_drop=0.097 -> PRESERVED
[dpo/router] baseline=0.938  m=1.1:acc=0.816  m=3.1:acc=0.587  m=5.1:acc=0.626  m=6.7:acc=0.636  max_drop=0.351 -> DEGRADED
```
(`max_drop` is trajectory-min vs baseline; the *binding* test is the final-step TOST above.)

## Secondaries

### FORMAT_BREAK & semantic-error rate at final (mean+-sd, 8 seeds)

| task | format_break | semantic_error |
|---|---|---|
| json | 0.0023+-0.0043 | 0.0453+-0.0124 |
| router | 0.1062+-0.0698 | 0.2578+-0.0738 |
| extraction | 0.0102+-0.0171 | 0.0203+-0.0189 |

### Gold per-token logp (direction-only, pooled 160x8), margin1 -> final

| task | margin1 | final | delta | dir |
|---|---|---|---|---|
| json | -0.9933 | -1.1604 | -0.1671 | FALLS |
| router | -0.2422 | -1.7258 | -1.4837 | FALLS |
| extraction | -0.0097 | -0.0169 | -0.0073 | FALLS |

Gold-logp deflation magnitude tracks the behavioral degradation (router >> json >> extraction) — direction-consistent with the primary verdict.

### Register-shift probe replication (nf4, seeds 0/1) — secondary 6

Frozen rule X=0.05 nats/token (`DECISION_NOTE_register.md`). `register_decision.py --probe kaggle_result/A1/register_probe_nf4.jsonl`:

- **PRIMARY VERDICT: GOLD_PROTECTED** on both seeds (paired stat Dgold-Dmean = +0.30 / +0.23, 95% CI clear of 0).
- Probe replicates the probability-layer branch; consistent with extraction behavioral preservation (no falsifier contradiction).

