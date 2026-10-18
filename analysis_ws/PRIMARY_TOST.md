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
