# Note 1 — Behavioral branch mapping + gate-failure rule (2026-06-11)

Written BEFORE the baseline_v2 GPU run. The mapping is fixed now; nothing is decided
after seeing data. Binding comparison is **matrix-vs-baseline_v2 on the SAME 160-item
suite** (scorer v2, identical EvalSession pipeline). Pilot 80-item endpoint levels are
used only as a *comparison anchor* (cross-suite, so caveated, never as the gate value).

## Gate metric & inclusion (scorer v2)
- **Gate metric = semantic accuracy** (PASS + SURFACE_FORM) / n, per task.
- **Inclusion ≥ 0.80.** **Ceiling flag if > 0.98** (a drop would be unobservable).
- Any task **failing inclusion stays in the suite** but is **excluded from the headline
  equivalence claim** and reported **descriptively with disclosure**.
- The on-disk reference is **baseline_v2** (step-0 SFT, 160-suite, scorer v2). The lost
  80-item / 160-item interactive baselines are NOT used (see Note 3).

## Behavioral branch mapping (decided on baseline_v2's 160-suite value)
Pilot batch-1 router endpoints on the 160-suite were ~0.725–0.775 (canon == strict);
seed0 was flat ~0.76–0.81 across margin1→final (margin-independent), seed1 non-monotonic.

- **router baseline_v2 ≈ pilot endpoint levels (~0.73–0.81 on the 160-suite)**
  → **no behavioral degradation; CLEAN-NULL branch live** (router was simply at this level;
  the apparent "drop" was an artifact of an unverified ~0.95 memory).
- **router baseline_v2 high (~0.9+)**
  → **task-selective, early, margin-independent drop branch** (router falls between step 0
  and margin1, then flat — would fail a ±5 TOST). The step20 diagnostic checkpoint
  (Note 2) localizes the onset.
- **router baseline_v2 intermediate (≈0.82–0.89)**
  → **TOST-continuous; report as measured** (equivalence test decides per its bound, no
  branch forcing).

## Memory-corroboration framing (recorded BEFORE the run; nothing conditions on it)
If baseline_v2 lands near **json 0.99 / router 0.94 / extraction 0.95**, the (unsaved,
unverifiable) interactive-audit memory is *corroborated* — informally reassuring only.
**No decision, claim, or branch conditions on this.** If it lands elsewhere, baseline_v2
is authoritative and the memory is simply discarded (it has no on-disk artifact).

## Gate-failure rule (binding)
1. Compute baseline_v2 semantic accuracy per task on the 160-suite.
2. Task with baseline_v2 < 0.80 → excluded from the headline equivalence claim, reported
   descriptively. Task > 0.98 → ceiling-flagged (drop may be unobservable; note it).
3. The equivalence test (Note 2) runs only on tasks that clear inclusion and are not
   ceiling-flagged; others are reported as measured with explicit disclosure.
