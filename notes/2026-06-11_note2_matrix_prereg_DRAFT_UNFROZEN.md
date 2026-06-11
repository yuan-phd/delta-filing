# Note 2 — Confirmatory matrix preregistration — **DRAFT, UNFROZEN** (2026-06-11)

> **STATUS: UNFROZEN DRAFT.** Every field marked **[TBD-calibration]** is filled from the
> 2-seed calibration; freezing is a *fill-in*, not a rewrite. Do not cite as final.
> Run order before freeze: **nf4 probe → baseline_v2 → 2-seed calibration → freeze this**.

## Thesis
Does **free-generation capability** on rigid exact-format tasks stay **preserved** as the
vanilla length-normalized DPO preference margin grows unbounded? Report whichever way it
goes (preserved = bounded-null headline; degraded = task-selective branch).

## Design
- **Single condition:** vanilla **length-normalized DPO** (β=0.5, **no clamp arm**;
  IPO/cDPO deferred). Completion-only length-normalized log-probs; assistant-only retention.
- **Batch = 8** (gradient accumulation; one forward/sample).
- **Training length:** **fixed optimizer-step count = [TBD-calibration]**, identical across
  all seeds. **No loss-based early stop** (removed; never fires mid-run differently per seed).
- **Seeds:** 8.
- **Init:** SFT checkpoint **sha256 `29a3794bf640336ac4622a6337437aba2a4b6c7da1c1868f255ec238e7585f02`**
  (hash-verified == pilot init == baseline_v2 init).
- **Checkpoints:** `step20` (diagnostic, onset localization) **+ fixed-step set ≈ epoch
  boundaries = [TBD-calibration]** (margin-keyed 1/3/5 retained as smoothed-margin tags for
  the curve; the fixed-step set is the binding one).
- **Eval:** 160-item suite (json/router/extraction), **frozen scorer v2**, **identical
  EvalSession pipeline as baseline_v2**.

## Primary endpoint
Per-task **semantic accuracy at the final fixed step vs baseline_v2**, **TOST ±5 points**,
aggregated over 8 seeds: **≥7/8 seeds within bound AND pooled estimate within bound**.
(Only tasks clearing the Note-1 inclusion gate and not ceiling-flagged.)

## Pre-registered secondaries
1. **Trajectory-min** per seed/task over the **margin-checkpoint set** (`step20` excluded).
2. **FORMAT_BREAK-rate trajectory** per seed/task.
3. **Held-out preference margin on all 98 pairs**, per-category (**style-axis vs
   tool_calling correctness-axis**).
4. **chosen/rejected reward decomposition per checkpoint** (displacement on training pairs;
   logging now staged — was never logged in the pilots).
5. **gold-logp trajectory × 3 tasks** (**direction-only cross-task**, not magnitude).
6. **Extraction gold + A/B/C control logp** — probe statistic **replicated in the matrix**;
   frozen threshold **X = 0.05 nats/token** (see DECISION_NOTE_register.md).
7. **gold-variant register tracking** ("Item 8" vs "8").
8. **KL-from-reference per checkpoint**.

## Falsifier (pre-registered)
- If **any task degrades beyond the equivalence bound**, the **bounded-null headline is
  dropped** and the **task-selective branch governs**.
- If the **probe-replication statistic contradicts the probe's branch**, the **disagreement
  is reported, not reconciled post-hoc**.

## Branches to write up (both behavioral, all three probability-layer)
- **Behavioral:** (a) **CLEAN-NULL** (capability preserved within ±5 on all included tasks);
  (b) **TASK-SELECTIVE** (≥1 task degrades — early margin-independent, per Note 1).
- **Probability-layer** (extraction control probe, per the frozen mapping in
  DECISION_NOTE_register.md): **GLOBAL_DRIFT** / **GOLD_PROTECTED** / **TARGETED_DISPLACEMENT**.
  The fp16 preview points to GLOBAL_DRIFT/GOLD_PROTECTED; the nf4 run is decisive.

## Reference, don't restate
- The **probe seed-disagreement rule** and the **X = 0.05** primary/secondary axis mapping
  live in **`DECISION_NOTE_register.md` (frozen)** — cited, not duplicated.
- Behavioral branch + gate-failure rule: **Note 1 (2026-06-11)**.

## [TBD-calibration] checklist (fill at freeze)
- [ ] fixed optimizer-step count (total training length)
- [ ] fixed-step checkpoint set (≈ epoch boundaries) — confirm `step20` + N more
- [ ] confirm margin reached at the final fixed step (~5–6 target)
- [ ] per-run wall-time × (8 seeds) fits the 30 h weekly quota; split if needed
- [ ] baseline_v2 per-task values + which tasks clear inclusion / ceiling flags (Note 1)
