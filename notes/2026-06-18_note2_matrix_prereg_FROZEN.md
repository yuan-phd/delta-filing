# Note 2 — Confirmatory matrix preregistration — **FROZEN 2026-06-18**

> **STATUS: FROZEN.** All `[TBD-calibration]` fields are filled. No further design
> changes — the matrix is confirmatory. Supersedes the 2026-06-11 UNFROZEN draft.

## Thesis
Does **free-generation capability** on rigid exact-format tasks stay **preserved** as the
vanilla length-normalized DPO preference margin grows unbounded? Report whichever way it
goes (preserved = bounded-null headline; degraded = task-selective branch).

## Design
- **Single condition:** vanilla **length-normalized DPO** (β=0.5, **no clamp arm**;
  IPO/cDPO deferred). Completion-only length-normalized log-probs; assistant-only retention.
- **Batch = 8** (gradient accumulation; one forward/sample).
- **Training length:** **fixed optimizer-step count = 180**, identical across all seeds.
  **No loss-based early stop** (the convergence early-stop is disabled in code via
  `no_early_stop`; the run ends at exactly 180 optimizer steps).
- **Seeds:** 8.
- **Init:** SFT checkpoint **sha256 `29a3794bf640336ac4622a6337437aba2a4b6c7da1c1868f255ec238e7585f02`**
  (hash-verified == pilot init == baseline_v2 init).
- **Checkpoints (evaluated set): `margin1 / margin3 / margin5 / final`**, plus `step20`
  (diagnostic onset-localization). `final` = the fixed 180-step endpoint (the primary
  comparison point). margin1/3/5 are saved when the smoothed margin first crosses 1/3/5 and
  give the margin-axis curve; their margin value is read from the training log.
- **Eval:** 160-item suite (json/router/extraction), **frozen scorer v2** (three-tier;
  semantic accuracy primary, frozen canonicalization section-id `\d+[A-Z]?`), **identical
  EvalSession pipeline as baseline_v2** (matrix and baseline_v2 both call
  `scorer_v2.score_v2` — like-for-like).

## Baseline_v2 reference (frozen; on-disk, scorer v2, 160-suite)
| task | baseline_v2 semantic acc | inclusion |
|------|--------------------------|-----------|
| json | **0.988** | **ceiling-flagged (>0.98), INCLUDED with headroom caveat** |
| router | **0.938** | included (≥0.80) |
| extraction | **0.975** | included (≥0.80) |

All three pass inclusion (≥0.80); json is ceiling-flagged but **included** (ceiling affects
detecting improvement, not degradation — per Note 1). Binding comparison = matrix-vs-
baseline_v2 on the SAME 160-suite.

## Primary endpoint
Per-task **semantic accuracy at the final fixed step (180) vs baseline_v2**, **TOST ±5 pts**,
over 8 seeds: **≥7/8 seeds within bound AND pooled estimate within bound** (all three tasks;
ceiling-flagged json included with caveat). **CI construction:** TOST = two one-sided tests
at α=0.05 ⇔ **90% CI within ±5 pts**. *Seed-level:* mean per-seed accuracy difference vs
baseline_v2, **t-based CI (df=7)**. *Item-level pooled:* **bootstrap over items, seeds
pooled**. **Both must lie within bound.** baseline_v2 is **deterministic (greedy)**, so
comparison variance is from **training seeds + eval items only**.

## Integrity safeguards (pre-registered)
- **Divergence guard (pathology abort, NOT convergence early-stop).** During training a seed
  **aborts** if `pref_loss` becomes **NaN** OR the **smoothed margin exceeds 20** (~3× the
  calibration max of ~7.7). Implemented in `train_pref.py` as a hard `break` with a logged
  reason (`abort_reason` in `checkpoints.json`/`train_config.json`). An **aborted seed is
  excluded from the primary analysis with the reason logged; it is NOT replaced or re-run.**
  (No post-hoc decision about keeping a blown-up seed.) Verified: fires on margin>20 / NaN,
  does NOT fire on normal convergence (~7.7 runs the full 180).
- **Seed-split assignment (pre-specified).** **Seeds 0–4 → quota window 1; seeds 5–7 →
  quota window 2.** All 8 seeds are committed before any results are inspected. **No peeking**
  at window-1 results to adjust window-2. Analysis begins **only after all 8 seeds complete.**

## Pre-registered secondaries
1. **Trajectory-min** per seed/task over the **margin checkpoint set** (`step20` excluded).
2. **FORMAT_BREAK-rate trajectory** per seed/task.
3. **Held-out preference margin on all 98 pairs**, per-category (**style vs tool_calling**).
4. **chosen/rejected reward decomposition per checkpoint** (training-pair displacement;
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
- **Behavioral:** (a) **CLEAN-NULL** (preserved within ±5 on all included tasks);
  (b) **TASK-SELECTIVE** (≥1 task degrades — early, margin-independent, per Note 1). Note:
  baseline_v2 router = 0.938 (high), so if the pilots' ~0.73–0.81 router replicates, the
  task-selective branch is live.
- **Probability-layer** (extraction control probe, frozen mapping in DECISION_NOTE_register.md):
  **GLOBAL_DRIFT / GOLD_PROTECTED / TARGETED_DISPLACEMENT**. fp16 preview → GLOBAL_DRIFT/
  GOLD_PROTECTED; nf4 run decisive.

## Operational (frozen)
- **Per-seed wall-time:** ~5–6 h.
- **If per-checkpoint eval cost exceeds estimate → cut checkpoints before seeds.**

## Reference, don't restate
- Probe seed-disagreement rule + X=0.05 axis mapping: **`DECISION_NOTE_register.md` (frozen)**.
- Behavioral branch + gate-failure rule: **Note 1 (2026-06-11)**.
