# Track-1 confirmatory matrix — consolidated analysis report

**Data:** `kaggle_result/final/seed{0..7}/` (matrix, scorer v2), `kaggle_result/A1/` (nf4 register probe),
`kaggle_result/A2/` (baseline_v2). **Prereg:** `notes/2026-06-18_note2_matrix_prereg_FROZEN.md`,
`DECISION_NOTE_register.md`. **Scripts:** `gold_replication.py`, `primary_analysis.py`,
`remaining_analyses.py`, `build_results.py`, `analyze.py`. All local, no GPU. 8 seeds, fixed 180
opt-steps, **none aborted** (final margins 6.08–7.68). Every number traces to a file in `kaggle_result/`.

---

## Bottom line — which pre-registered verdicts hold

| # | analysis | pre-registered verdict | status |
|---|---|---|---|
| 1 | gold-logp GOLD_PROTECTED replication | paired stat ≥+X, CI>0 | **N/A at 8-seed scale** (controls not logged); holds 2/2 where computable; thesis NOT refuted |
| 2 | TOST equivalence ±5 vs baseline_v2 | ≥7/8 + pooled within bound, all tasks | **TASK-SELECTIVE**: extraction PASS; json & router FAIL |
| 3 | chosen/rejected reward decomposition | margin driven by chosen↑ or rejected↓ | **NOT RUN — not logged** |
| 4 | held-out margin per-category | style axis faster than correctness | **PARTIAL** — aggregate only; per-category NOT logged |
| 5 | per-item flip / McNemar | — | regressions dominate; json strongly asymmetric |
| 6 | router format-break vs content-error trajectory | — | both rise; non-monotonic, worst at margin≈3 |
| 7 | margin-vs-capability per-seed figure | — | done (tight extraction/json, router fan) |

**Headline (binding primary, #2):** the pre-registered **bounded-null is DROPPED**; the
**TASK-SELECTIVE branch governs**. Free-generation capability is preserved within ±5 pts on
**extraction only**; **router degrades catastrophically (−30 pts)** and **json marginally (−3.5 pts,
ceiling-flagged)**.

**Complications to flag (see each section):** three logging gaps vs the frozen prereg (controls A/B/C,
reward decomposition, per-category held-out margin were not logged in the matrix); the damage is
**non-monotonic** in margin (worst at margin≈3, partial stochastic recovery by final) — not the
monotonic "grows with margin" picture; and the nf4 absolute gold decline is far smaller than the
fp16 preview.

---

## #1 — Gold-logp 8-seed GOLD_PROTECTED replication  (HIGHEST PRIORITY)

**Critical check:** matrix goldlogp files are **GOLD-ONLY** — union of keys across all 120 files is
exactly `{item_id, gold_logp, gold_completion_len}`. Controls A/B/C were never logged in the matrix.
A2 baseline_v2 has no gold-logp either. Controls exist **only** in `kaggle_result/A1/register_probe_nf4.jsonl`
(extraction, seeds 0/1).

Therefore the probe's **primary verdict is the paired statistic** `stat = Δgold − Δmean(A,B)`
(`DECISION_NOTE_register.md` §"Primary axis") — intrinsically **relative to controls** — and is
**not computable for any matrix seed**. It is established only on A1 seeds 0,1: both **GOLD_PROTECTED**
(stat +0.30 / +0.23, 95% CI clear of 0). The 8 matrix seeds replicate only the note's separately-flagged
**load-bearing absolute** leg `Δgold = gold(final) − gold(sft_baseline)`:

```
[LOAD-BEARING] absolute Delta_gold (extraction), ref = A1 sft_baseline per item:
  seed0 -0.0014 [-0.0065,+0.0031]  seed1 -0.0040 [-0.0076,-0.0010]  seed2 -0.0077 [-0.0127,-0.0035]
  seed3 +0.0065 [+0.0029,+0.0094]  seed4 +0.0102 [+0.0075,+0.0124]  seed5 -0.0085 [-0.0128,-0.0044]
  seed6 -0.0058 [-0.0108,-0.0015]  seed7 -0.0051 [-0.0109,+0.0001]
```

**Read:** extraction gold is **essentially flat in nf4** (|Δgold| ≤ 0.01 nats/token; 4/8 tiny decline,
2/8 tiny rise, 2/8 ambiguous) — consistent with *strong* protection (gold barely moves while, per A1,
controls collapse). **No seed shows gold deflating beyond controls → the two-layer thesis is NOT refuted**,
so per the stop-rule I continued.

**Three caveats (do not skip):**
- **Prereg gap:** Note 2 secondary #6 specified the *full* statistic replicated in the matrix with
  A/B/C controls. The matrix logged gold-only — so 8-seed paired replication is impossible; only seeds 0/1
  (probe) carry the verdict.
- **nf4 ≠ fp16:** the note's load-bearing expectation cited an fp16 decline (seed0 −0.023→−0.210). In nf4
  gold barely moves; the *large absolute decline does not reproduce* (direction/protection do).
- **Reference-limited:** matrix vs A1 final gold differ by up to 0.038 (seed0; A1 = pilot adapter), which
  exceeds the Δgold effect — so the absolute number is reference-limited, not a precise decline.

Source: `gold_replication.py` → `analysis_ws/01_gold_replication.txt`.

---

## #2 — Formal TOST (pre-registered CI construction)

Per-task semantic accuracy at the **final 180-step** checkpoint vs baseline_v2. TOST = two one-sided
tests at α=0.05 ⇔ **90% CI within ±5 pts**. Seed-level: mean per-seed diff, **t-CI df=7**. Item-level:
**bootstrap over items, seeds pooled**. Both must lie within bound; **≥7/8** seeds within bound.

| task | baseline | mean final | seed-level 90% t-CI | TOST p | item-level pooled CI | ≥7/8 | verdict |
|---|---|---|---|---|---|---|---|
| extraction | 97.50 | 96.95 | [−2.07, +0.98] ✓ | 4.4e-4 | [−1.80, +0.86] ✓ | 8/8 | **PASS (equivalent)** |
| json* | 98.75 | 95.23 | [−4.56, −2.47] ✓ | 0.016 | [−5.31, −1.95] ✗ | 6/8 | **FAIL** |
| router | 93.75 | 63.59 | [−35.65, −24.66] ✗ | 1.0 | [−34.92, −25.47] ✗ | 0/8 | **FAIL** |

\* **json is ceiling-flagged** (baseline >98 — equivalence can only detect degradation). It is the
**marginal** fail: point estimate −3.5 pts and the seed-level CI are *within* ±5; it fails only on the
item-level pooled CI (lower bound −5.31, just past −5) and the seed count.

**Flagged per request — json per-seed diffs:** `-6.2 -1.9 -3.1 -2.5 -5.0 -1.9 -4.4 -3.1`.
**seed0 = −6.25** (clearly outside ±5) and **seed4 = −5.00** (exactly at the bound; floating-point places
it just outside, giving 6/8 — counting it as "within" gives 7/8, but json still fails criterion (3), so the
FAIL is robust to this boundary).

Source: `primary_analysis.py` → `analysis_ws/02_primary_tost.txt`, `RESULTS_track1_matrix.md`.

---

## #3 — Chosen/rejected reward decomposition

**NOT RUN — the data does not exist.** The margin logs (`kaggle_result/final/seed*/logs/*_margin.jsonl`)
contain only `step`, `train_margin`, `pref_loss` (+ `val_margin`/`val_pref_loss` at 3 steps). There are
**no `chosen_reward` / `rejected_reward` fields** (the prereg's "3-line logging" secondary #4 was never
applied to the matrix run). The displacement question — is margin growth driven by chosen↑ or rejected↓ —
**cannot be answered from the matrix artifacts**; it requires re-running with the reward logging, or
teacher-forcing the chosen/rejected sides of the 98 training pairs through each checkpoint (a GPU job).

---

## #4 — Held-out margin per-category (style vs tool_calling)

**PARTIAL.** Held-out margin **is** logged, but **aggregate only** (over a 20-pair val subset) at steps
50/100/150 — there is **no style-axis vs tool_calling split**, so the pre-registered prediction (style
learned faster/larger than correctness) **cannot be tested**. What is available:

```
val_margin trajectory (aggregate), per seed, steps 50/100/150:
  seed0 1.70/3.53/5.75   seed1 1.65/3.09/5.90   seed2 1.68/3.62/5.75   seed3 1.66/3.51/7.40
  seed4 1.70/3.48/6.45   seed5 1.80/3.50/4.97   seed6 1.74/3.52/6.71   seed7 1.53/3.29/5.77
```

Held-out margin grows monotonically and tracks train margin → **the preference generalizes off the
training pairs** (it is not memorization). The per-category decomposition needs the category-split logging
or a re-eval of the 98 held-out pairs by category.

Source: `remaining_analyses.py` → `analysis_ws/03_remaining.txt`.

---

## #5 — Per-item flip analysis / McNemar (extraction, json; final vs baseline_v2)

Per-item `semantic_pass` flips, `b10` = pass→fail (regressions), `b01` = fail→pass (gains).

**extraction** — near-symmetric, small counts: pooled `b01=15, b10=22, net=-7`, McNemar χ²=1.0, **p=0.32**
(no significant asymmetry). Per-seed mostly p>0.05; seed6 (−6, p=0.031) and seed1 (−5, p=0.062) are the
only leaners. Consistent with the TOST "preserved" verdict — extraction churns a few items but does not
systematically regress.

**json** — **strongly asymmetric, one-directional**: pooled `b01=0, b10=45, net=-45`, McNemar χ²=43,
**p=5.4e-11**. **Every json flip is a regression (0 gains across all 8 seeds, 160 items).** seed0 (−10,
p=0.002) and seed4 (−8, p=0.008) are individually significant. So json's small mean drop is **not noise** —
it is a consistent, one-directional loss of ~3–10 items per seed, never offset by gains. This *complicates*
the "json marginal/ceiling" reading: the degradation is statistically clean even though small.

Source: `remaining_analyses.py` → `analysis_ws/03_remaining.txt`.

---

## #6 — Router FORMAT_BREAK vs content-error trajectory

Format-break (malformed output) and content/semantic error reported **separately** per checkpoint
(mean±std over 8 seeds):

```
ckpt     margin  sem_acc        fmt_break       content_err
step20   0.75    0.852±0.015    0.001±0.002     0.148±0.016
margin1  1.09    0.816±0.030    0.007±0.010     0.177±0.038
margin3  3.12    0.587±0.061    0.196±0.089     0.217±0.071
margin5  5.13    0.626±0.049    0.133±0.096     0.241±0.083
final    6.73    0.636±0.077    0.106±0.070     0.258±0.074
per-seed fmt_break @ final: 0.05 0.11 0.12 0.02 0.04 0.26 0.11 0.14
```

**Both failure modes contribute, and the structure is non-monotonic.** Format-break **spikes at margin≈3**
(0.196) then partly subsides (0.106 at final) — a transient formatting breakdown with stochastic recovery.
Content-error rises **monotonically** (0.148→0.258) — a steady, margin-driven loss of routing correctness
that does *not* recover. So the router collapse is **two distinct processes**: a transient format break
(worst mid-trajectory) plus a persistent content-correctness erosion. High per-seed format-break variance
at final (0.02–0.26) drives the fan in the figure.

Source: `remaining_analyses.py` → `analysis_ws/03_remaining.txt`.

---

## #7 — Core figure: margin vs capability

- **PRIMARY (per-seed):** `figures/capability_vs_margin_perseed.png` — 3 panels, one line per seed,
  x = each seed's own smoothed margin at the checkpoint. **extraction** = tight band on baseline;
  **json** = tight band with a shallow dip-and-recover; **router** = wide fan collapsing to 0.45–0.77.
  The variance structure (tight damage on stable tasks, stochastic high-variance recovery on router) is
  visible, as is the **non-monotonic dip at margin≈3** for both json and router.
- **Supplementary (mean±std):** `figures/capability_vs_margin_meanband.png`.
- Also: `figures/capability_vs_margin_{json,router,extraction}.png` (per-task, from `analyze.py`) and
  `figures/margin_trajectories.png` (train-margin vs step).

Source: `remaining_analyses.py` (figures), `analyze.py` (per-task + trajectories).

---

## What contradicts or complicates the thesis

1. **The binding primary fails on 2/3 tasks** — bounded-null is dropped; the result is task-selective
   degradation, with router catastrophic and json a clean (if small, ceiling-flagged) one-directional loss
   (#2, #5).
2. **Three prereg logging gaps** mean the mechanistic secondaries cannot be run as specified: matrix
   controls A/B/C (#1), reward decomposition (#3), per-category held-out margin (#4). The two-layer thesis's
   probability-layer leg rests on **2 probe seeds**, not 8.
3. **Damage is non-monotonic in margin** (worst at margin≈3, partial recovery by final) for both router and
   json (#6, #7). The trajectory-min is below the final endpoint — the simple "capability falls as margin
   grows" framing is incomplete; it is "capability dips through a transient, then partially recovers."
4. **nf4 gold is far more stable than the fp16 preview** (#1) — gold protection is, if anything, stronger in
   nf4, but the specific fp16 magnitudes the note anchored on do not reproduce.

No result contradicts GOLD_PROTECTED where it is measurable; the complications are about **coverage**
(logging gaps) and **shape** (non-monotonic), not a sign flip.
