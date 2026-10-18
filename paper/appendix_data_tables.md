# Appendix Data Tables

_Generated from existing result files. No re-computation of model outputs._

## Appendix A — Full per-seed TOST data

Per-task **semantic_accuracy** (%) at each checkpoint, all 8 seeds, plus baseline_v2. Δ = final − baseline (percentage points). TOST per-seed equivalence band is |Δ| ≤ 5 pp.

### extraction

| Task | Seed | Baseline | step20 | margin1 | margin3 | margin5 | final | Δ (final−baseline) | TOST (|Δ|≤5) |
|------|------|----------|--------|---------|---------|---------|-------|--------------------|--------------|
| extraction | 0 | 97.50 | 97.50 | 97.50 | 98.12 | 96.25 | 97.50 | +0.00 | PASS |
| extraction | 1 | 97.50 | 98.75 | 100.00 | 98.75 | 98.75 | 94.38 | -3.12 | PASS |
| extraction | 2 | 97.50 | 99.38 | 99.38 | 99.38 | 98.12 | 98.12 | +0.62 | PASS |
| extraction | 3 | 97.50 | 98.12 | 99.38 | 98.12 | 97.50 | 98.12 | +0.62 | PASS |
| extraction | 4 | 97.50 | 99.38 | 100.00 | 99.38 | 95.62 | 98.75 | +1.25 | PASS |
| extraction | 5 | 97.50 | 100.00 | 100.00 | 99.38 | 99.38 | 95.00 | -2.50 | PASS |
| extraction | 6 | 97.50 | 97.50 | 99.38 | 97.50 | 97.50 | 93.75 | -3.75 | PASS |
| extraction | 7 | 97.50 | 99.38 | 100.00 | 100.00 | 100.00 | 100.00 | +2.50 | PASS |

### json

| Task | Seed | Baseline | step20 | margin1 | margin3 | margin5 | final | Δ (final−baseline) | TOST (|Δ|≤5) |
|------|------|----------|--------|---------|---------|---------|-------|--------------------|--------------|
| json | 0 | 98.75 | 82.50 | 93.75 | 93.12 | 94.38 | 92.50 | -6.25 | FAIL |
| json | 1 | 98.75 | 93.75 | 95.62 | 80.62 | 89.38 | 96.88 | -1.88 | PASS |
| json | 2 | 98.75 | 88.75 | 88.75 | 93.12 | 85.00 | 95.62 | -3.12 | PASS |
| json | 3 | 98.75 | 96.88 | 95.00 | 93.75 | 93.75 | 96.25 | -2.50 | PASS |
| json | 4 | 98.75 | 91.88 | 93.75 | 76.88 | 95.62 | 93.75 | -5.00 | FAIL |
| json | 5 | 98.75 | 95.00 | 95.00 | 94.38 | 96.25 | 96.88 | -1.88 | PASS |
| json | 6 | 98.75 | 90.62 | 95.62 | 86.88 | 95.00 | 94.38 | -4.38 | PASS |
| json | 7 | 98.75 | 95.62 | 93.75 | 93.75 | 94.38 | 95.62 | -3.12 | PASS |

### router

| Task | Seed | Baseline | step20 | margin1 | margin3 | margin5 | final | Δ (final−baseline) | TOST (|Δ|≤5) |
|------|------|----------|--------|---------|---------|---------|-------|--------------------|--------------|
| router | 0 | 93.75 | 85.62 | 78.12 | 59.38 | 61.88 | 64.38 | -29.37 | FAIL |
| router | 1 | 93.75 | 87.50 | 86.88 | 61.25 | 66.88 | 76.88 | -16.87 | FAIL |
| router | 2 | 93.75 | 86.25 | 83.75 | 61.25 | 60.62 | 58.13 | -35.62 | FAIL |
| router | 3 | 93.75 | 83.75 | 77.50 | 65.00 | 61.25 | 63.12 | -30.63 | FAIL |
| router | 4 | 93.75 | 82.50 | 80.62 | 54.37 | 69.38 | 61.88 | -31.87 | FAIL |
| router | 5 | 93.75 | 84.38 | 82.50 | 44.38 | 51.88 | 48.75 | -45.00 | FAIL |
| router | 6 | 93.75 | 85.62 | 83.75 | 61.25 | 64.38 | 69.38 | -24.38 | FAIL |
| router | 7 | 93.75 | 85.62 | 79.38 | 62.50 | 64.38 | 66.25 | -27.50 | FAIL |

## Appendix D — Suite hygiene

**Source files:** `data_eval/track1_{router,json,extraction}.jsonl`; leakage reference `kaggle_upload/training_data/dpo_pytorch.jsonl`; degeneracy floors `kaggle_upload/audit_eval.py:91`; gold self-consistency via `kaggle_upload/scorer.py` (gold-as-output).

| Property | extraction | json | router |
|----------|------------|------|--------|
| n items | 160 | 160 | 160 |
| Unique prompts | 160 | 160 | 160 |
| Duplicate prompts | 0 | 0 | 0 |
| Verbatim leakage (eval∩train) | 0 | 0 | 0 |
| Gold self-consistency (gold-as-output PASS) | 160/160 | 160/160 | 160/160 |
| Degeneracy floor (always-one-answer) | ~0.006 (1/160) | 1/160 | 0.20 (1/5 classes) |

**Class balance (items per class):**

- **router** (5 classes): COMPANY_DEEP_DIVE=32, FILING_ANALYSIS=32, FILING_DIFF=32, OUT_OF_SCOPE=32, SIMPLE_QUERY=32.
- **json** (8 tool classes): analyst_ratings=20, company_metrics=20, company_news=20, diff_filing_sections=20, get_filing_section=20, insider_trades=20, search_filings=20, stock_price=20.
- **extraction** (6 section classes): Item 1=27, Item 1A=27, Item 3=26, Item 7=27, Item 7A=26, Item 8=27.

**Factual statements:**

- Zero verbatim leakage: 0/160 eval prompts appear as a training `question` (exact-string match) for all three tasks. Source: `kaggle_upload/audit_eval.py:77` logic, `dpo_pytorch.jsonl` vs `data_eval/*.jsonl`.
- Gold self-consistency: 160/160 gold answers pass their own scorer for all three tasks. Source: `kaggle_upload/scorer.py`.
- Zero duplicate prompts: 160 unique prompts / 160 items for all three tasks. Source: `data_eval/*.jsonl`.
- Degeneracy floors (a constant always-one-answer policy): extraction/json ~1/160≈0.006; router 0.20 (5 balanced classes). Source: `kaggle_upload/audit_eval.py:91`.

## Appendix F — Per-seed reward decomposition

Source: `kaggle_result/reward_decomposition_kernel/reward_decomposition/summary.json` (β=0.5, reference=sft_baseline). r = policy_logp − sft_logp (length-normalized); margin = β·(r_chosen − r_rejected).

### F.1 — r_chosen, r_rejected, margin per checkpoint

| Seed | Ckpt | r_chosen | r_rejected | margin |
|------|------|----------|------------|--------|
| 0 | margin1 | -0.044 | -2.087 | 1.021 |
| 0 | margin3 | -0.094 | -5.569 | 2.738 |
| 0 | margin5 | -0.001 | -11.016 | 5.508 |
| 0 | final | 0.051 | -14.518 | 7.285 |
| 1 | margin1 | -0.082 | -2.365 | 1.141 |
| 1 | margin3 | -0.060 | -6.380 | 3.160 |
| 1 | margin5 | 0.006 | -10.146 | 5.076 |
| 1 | final | 0.055 | -15.344 | 7.699 |
| 2 | margin1 | -0.039 | -2.075 | 1.018 |
| 2 | margin3 | -0.063 | -5.714 | 2.825 |
| 2 | margin5 | -0.033 | -10.861 | 5.414 |
| 2 | final | 0.052 | -15.909 | 7.981 |
| 3 | margin1 | -0.045 | -2.024 | 0.990 |
| 3 | margin3 | -0.052 | -5.822 | 2.885 |
| 3 | margin5 | -0.033 | -9.320 | 4.644 |
| 3 | final | 0.060 | -14.615 | 7.337 |
| 4 | margin1 | -0.054 | -2.285 | 1.116 |
| 4 | margin3 | -0.074 | -6.235 | 3.080 |
| 4 | margin5 | 0.013 | -8.973 | 4.493 |
| 4 | final | 0.054 | -12.302 | 6.178 |
| 5 | margin1 | -0.047 | -2.014 | 0.984 |
| 5 | margin3 | -0.100 | -6.196 | 3.048 |
| 5 | margin5 | 0.018 | -9.233 | 4.626 |
| 5 | final | 0.059 | -12.631 | 6.345 |
| 6 | margin1 | -0.051 | -2.005 | 0.977 |
| 6 | margin3 | -0.075 | -5.799 | 2.862 |
| 6 | margin5 | -0.023 | -11.535 | 5.756 |
| 6 | final | 0.057 | -13.346 | 6.701 |
| 7 | margin1 | -0.051 | -2.052 | 1.001 |
| 7 | margin3 | -0.057 | -6.206 | 3.074 |
| 7 | margin5 | 0.020 | -10.166 | 5.093 |
| 7 | final | 0.032 | -14.890 | 7.461 |

### F.2 — Per-axis margin at final checkpoint (style vs tool_calling)

| Seed | style r_chosen | style r_rejected | style margin | tool r_chosen | tool r_rejected | tool margin |
|------|----------------|------------------|--------------|---------------|-----------------|-------------|
| 0 | 0.023 | -16.560 | 8.291 | 0.113 | -10.005 | 5.059 |
| 1 | 0.013 | -17.726 | 8.870 | 0.147 | -10.077 | 5.112 |
| 2 | 0.013 | -18.820 | 9.417 | 0.139 | -9.475 | 4.807 |
| 3 | 0.019 | -16.696 | 8.358 | 0.149 | -10.016 | 5.082 |
| 4 | 0.013 | -13.536 | 6.774 | 0.147 | -9.572 | 4.860 |
| 5 | 0.018 | -14.142 | 7.080 | 0.149 | -9.292 | 4.720 |
| 6 | 0.016 | -15.118 | 7.567 | 0.148 | -9.428 | 4.788 |
| 7 | -0.019 | -17.206 | 8.594 | 0.144 | -9.770 | 4.957 |

### F.3 — Held-out margin by category per checkpoint

Source: `analysis_ws/supplementary/heldout_margin_per_category.json` (n_pairs=98 held-out, β=0.5, reference=sft_baseline).

| Seed | Ckpt | style margin | tool_calling margin |
|------|------|--------------|---------------------|
| 0 | margin1 | 0.650 | 1.301 |
| 0 | margin3 | 1.738 | 2.664 |
| 0 | margin5 | 2.480 | 3.253 |
| 0 | final | 2.501 | 4.034 |
| 1 | margin1 | 0.698 | 1.484 |
| 1 | margin3 | 1.843 | 3.069 |
| 1 | margin5 | 2.241 | 3.383 |
| 1 | final | 2.476 | 3.905 |
| 2 | margin1 | 0.615 | 1.361 |
| 2 | margin3 | 1.729 | 2.961 |
| 2 | margin5 | 2.498 | 3.285 |
| 2 | final | 1.904 | 3.709 |
| 3 | margin1 | 0.671 | 1.189 |
| 3 | margin3 | 1.987 | 2.743 |
| 3 | margin5 | 2.989 | 3.184 |
| 3 | final | 2.230 | 4.002 |
| 4 | margin1 | 0.721 | 1.411 |
| 4 | margin3 | 1.882 | 2.895 |
| 4 | margin5 | 2.044 | 3.270 |
| 4 | final | 2.115 | 3.856 |
| 5 | margin1 | 0.595 | 1.230 |
| 5 | margin3 | 1.977 | 2.782 |
| 5 | margin5 | 2.225 | 3.230 |
| 5 | final | 1.998 | 3.814 |
| 6 | margin1 | 0.680 | 0.952 |
| 6 | margin3 | 1.859 | 2.777 |
| 6 | margin5 | 2.568 | 3.297 |
| 6 | final | 1.998 | 3.739 |
| 7 | margin1 | 0.706 | 1.077 |
| 7 | margin3 | 2.055 | 2.913 |
| 7 | margin5 | 2.490 | 3.505 |
| 7 | final | 3.354 | 3.908 |

## Appendix G.1 — fp16 Δgold per-seed

Source: `analysis_ws/gold_3task/gold_3task_summary.json` (fp16_mac, reference=sft_baseline). Δgold = gold(final) − gold(sft), per-token length-normalized. Bracketed values are per-seed bootstrap 95% CI.

| Seed | Extraction Δgold [95% CI] | Json Δgold [95% CI] | Router Δgold [95% CI] |
|------|---------------------------|---------------------|------------------------|
| 0 | -0.102 [-0.108, -0.095] | -1.505 [-1.581, -1.430] | -1.259 [-1.645, -0.902] |
| 1 | -0.226 [-0.232, -0.220] | -1.177 [-1.281, -1.074] | -1.616 [-2.029, -1.226] |
| 2 | -0.147 [-0.153, -0.141] | -1.001 [-1.078, -0.925] | -1.458 [-1.903, -1.044] |
| 3 | -0.102 [-0.108, -0.096] | -0.957 [-1.057, -0.858] | -1.305 [-1.723, -0.919] |
| 4 | -0.150 [-0.157, -0.143] | -0.906 [-1.001, -0.812] | -1.915 [-2.406, -1.451] |
| 5 | -0.194 [-0.200, -0.188] | -0.761 [-0.854, -0.671] | -1.307 [-1.692, -0.939] |
| 6 | -0.138 [-0.145, -0.131] | -1.175 [-1.252, -1.101] | -1.377 [-1.749, -1.028] |
| 7 | -0.168 [-0.177, -0.160] | -0.803 [-0.904, -0.703] | -1.539 [-1.999, -1.109] |

Across-seed means (8 seeds): extraction -0.153 [-0.182, -0.126], json -1.036 [-1.204, -0.891], router -1.472 [-1.624, -1.345].

## Appendix G.3 — GOLD_PROTECTED per-seed (fp16 paired stat)

Source: `analysis_ws/gold_controls/gold_controls_summary.json` (method: fp16/bf16 unquantized, paired stat Dgold-Dmean(A,B), reference=sft_baseline, X=0.05). stat = Δgold − mean(ΔctrlA, ΔctrlB); GOLD_PROTECTED verdict when stat ≥ X.

| Seed | stat | 95% CI | Δgold | Δctrl (mean A,B) | Verdict |
|------|------|--------|-------|------------------|---------|
| 0 | 0.2561 | [0.2480, 0.2644] | -0.1016 | -0.3576 | GOLD_PROTECTED |
| 1 | 0.2425 | [0.2348, 0.2502] | -0.2262 | -0.4687 | GOLD_PROTECTED |
| 2 | 0.2125 | [0.2053, 0.2198] | -0.1470 | -0.3595 | GOLD_PROTECTED |
| 3 | 0.2389 | [0.2316, 0.2462] | -0.1022 | -0.3411 | GOLD_PROTECTED |
| 4 | 0.2953 | [0.2870, 0.3038] | -0.1502 | -0.4455 | GOLD_PROTECTED |
| 5 | 0.2172 | [0.2108, 0.2236] | -0.1938 | -0.4110 | GOLD_PROTECTED |
| 6 | 0.2477 | [0.2403, 0.2553] | -0.1379 | -0.3856 | GOLD_PROTECTED |
| 7 | 0.2666 | [0.2574, 0.2756] | -0.1684 | -0.4350 | GOLD_PROTECTED |

Aggregate: 8/8 seeds GOLD_PROTECTED.

## Appendix H — Val margin trajectories

Source: `kaggle_result/final/seed*/logs/track1_dpo_seed*_margin.jsonl`.

> **Availability note:** `val_margin` is logged only at the evaluation cadence (`eval_every` = 50 → steps **50, 100, 150**). It is **not** recorded at the checkpoint steps (step20, margin1/3/5 steps, or final step 180). The table below reports every logged val_margin step; checkpoint-step val_margin is unavailable in the saved logs.

| Seed | val@50 | val@100 | val@150 | (margin1 step) | (margin3 step) | (margin5 step) |
|------|--------|---------|---------|----------------|----------------|----------------|
| 0 | 1.701 | 3.534 | 5.752 | 27 | 84 | 134 |
| 1 | 1.648 | 3.091 | 5.904 | 29 | 93 | 131 |
| 2 | 1.685 | 3.618 | 5.746 | 26 | 87 | 124 |
| 3 | 1.660 | 3.506 | 7.397 | 28 | 90 | 112 |
| 4 | 1.701 | 3.483 | 6.449 | 30 | 91 | 130 |
| 5 | 1.804 | 3.502 | 4.968 | 24 | 87 | 135 |
| 6 | 1.743 | 3.522 | 6.709 | 27 | 88 | 138 |
| 7 | 1.528 | 3.288 | 5.768 | 26 | 98 | 135 |

**8-seed mean val_margin at step 150 = 6.087** (§4 references ~5.9 — see value).

