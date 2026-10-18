# §4 Results (revised)

All 8 seeds completed 180 fixed optimizer steps without triggering the divergence guard. Final margins range from 6.08 to 7.68 (mean 6.73; β = 0.5, corresponding to ~13.5 nats/token unscaled preferred-vs-rejected differential). Validation margin tracks training margin on every seed (mean val margin at step 150: 6.1; Appendix H), confirming that margin growth reflects generalized preference learning rather than training-set memorization.

## 4.1 Graded, Task-Selective Degradation

Table 1 reports the primary TOST equivalence verdicts. The pre-registered falsifier fires: router degrades far beyond the ±5-point bound, and the bounded-null headline is dropped. The task-selective branch governs.

**[Table 1: Primary TOST equivalence test (±5 pts, semantic accuracy at step 180 vs baseline_v2)]**

| Task | Baseline | Mean final | Δ | Seed-level 90% CI | Item-level 90% CI | Seeds ≤ ±5 | Verdict |
|---|---|---|---|---|---|---|---|
| Extraction | 97.50 | 96.95 | −0.55 | [−2.07, +0.98] | [−1.80, +0.86] | 8/8 | Equivalent |
| Json | 98.75 | 95.23 | −3.52 | [−4.56, −2.47] | [−5.31, −1.95] | 6/8 | Not equivalent |
| Router | 93.75 | 63.59 | −30.16 | [−35.65, −24.66] | [−34.92, −25.47] | 0/8 | Not equivalent |

Extraction passes all three criteria comfortably: the point estimate is −0.55, both confidence intervals lie well within ±5, and all 8 seeds fall within the bound. A per-item flip analysis finds no systematic asymmetry (15 gains vs 22 losses, p = 0.32). Extraction capability is preserved as the margin grows to ~7.

Json fails the equivalence test, though the failure is marginal and ceiling-driven. The point estimate (−3.52) and the seed-level CI are within ±5, but the item-level pooled CI lower bound (−5.31) crosses the −5 threshold, and only 6 of 8 seeds fall within the bound (seed0 at −6.25 is clearly outside; seed4 at −5.00 sits exactly at the boundary). The per-item flip analysis confirms the drop is not noise: across all 8 seeds and 160 items, there are 0 gains and 45 losses (p = 5.4 × 10⁻¹¹). Every json flip is a regression, never offset by a compensating gain. The degradation is statistically genuine, one-directional, and consistent — but small in absolute magnitude. The TOST failure is partly an artifact of testing near the ceiling (baseline 98.75 leaves minimal room), but the underlying drop is real.

Router degrades catastrophically: −30.16 points, with both confidence intervals far outside ±5 and 0 of 8 seeds within the bound. The endpoint distribution is approximately symmetric — mean 0.636, median 0.638, range 0.487 to 0.769 — but the spread (0.282) is wide, reflecting high seed-to-seed variance in outcomes despite consistent damage depth (§4.2).

The alignment tax is real, graded, and task-selective: it ranges from undetectable (extraction, Δ = −0.55) through small but genuine (json, Δ = −3.52) to catastrophic (router, Δ = −30.16). The gradient of degradation is directionally consistent with the task's format-alignment to the preference data across all 8 seeds — the task whose terse enum format is most orthogonal to the preference data's dominant verbose style axis (router) degrades most; the task with matching tool_calling coverage (json) degrades modestly; and the task with an orthogonal but non-antagonistic format (extraction) is preserved. However, with only three tasks, we report this co-occurrence without claiming the causal link is established. Extraction also lacks direct preference coverage yet is preserved, so coverage absence alone does not predict degradation.

Json has 200 matching tool_calling preference pairs, yet still shows degradation. The per-axis training dynamics (§4.3) explain why: tool_calling coverage provides partial but not complete protection against the dominant style-axis aggregate pressure.

## 4.2 Non-Monotonic Dynamics with Two Distinct Processes

Router's trajectory across checkpoints reveals that the degradation is not monotonic in margin. Table 2 decomposes router accuracy into its two failure modes — format breaks (malformed output) and content errors (well-formed but incorrect) — at each checkpoint.

**[Table 2: Router trajectory decomposition (mean ± std over 8 seeds)]**

| Checkpoint | Margin | Semantic acc | FORMAT_BREAK rate | Content-error rate |
|---|---|---|---|---|
| step20 | 0.75 | 0.852 ± 0.015 | 0.001 ± 0.002 | 0.148 ± 0.016 |
| margin1 | 1.09 | 0.816 ± 0.030 | 0.007 ± 0.010 | 0.177 ± 0.038 |
| margin3 | 3.12 | 0.587 ± 0.061 | 0.196 ± 0.089 | 0.217 ± 0.071 |
| margin5 | 5.13 | 0.626 ± 0.049 | 0.133 ± 0.096 | 0.241 ± 0.083 |
| final | 6.73 | 0.636 ± 0.077 | 0.106 ± 0.070 | 0.258 ± 0.074 |

Two processes with different dynamics are visible. FORMAT_BREAK rate is non-monotonic: it peaks at margin ≈ 3 (0.196), then partly subsides by the final checkpoint (0.106). Content-error rate rises monotonically from 0.148 to 0.258, never recovering. The apparent partial recovery in overall accuracy between margin3 and the final checkpoint is driven by format-break subsiding, not by content correctness improving.

Format breaks at margin ≈ 3 consistently take the form of invented UPPER\_SNAKE\_CASE pseudo-labels derived from query content (e.g., TOTAL\_REVENUES\_2023\_Q2, REPORT\_PER\_ANALYSIS). Notably, the valid router labels are also UPPER\_SNAKE\_CASE (FILING\_ANALYSIS, COMPANY\_DEEP\_DIVE, etc.), so the model generates in the correct register but invents labels outside the valid 5-category set — a label-vocabulary interference specific to the preference data's financial domain, rather than degeneration, truncation, or style bleed. This pattern replicates across matrix seeds.

The margin ≈ 3 instability is not limited to router. On extraction, the surface-form register-shift rate (SURFACE_FORM tier in scorer v2) mirrors the non-monotonic shape: it spikes at margin ≈ 3 (mean 0.37, range 0.03–0.77 across seeds) then partly subsides to 0.09 by final, confirming that the margin 1–3 damage window produces surface-form instability across tasks. Under the strict scorer, this register shift would have been counted as a capability collapse (extraction accuracy 0.550 at margin3); under scorer v2's canonicalization, the same checkpoint scores 0.969 — 93% of apparent errors are surface-form shifts ("Item 1" rendered as "1"), not semantic failures. This is the paper's most concrete demonstration of why artifact-aware scoring is necessary.

Router accuracy reaches its trajectory minimum at the margin3 checkpoint on all 8 seeds (mean 0.587, range 0.444 to 0.650). The resolution of our checkpoint schedule (margin1, margin3, margin5, final) cannot pinpoint the trough more precisely; the damage accumulates between margin ~1 and ~3 and stabilizes thereafter. Despite the consistent trough depth, the degree of subsequent recovery is highly seed-dependent: endpoint accuracy ranges from 0.487 (seed5, which shows near-zero recovery) to 0.769 (seed1). The per-seed format-break rate at the final checkpoint ranges from 0.019 to 0.256, indicating that the mix of failure modes is itself stochastic across seeds. The damage is reproducible; the recovery is stochastic.

Figure 1 shows per-seed margin-vs-capability trajectories for all three tasks. Extraction and json form tight bands near the baseline, while router forms a wide fan — the variance structure itself communicates the finding.

## 4.3 Mechanism: Rejected Collapse and Style-Axis Dominance

To understand the mechanism driving margin growth, we decompose the implicit reward (policy log-prob minus reference log-prob) into its chosen and rejected components at each checkpoint. Table 3 reports the 8-seed mean.

**[Table 3: Reward decomposition (mean over 8 seeds; reward = policy logp − SFT logp)]**

| Checkpoint | r_chosen | r_rejected | Margin (β = 0.5) |
|---|---|---|---|
| margin1 | −0.052 | −2.113 | +1.031 |
| margin3 | −0.072 | −5.990 | +2.959 |
| margin5 | −0.004 | −10.156 | +5.076 |
| final | +0.052 | −14.194 | +7.123 |

Chosen reward is approximately stationary throughout training (range −0.072 to +0.052), consistent with the SFT retention term (weight 0.5) explicitly anchoring chosen-completion likelihood. Rejected reward falls monotonically from −2.1 to −14.2. The full 7-unit margin growth is accounted for by rejected-completion collapse alone. The retention term protects chosen but provides no protection to rejected, so DPO's gradient pressure is channeled entirely into rejected suppression. Cal-DPO adds a calibration term to prevent chosen-reward decline; our regime includes an SFT retention term that achieves a similar anchoring effect. The more informative observation is that even with chosen explicitly anchored, the behavioral alignment tax remains: router degrades −30 points and json shows one-directional loss, driven entirely by the rejected-suppression channel that the retention term does not protect. Rafailov et al. predict that chosen log-prob should decrease under vanilla DPO from an SFT-initialized model; our training objective includes an SFT retention term on chosen completions, which directly counteracts this decline. The chosen-flatness we observe is consistent with this design choice rather than a contradiction of their framework.

The per-axis decomposition reveals which preference dimension dominates the optimization. The relevant quantity for behavioral impact is aggregate suppression pressure — the total margin contributed by each axis — not per-pair potency, because weight updates accumulate over all pairs in the training set. Table 4 reports the decomposition at the final checkpoint.

**[Table 4: Per-axis reward decomposition at final checkpoint (mean over 8 seeds)]**

| Axis | Pairs | Aggregate margin | r_chosen | r_rejected |
|---|---|---|---|---|
| Style | 447 | +8.119 | +0.012 | −16.226 |
| Tool_calling | 200 | +4.923 | +0.142 | −9.704 |

The style axis produces 65% more total margin than the tool_calling axis, driven by 67% more rejected suppression (−16.2 vs −9.7). This aggregate dominance arises because the style axis contributes 2.2× more pairs; per-pair, tool_calling is slightly more potent (~0.025 vs ~0.018 margin/pair). The optimization pushes harder on the style dimension in total, and this aggregate pressure shapes the behavioral gradient.

This aggregate-versus-per-pair distinction is confirmed on held-out data. On 98 held-out preference pairs (~69 style, ~29 tool_calling), tool_calling margin exceeds style margin at every checkpoint on every seed (Table 5). The pre-registered prediction that style margin would exceed tool_calling margin on held-out pairs is falsified (0/4 checkpoints, 0/8 seeds).

**[Table 5: Per-category held-out margin (mean over 8 seeds; β = 0.5)]**

| Checkpoint | Style margin | Tool_calling margin | Prediction (style > tool)? |
|---|---|---|---|
| margin1 | +0.667 | +1.251 | No |
| margin3 | +1.884 | +2.851 | No |
| margin5 | +2.442 | +3.301 | No |
| final | +2.322 | +3.871 | No |

This confirms that style dominance is an aggregate-volume effect: each tool_calling pair contributes more margin and generalizes more strongly, but 2.2× fewer pairs means less total optimization pressure. The behavioral gradient tracks aggregate training pressure, not per-pair generalization strength.

## 4.4 Gold-Logp Dynamics Are Task-Selective

The reward decomposition shows that chosen completions are approximately stationary on training pairs. We now examine whether gold (correct) completions on evaluation items are similarly stable, and whether this pattern generalizes across tasks.

All Δgold values below are computed in fp16 on Mac; the adapters were trained under nf4 on T4. Adapter weights are stored in full precision regardless of base quantization, so loading nf4-trained adapters onto an fp16 base is valid. Within-backend comparisons are internally consistent; absolute magnitudes may differ from nf4 inference, as discussed below.

Table 6 reports the same-backend (fp16) gold-logp change from SFT baseline to final checkpoint across all three tasks.

**[Table 6: Same-backend fp16 Δgold (gold_logp(final) − gold_logp(SFT)), 8-seed mean]**

| Task | Δgold | 95% CI | Behavioral Δ |
|---|---|---|---|
| Extraction | −0.15 | [−0.18, −0.13] | −0.5 pts (equivalent) |
| Json | −1.04 | [−1.20, −0.89] | −3.5 pts (not equivalent) |
| Router | −1.47 | [−1.62, −1.35] | −30 pts (not equivalent) |

Gold deflation is task-selective. Extraction gold deflates modestly (−0.15 nats/token); json and router gold deflate substantially (−1.04 and −1.47 respectively). The ordering — extraction least, json intermediate, router most — is directionally consistent with the behavioral gradient across all 8 seeds, with non-overlapping confidence intervals between tasks. However, the relationship is highly non-linear: router's gold deflates only 1.4× more than json's, but its behavioral degradation is 8.6× worse. Gold-logp magnitude alone does not explain the behavioral gradient. Json's substantial gold deflation (−1.04) paired with only minor behavioral loss (−3.5 pts) suggests that structured output formats with redundant constraints (typed fields, function signatures) may maintain correct greedy-decode behavior despite large probability-mass redistribution — a task-structural buffer that router's terse single-token enum format does not provide.

The task-selective ordering is robust across quantization backends. Using the matrix's own nf4 gold-logp records (step20-to-final, as no nf4 SFT baseline was recorded), the ordering extraction ≪ json < router is identical (nf4 Δgold: extraction −0.008, json −0.22, router −1.59; non-overlapping CIs; Appendix G). Absolute magnitudes diverge substantially, reflecting both the quantization difference and the different baselines (fp16 uses the pure SFT adapter; nf4 uses step20, which has absorbed 20 DPO steps). The divergence is itself task-selective: extraction's deflation nearly vanishes under nf4 (~18× compression), json compresses ~5×, while router's magnitude is essentially unchanged. Whether this reflects quantization-dependent sensitivity, baseline differences, or an interaction between quantization and task-specific token distributions is unresolved. The task-selective ordering — the load-bearing claim — holds under both backends.

On extraction specifically, we can go further. Teacher-forcing gold completions alongside control A (correct-but-wrong-register) and control B (wrong-but-plausible) through each seed's final checkpoint in fp16 yields the paired GOLD_PROTECTED statistic: how much gold deflates relative to incorrect alternatives. Table 7 reports the results.

**[Table 7: Paired GOLD_PROTECTED statistic (fp16, extraction, 160 items per seed)]**

| Seed | Stat | 95% CI | Δgold | Δctrl(A,B) |
|---|---|---|---|---|
| 0 | +0.256 | [+0.248, +0.264] | −0.102 | −0.358 |
| 1 | +0.242 | [+0.235, +0.250] | −0.226 | −0.469 |
| 2 | +0.213 | [+0.205, +0.220] | −0.147 | −0.360 |
| 3 | +0.239 | [+0.232, +0.246] | −0.102 | −0.341 |
| 4 | +0.295 | [+0.287, +0.304] | −0.150 | −0.446 |
| 5 | +0.217 | [+0.211, +0.224] | −0.194 | −0.411 |
| 6 | +0.248 | [+0.240, +0.255] | −0.138 | −0.386 |
| 7 | +0.267 | [+0.257, +0.276] | −0.168 | −0.435 |

All 8 seeds return GOLD_PROTECTED: every stat exceeds the pre-registered threshold (X = 0.05) with confidence intervals far from zero. On extraction, gold is selectively protected — it declines modestly (−0.10 to −0.23 nats/token) while incorrect controls collapse two to three times more (−0.34 to −0.47). The relative ordering of correct over incorrect completions is preserved and strengthened.

The paired GOLD_PROTECTED statistic requires task-specific control completions; controls were designed for extraction only (control A = different section identifier; control B = different ticker). On json and router, gold deflation is reported as absolute Δgold (Table 6; nf4 values in Appendix G) without a paired control baseline. Whether gold is relatively protected on these tasks — deflating less than incorrect alternatives — is untested.

No seed shows evidence of the catastrophic CHES failure mode: control-C rejected-register text falls sharply on both probe seeds where it was measured (−4.6 to −5.1 nats/token).

Finally, KL(policy || reference) provides context for the magnitude of policy drift. Computed as proper token-level KL on extraction gold completions (160 items, fp16, 8 seeds), it rises from 0.022 at margin1 to 0.082 at final, confirming mild policy drift that plateaus as margin continues to grow from ~5 to ~7 (Appendix G). KL is computed on extraction gold completions only; drift on router, the most behaviorally affected task, may be larger.
