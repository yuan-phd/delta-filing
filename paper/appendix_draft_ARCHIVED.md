# Appendices

## Appendix A: Full Per-Seed TOST Data

*[CC: Generate from matrix results. Table format below.]*

For each task (extraction, json, router), report semantic accuracy at each checkpoint (step20, margin1, margin3, margin5, final) for all 8 seeds, plus the baseline\_v2 value. Include per-seed Δ from baseline at the final checkpoint. This is the complete data behind Table 1 in the main text.

| Task | Seed | step20 | margin1 | margin3 | margin5 | final | Δ (final − baseline) |
|---|---|---|---|---|---|---|---|
| extraction | 0 | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

Also report the per-seed TOST pass/fail (whether |Δ| ≤ 5) and the item-level bootstrap 90% CI for each seed × task at the final checkpoint.

---

## Appendix B: Scorer v2 Rules and Artifact Walkthrough

### Scorer architecture

Scorer v2 applies a deterministic three-tier classification to each generated output:

1. **FORMAT\_BREAK** — output cannot be parsed by the task-specific format gate. For router, the gate checks whether the output is a bare string matching one of the five valid labels; for json and extraction, whether the output parses as valid JSON with required keys.
2. **PASS** — parsed output matches the gold answer after canonicalization.
3. **SURFACE\_FORM** — parsed output contains the correct semantic content but differs in surface form (e.g., "Item 1" vs "1" for section identifiers).
4. **SEMANTIC\_ERROR** — parsed, well-formed output with incorrect content.

The primary metric (semantic accuracy) counts PASS + SURFACE\_FORM as correct. FORMAT\_BREAK and SEMANTIC\_ERROR are failures.

### Canonicalization rules (extraction)

The section-identifier field is canonicalized before comparison:
- Extract numeric portion via regex `\d+[A-Z]?` (e.g., "Item 7A" → "7A", "1" → "1", "Item 1" → "1")
- Case-fold the alpha suffix
- Ignore appended titles (e.g., "Item 7 — Management's Discussion" → "7")

Ticker and filing\_date fields use exact string match (case-insensitive for ticker).

### The artifact: 0.550 → 0.969

At the margin3 checkpoint, extraction accuracy under strict string matching (no canonicalization) is 0.550. Under scorer v2's canonicalization, the same checkpoint scores 0.969.

The 67 items that flip from FAIL to PASS are exclusively section-identifier surface-form variants: the model outputs "1" where gold says "Item 1", or "7A" where gold says "Item 7A". The semantic content (which section of which filing) is correct in every case. No ticker or filing\_date errors are involved.

This is not a marginal scorer adjustment — it is the difference between "catastrophic capability collapse at margin ≈ 3" and "capability preserved with a transient register shift." The register-shift rate (SURFACE\_FORM tier) peaks at margin ≈ 3 (mean 0.37 across 8 seeds) then subsides to 0.09 by final, mirroring router's FORMAT\_BREAK non-monotonic trajectory.

The canonicalization rules were frozen before the confirmatory matrix. No rules were added or modified after observing matrix results.

---

## Appendix C: Pre-Registration Reconciliation and Falsification Chronology

### C.1 Pre-registered secondary reconciliation

Every pre-registered secondary from Note 2 (frozen 2026-06-18) is accounted for below with its final status.

| # | Pre-registered secondary | Status | Location |
|---|---|---|---|
| 1 | Trajectory-min per seed/task | ✅ Reported | §4.2 (router trough at margin3, 8/8 seeds) |
| 2 | FORMAT\_BREAK trajectory | ✅ Reported | §4.2 (Table 2, non-monotonic) |
| 3 | Held-out margin per category (style vs tool\_calling) | ✅ Reported — **prediction falsified** | §4.3 (Table 5, tool > style at all checkpoints, all seeds) |
| 4 | Chosen/rejected reward decomposition | ✅ Reported (filled post-hoc via dedicated Kaggle GPU inference run) | §4.3 (Tables 3–4) |
| 5 | Gold-logp trajectory × 3 tasks | ✅ Reported (fp16 same-backend, 8 seeds, Mac) | §4.4 (Table 6) |
| 6 | Probe statistic replication (paired) | ✅ Extraction 8/8 GOLD\_PROTECTED (fp16 Mac); json/router: paired stat not computable (controls are extraction-specific) | §4.4 (Table 7) |
| 7 | Register-shift tracking | ✅ Reported (SURFACE\_FORM rate from existing matrix results) | §4.2 |
| 8 | KL-from-reference per checkpoint | ✅ Reported (computed post-hoc from saved adapters, fp16 Mac, 8 seeds) | §4.4 (Appendix G) |

Implementation notes: Secondaries #4 and #8 were not logged during the matrix training run. #4 was computed via a dedicated GPU inference run on saved adapters (Kaggle T4); #8 was computed via Mac inference on saved adapters (fp16, proper token-level KL using `forward_full` + `kl_over_positions`). #3 was computed via Mac inference on saved adapters (fp16). All three use the same checkpoint adapters as the matrix; no retraining was involved. The primary TOST analysis (§4.1) was unaffected by these post-hoc computations.

### C.2 Pre-registration deviations

**Checkpoint mechanism.** The pre-registered design (Note 2) specified fixed-step checkpoints with margin as a post-hoc label. The implementation used margin-crossing triggers for intermediate checkpoints (margin1, margin3, margin5), so these checkpoints occur at different optimizer steps across seeds (margin1: steps 24–30, margin3: steps 84–98, margin5: steps 112–138). Step20 and final (step 180) are fixed and identical across seeds. The primary endpoint (step 180) is unaffected. Cross-seed averages at intermediate checkpoints pool seeds at matched margin but different optimizer steps.

**Early-stop configuration.** The pre-registered design specified `max_opt_steps=180` and `no_early_stop=True`. The runtime enforcement was applied in `matrix_kernel.py` (passing `--max-opt-steps 180`) and `run_matrix.py` (passing `no_early_stop=True`), and all 8 seeds terminated at exactly step 180. However, `train_config.json` does not record these runtime parameters (it records module-level constants only), which may create a misleading impression when inspecting saved artifacts. The execution matched the pre-registration; the config-file serialization did not.

### C.3 Falsification chronology

**Design phase (before confirmatory matrix):**
- **Causal-clamp experiment dropped.** An experiment to clamp the margin at a fixed value and measure capability was designed but dropped as confounded — clamping the margin also alters the KL penalty and total optimization pressure, making the causal attribution "margin magnitude → capability loss" untestable.
- **Router-as-primary dropped.** Router was initially the primary endpoint. Pilot trajectory analysis revealed a non-monotonic transient (the margin ≈ 3 format-break spike), making the endpoint accuracy an unreliable summary of the trajectory. Router was moved to a secondary with trajectory decomposition.
- **Targeted-displacement hypothesis dropped.** The A1 register probe was designed to test whether DPO specifically displaces gold probability mass onto incorrect alternatives (targeted displacement). The probe found that controls fell more than gold (global drift, not targeted displacement). The hypothesis was reformulated as "selective deflation" — gold declines less than controls — rather than "gold doesn't decline."
- **Dissolved numbers.** Three numbers from the exploration phase were found to lack on-disk artifacts and were re-measured or dropped: (a) 80-item baseline accuracies (never saved; re-derived from the 160-item suite); (b) router baseline 0.950 (traced to a conflation between DPO json accuracy and router baseline; corrected to 0.938); (c) 160-item gate verification (floor values unpersisted; re-derived from source files: extraction 0.000, router 0.200, json 0.006).
- **80 → 160 suite expansion.** The evaluation suite was expanded from 80 to 160 items per task between the pilot and confirmatory phases to improve TOST power. The 160-item suite was frozen before the matrix.
- **X = 0.05 threshold.** The GOLD\_PROTECTED threshold (X = 0.05 nats/token) was frozen in the decision-mapping note before cc's n = 12 preliminary probe existed (timestamp-verifiable from commit history).

**Matrix phase (confirmatory runs):**
- **Falsified: bounded-null headline.** The pre-registered falsifier fired when json and router both failed the ±5 TOST equivalence test. The task-selective branch governed all subsequent analysis. (§4.1)
- **Falsified: extraction collapse at margin ≈ 3.** Extraction accuracy appeared to crash to 0.550 at margin3 under the strict scorer. Canonicalized scoring revealed this was a surface-form register shift, not a semantic failure (0.550 → 0.969). The strict-scorer measurement was killed, not refined. (§3.2, §4.2)
- **Falsified: style margin > tool\_calling on held-out.** The pre-registered prediction that per-pair held-out margin would be larger for style than tool\_calling was falsified at every checkpoint on every seed (0/4 × 0/8). Tool\_calling pairs are individually more potent; style dominance is an aggregate-volume effect. (§4.3)
- **Refined: margin-independent drop → damage window.** The pilot observation of a "margin-independent early drop" on router was refined to a damage window at margin ≈ 1–3 with two distinct processes: transient format-break (non-monotonic) and persistent content-error (monotonic). (§4.2)
- **Refined: gold absolutely stable → gold task-selectively deflates.** The initial characterization from nf4 matrix data (|Δgold| ≤ 0.01 on extraction) was refined when fp16 same-backend measurement revealed substantial gold deflation on json (−1.04) and router (−1.47), with extraction deflating least (−0.15). Gold protection is real on extraction (8/8 GOLD\_PROTECTED) but gold deflation is task-selective and non-linearly related to behavioral outcomes. (§4.4)

---

## Appendix D: Suite Hygiene

*[CC: Generate from source files and hygiene audit results.]*

Report the following, all re-derived from source files (not from memory or prior reports):

**Degeneracy floors** (measured: accuracy of one fixed constant output against all 160 golds):
- Extraction: 0.000 (constant output matches no gold exactly)
- Json: 0.006 (1/160; the constant `search_filings(ticker="AAPL")` matches one gold)
- Router: 0.200 (1/5 uniform over 5 balanced classes)

**Class balance:**
- Router: 32 items × 5 classes (FILING\_ANALYSIS, FILING\_DIFF, COMPANY\_DEEP\_DIVE, SIMPLE\_QUERY, OUT\_OF\_SCOPE)
- Json: 20 items × 8 tool types

**Leakage audit:** Zero verbatim prompt overlap between eval suite and preference/SFT training data (exact-string match on full prompt text).

**Gold self-consistency:** 160/160 per task (gold completions score PASS under scorer v2 on all items).

**Duplicate check:** Zero duplicate prompts within each task's 160-item set.

---

## Appendix E: Training Configuration and Hash Verification

**SFT checkpoint:** hash 29a3794b…585f02, verified identical across all 8 seed runs. Adapter weights loaded from `adapters/sft_pytorch/adapter_model.safetensors`.

**Tokenizer and chat template:** byte-identical across all sources (SFT adapter, base model, eval harness). Verified by SHA-256 hash comparison.

**LoRA configuration:**
- r = 8, α = 16, dropout = 0.05
- target\_modules = [q\_proj, k\_proj, v\_proj, o\_proj] (explicitly configured, not peft default)
- bias = none
- peft version 0.19.1

**Training hyperparameters:**
- Optimizer: AdamW, lr = 5 × 10⁻⁵, constant schedule, weight\_decay = 0.01
- Gradient clipping: 1.0
- Batch size: 8 (via gradient accumulation)
- β (DPO): 0.5
- SFT retention weight: 0.5
- Max optimizer steps: 180 (enforced via `matrix_kernel.py` runtime parameter)
- Early stop: disabled (`no_early_stop=True` passed at runtime via `run_matrix.py`)
- Divergence guard: abort if NaN or smoothed margin > 20

**Config-file note:** `train_config.json` records module-level constants and does not serialize the runtime parameters `max_opt_steps` and `no_early_stop`. The runtime enforcement of `max_opt_steps=180` and `no_early_stop=True` was applied in `matrix_kernel.py` and `run_matrix.py`, as confirmed by source-code inspection. All 8 seeds terminated at exactly step 180.

**Checkpoint adapters on disk:** 8 seeds × 5 checkpoints (step20, margin1, margin3, margin5, final) = 40 adapter directories, all verified present with complete `adapter_model.safetensors` files.

---

## Appendix F: Reward Decomposition Per-Seed

*[CC: Generate from reward decomposition output files in `analysis_ws/reward_decomposition/`.]*

Report per-seed chosen reward, rejected reward, and margin at each checkpoint (margin1, margin3, margin5, final) for all 8 seeds. Include the per-axis (style vs tool\_calling) breakdown at the final checkpoint, per seed.

Also report per-seed held-out margin by category (style vs tool\_calling) at each checkpoint — the data behind Table 5 in the main text.

---

## Appendix G: Gold-Logp Per-Seed Data, nf4 Cross-Check, and KL Trajectory

### G.1 fp16 same-backend Δgold (per-seed)

*[CC: Generate from `analysis_ws/gold_3task/gold_3task_summary.json`.]*

Report per-seed Δgold for all three tasks (extraction, json, router) at the final checkpoint, with per-seed bootstrap 95% CI. This is the complete data behind Table 6 in the main text.

### G.2 nf4 cross-check (moved from main text)

nf4 Δgold is computed as gold\_logp(final) − gold\_logp(step20), using the matrix's own gold-logp records. No nf4 SFT baseline gold-logp was recorded; step20 is the earliest available reference.

**[Table: nf4 Δgold (step20→final), 8-seed mean]**

| Task | Δgold (nf4) | 95% CI | Δgold (fp16) for comparison |
|---|---|---|---|
| Extraction | −0.008 | [−0.014, −0.003] | −0.15 |
| Json | −0.22 | [−0.34, −0.11] | −1.04 |
| Router | −1.59 | [−1.99, −1.19] | −1.47 |

The task-selective ordering (extraction ≪ json < router) is identical under both backends, with non-overlapping CIs. Absolute magnitudes diverge, reflecting both the quantization difference and the different baselines (fp16 uses the pure SFT adapter; nf4 uses step20, which has absorbed 20 DPO steps). The divergence is itself task-selective: extraction's deflation nearly vanishes under nf4 (~18× compression), json compresses ~5×, while router's magnitude is essentially unchanged.

### G.3 GOLD\_PROTECTED paired statistic (fp16, extraction, per-seed)

*[CC: Generate from `analysis_ws/gold_controls/` output files.]*

Report per-seed paired statistic, Δgold, Δctrl(A,B), and 95% bootstrap CI for all 8 seeds. This is the complete data behind Table 7 in the main text.

### G.4 KL(policy || reference) trajectory (moved from main text)

**[Table: KL trajectory (extraction gold, fp16, 8-seed mean ± std)]**

| Checkpoint | Mean KL | Std |
|---|---|---|
| margin1 | 0.022 | 0.005 |
| margin3 | 0.057 | 0.017 |
| margin5 | 0.083 | 0.014 |
| final | 0.082 | 0.008 |

KL rises monotonically through margin5 then plateaus (0.083 → 0.082), confirming mild policy drift (all values < 0.1 nats/token) that saturates as margin continues to grow from ~5 to ~7. KL is computed on extraction gold completions only; drift on other tasks may be larger.

Per-seed KL values at final: seed0 = 0.071, seed1 = 0.089, seed2 = 0.081, seed3 = 0.075, seed4 = 0.077, seed5 = 0.092, seed6 = 0.077, seed7 = 0.093.

---

## Appendix H: Validation Margin Trajectories

*[CC: Generate from margin log files (`*_margin.jsonl`, val\_margin field).]*

Report per-seed validation margin at selected optimizer steps (every 10 steps, or at checkpoint steps) for all 8 seeds. Show that validation margin tracks training margin throughout, confirming that margin growth reflects generalized preference learning rather than training-set memorization. Report the mean validation margin at step 150 across 8 seeds (referenced in §4's opening paragraph as ~6.1).
