# Earlier ablation: five alignment methods

> **Note:** This ablation is a separate, earlier experiment from the paper. It used a different trainer and objective setup, a custom 87-test suite and a single seed. Its observations (including "Observation 1" below) are not the paper's results and are not superseded or confirmed by them. The paper studies length-normalized DPO with an assistant-only SFT retention term over 8 seeds with pre-registered TOST equivalence testing; its code and data are tagged `paper-v1`.

Back to the [README](../README.md#earlier-ablation).

All results on this page come from one seed and 87 tests. They are stated as observations, not as established effects.

To make on-device inference work at the system's quality bar, I trained and compared **five alignment methods** on real SEC 10-K data. All adapters are LoRA (rank 8, alpha 16, q/k/v/o_proj) on Qwen3.5-4B in 4-bit NF4.

## Final scores (custom 87-test evaluation suite)

| Adapter | Method | Final score | Notes |
|---------|--------|-------------|-------|
| SFT | Supervised cross-entropy | 73/87 (84%) | Baseline |
| DPO | `-logσ(β·margin)` | 65/87 (75%) | Unbounded margin; Review 1/9 |
| IPO | `(margin - 1/(2β))²` | 70/87 (80%) | Bounded margin via identity loss |
| cDPO | `-(1-ε)·logσ(m) - ε·logσ(-m)` | 69/87 (79%) | Label smoothing |
| **GRPO** | RL with k3 KL estimator | **76/87 (87%)** ★ | Used by the agent in this repository |

Only the DPO row currently has a result file in this repository, see [Recorded sources](#recorded-sources).

Six loss functions implemented from scratch: DPO, IPO, cDPO, GRPO, InfoNCE (for embedding training), SFT retention.

## Observation 1: in this run, the size of the capability loss followed whether the objective was bounded

The three preference-optimization methods (DPO, IPO, cDPO) train on **identical data** with **identical hyperparameters**; only the loss function differs. In this run their results fell in the following order:

- **DPO** (unbounded `-logσ(β·margin)`): during training, the implicit reward margin grew from 0.019 at step 10 to **8.76 at step 520**. The resulting model scored low on rigid-format tasks (Review 1/9, Refusal 3/6) while still scoring 8/8 on JSON tool parsing. This pattern is consistent with a shift in output style under the unbounded objective rather than a general loss of capability, but one seed cannot separate the two.
- **IPO** (bounded `(margin - 1/(2β))²`): the margin stayed near the target value 1.0 (= `1/(2β)` with β=0.5). It lost 3 points on tool calling and kept Review at 7/9.
- **cDPO** (label smoothing): extreme margins were partly limited; Review at 6/9.

GRPO, trained on the same data with the same LoRA rank but with an **explicit multi-objective reward function** (tool-call validity + section reference + numerical specificity + analytical judgment + format compliance), scored 15/15 on tool calling *and* the highest synthesis score (7/8 vs SFT's 5/8). In this run it showed no trade-off between those two categories.

**Interpretation**: a common framing is that *"LoRA has a small parameter budget, so alignment must trade off against base capabilities."* This run does not support the strong version of that claim: with the same LoRA rank and the same data, only the optimization signal changed, and the trade-off differed. A single seed and 87 tests are not enough to establish that the trade-off sits in the optimizer's incentive structure rather than in the parameter budget.

## Observation 2: KL-divergence implementation bug in GRPO

The initial GRPO implementation used `kl = pi_log_prob - ref_log_prob`. **This is the log-ratio, not KL.** A single-sample log-ratio can be negative whenever the policy probability drops below the reference, which happens routinely during learning. With `β · kl` added to the loss, a negative `kl` *rewards* divergence from the reference SFT model.

Training log evidence:
```
Step  5  | pg= 0.107 | kl= +0.0011
Step 10  | pg= 0.227 | kl= -0.0014   ← already negative
Step 25  | pg=-0.083 | kl= -0.0848
Step 50  | pg=-0.161 | kl= -0.8945   ← strongly negative
```

The policy drifted away from SFT and scored **66/87** (below the SFT baseline).

**Fix**: the Schulman 2020 k3 estimator:
```python
log_ratio = ref_log_prob - pi_log_prob
log_ratio = torch.clamp(log_ratio, -10, 10)
kl = torch.exp(log_ratio) - 1 - log_ratio
```

k3 is always non-negative (a property of the exponential's Taylor expansion) and unbiased.

After the fix, **KL stayed in [0, 0.0006]** in the logged training steps. With the same data, hyperparameters and reward function, the final score was **76/87, a +10 difference**. The KL fix was the only change between the two runs.

## Observation 3: continuous metrics showed a difference that binary pass/fail did not

The Red flag category includes *false-positive tests*, which ask the model to find red flags in **healthy companies** (JNJ, PG, V). All adapters fail binary thresholds, so we measured **alarm word count** instead:

| Company | SFT | DPO | IPO | cDPO | GRPO |
|---------|-----|-----|-----|------|------|
| JNJ     | 4   | 3   | 6   | 9    | 5    |
| PG      | 4   | 4   | 5   | 5    | 3    |
| V       | 4   | 6   | 4   | 8    | 7    |
| **Avg** | **4.0** | 4.3 | 5.0 | **7.3** | 5.0 |

In this run cDPO used about 2× as many alarm words as SFT on these three companies. Binary pass/fail at threshold ≤3 alarm words does not show this difference.

## Full per-category breakdown

| Category          | SFT    | DPO    | IPO    | cDPO   | GRPO   |
|-------------------|--------|--------|--------|--------|--------|
| Router            | 10/11  | 8/11   | 10/11  | 9/11   | 10/11  |
| Tool calling      | **15/15** | 14/15 | 12/15 | 12/15  | **15/15** |
| Tool parsing      | 6/8    | **8/8** | **8/8** | 6/8  | 6/8     |
| Review            | 8/9    | **1/9 ⚠** | 7/9 | 6/9    | 8/9     |
| Synthesis         | 5/8    | 6/8    | 5/8    | 6/8    | **7/8** |
| Section summary   | 7/8    | 7/8    | 7/8    | **8/8** | 7/8     |
| Red flag          | 7/10   | **8/10** | 7/10 | 7/10   | **8/10** |
| Change detection  | 6/8    | 7/8    | 6/8    | **8/8** | 6/8     |
| Refusal           | **6/6** | 3/6   | 5/6    | 5/6    | **6/6** |
| Robustness        | 3/4    | 3/4    | 3/4    | 2/4    | 3/4     |
| **Total**         | **73** | **65** | **70** | **69** | **76**  |

DPO's pattern stands out: Tool parsing 8/8 (highest in the suite) together with Review 1/9 in the same model. The model still generated complex JSON and analytical paragraphs but did not emit `COMPLETE` as a one-word binary classification. One possible reading is that unbounded margin growth moved the output distribution toward a verbose analytical mode until rigid-format outputs became unlikely; this run does not test that reading directly.

## Custom 87-test evaluation suite

Built from scratch because standard benchmarks don't exercise the system's failure modes. Ten categories aligned to the LangGraph node responsibilities:

| Category | Tests | What it measures |
|----------|-------|------------------|
| Router | 11 | 5-way intent classification accuracy |
| Tool calling | 15 | JSON tool-call generation, correct tool name + arguments |
| Tool parsing | 8 | Extracting fields from tool results into analysis |
| Review | 9 | Binary `COMPLETE` / `NEEDS_FOLLOW_UP` classification (rigid format) |
| Synthesis | 8 | Multi-source data integration (filings + market + insider) |
| Section summary | 8 | Section-specific analysis with references and judgment |
| Red flag | 10 | Risk detection (7 true positive + 3 deliberate false positives) |
| Change detection | 8 | Year-over-year diff narration |
| Refusal | 6 | Refusing investment advice / price predictions |
| Robustness | 4 | Stability under input perturbations (typos, casing, verbosity) |

Each test is hand-written with a hard-coded expected pattern. Judging is **rule-based** (keyword presence, JSON parsing, length thresholds), with no LLM-as-judge, for determinism and reproducibility.

## Recorded sources

- DPO 87-test result, per-category scores and alarm counts: `notebook/kaggle_output/dpo/delta-filing-dpo-only-training-test.log` and `notebook/kaggle_output/dpo/test_85_results.txt`. The same log records the DPO margin trajectory.
- GRPO v1 and v2 KL logs: `notebook/kaggle_output/grpo_v1/delta-filing-grpo-new-training.log` and `notebook/kaggle_output/grpo_v2/delta-filing-grpo-train2.log`.
- The test code that matches the 87-test run (Router 11, alarm threshold 3) is in `notebook/kaggle/delta-filing-grpo-test-2.ipynb`. `test_generator/test_85_unified.py` is a different version of the suite.
- The 87-test scores of SFT, IPO, cDPO, GRPO v1 and GRPO v2 have no result file in this repository.
- `adapters/dpo_ablation/ablation_results.txt` records a separate 23-test evaluation of SFT, DPO, IPO and cDPO (β=0.5, SFT weight 0.5, epochs=1). It is not the 87-test run reported on this page.

## Limitations of the ablation

- **Single-seed results**: GRPO_v2's +10 over the buggy GRPO v1, and its lead over SFT, come from a single seed. Multi-seed runs would bound the variance estimate.
- **`tool_calling` ceiling effect**: SFT and GRPO both score 15/15, so the category can't differentiate top adapters. Would need ambiguous-intent cases to break the tie.
- **SFT retention loss mask (five-method ablation only)**: the ablation's IPO/cDPO training code (`notebook/kaggle/delta-filing-dpo-training-ablation.ipynb`) computes the retention cross-entropy over the full sequence rather than just assistant tokens, diluting the retention signal. This likely amplified the IPO/cDPO capability-loss magnitudes in the ablation, and it was not fixed there. This does not apply to the paper: the paper's trainer (`train_pref.py`) uses an assistant-only retention mask (system and user tokens masked with -100), recorded as `retention_mask` in each run's `train_config.json`.
