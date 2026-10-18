# §3 Method

## 3.1 Setup

We fine-tune Qwen3.5-4B with LoRA (r = 8, α = 16, dropout = 0.05, explicitly targeting attention projections q/k/v/o\_proj; MLP projections are excluded) using vanilla length-normalized DPO (β = 0.5) with an SFT retention term. The total loss is L = L\_pref + 0.5 × L\_sft, where L\_pref is the standard DPO preference loss and L\_sft is an assistant-only cross-entropy on the chosen completion (system and user tokens masked). The retention term anchors chosen-completion likelihood during preference optimization. Training uses AdamW (weight decay 0.01) with learning rate 5 × 10⁻⁵, constant schedule (no warmup or decay), gradient clipping at 1.0, batch size 8 (gradient accumulation), 180 fixed optimizer steps with no loss-based early stop, and a divergence guard that aborts if the loss becomes NaN or the smoothed margin exceeds 20 (~3× the calibration maximum). Eight seeds (0–7) are run from the same SFT checkpoint (hash-verified 29a3794b…585f02 across all runs; tokenizer and chat template byte-identical).

The preference data consists of 647 pairs (549 train, 98 held-out; deterministic split). Two axes are represented: a correctness axis (200 tool\_calling pairs where rejected completions are wrong-tool swaps — identical arguments with a randomly substituted tool name — making chosen and rejected length-matched at ~7.5 words each) and a style axis (447 pairs across four prose categories where chosen completions are verbose analytical responses (~190–226 words) and rejected completions are GPT-4o-mini degradations with numerical values, dates, and section references stripped, specific claims replaced with vague generalizations, and text truncated to ~86–100 words). Length normalization is applied because without it, the 447 style pairs would be winnable on length alone. Seeds 0–4 were run in the first quota window and seeds 5–7 in the second, with the assignment pre-registered and results inspected only after all 8 seeds completed.

## 3.2 Tasks and Evaluation Suite

We evaluate on three structured exact-format tasks, each with 160 items. Each task presents a user query about SEC filings and requires the model to produce a specific structured output: json asks the model to select and parameterize a tool call; router asks it to classify the query into one of five categories; extraction asks it to extract structured metadata (ticker, section, filing date) from a filing passage. Free generation uses greedy decoding with no constrained decoding and no LLM judge.

**[Table: Evaluation tasks]**

| Task | Output format | Example | Preference coverage |
|---|---|---|---|
| Json | Structured JSON tool call (function name + typed arguments) | `{"name":"get_filing","args":{"ticker":"AAPL",...}}` | On-target (200 tool\_calling pairs) |
| Router | Bare UPPER\_SNAKE\_CASE category string from a fixed 5-label set (FILING\_ANALYSIS, FILING\_DIFF, COMPANY\_DEEP\_DIVE, SIMPLE\_QUERY, OUT\_OF\_SCOPE) | `COMPANY_DEEP_DIVE` | Off-target (zero) |
| Extraction | Three-field JSON: ticker, section, filing\_date | `{"ticker":"MSFT","section":"Item 7","filing_date":"2024-02-15"}` | Off-target (zero) |

All tasks produce short outputs; greedy decoding uses max\_new\_tokens = 16 (router) or 128 (json, extraction). The 16-token router limit truncates a small number of FORMAT\_BREAK outputs (11/298 at margin3), but truncation does not affect FORMAT\_BREAK classification, which depends on the leading token.

The baseline (baseline\_v2) is the step-0 SFT checkpoint evaluated on the same 160-item suite with the same scorer. Baseline uses greedy decoding and is therefore deterministic — all comparison variance comes from training seeds and item-level variation.

Scoring uses scorer v2, a frozen three-tier scheme: PASS (semantically correct), SURFACE\_FORM (correct content, surface-level format difference such as "Item 1" rendered as "1"), and SEMANTIC\_ERROR (wrong content). The primary metric is semantic accuracy (PASS + SURFACE\_FORM). FORMAT\_BREAK (malformed output that cannot be parsed) is tracked separately.

The canonicalization rule for extraction — section identifier matching via regex `\d+[A-Z]?`, case-folded, ignoring appended titles — was frozen before the matrix. This rule prevents false capability signals: under strict string matching, extraction accuracy at margin3 appears to crash to 0.550; under canonicalized scoring, the same checkpoint scores 0.969, with 93% of apparent errors being surface-form register shifts ("Item 1" vs "1"), not semantic failures.

## 3.3 Suite Independence and Domain Overlap

No eval prompt appears verbatim in the preference or SFT training data (exact-string match). However, both the eval suite and the preference data derive from EDGAR ticker×section fetches, producing domain overlap by design. We report the most specific overlap level available for each task: for extraction, at least 81/160 items share a ticker×section combination with a preference pair's source filing (lower bound; ticker×section provenance recoverable for 209/647 pairs); for json, at least 131/160 items share a ticker (section is not a json output field, so ticker-only is the meaningful comparison). At the ticker level, both tasks have high overlap; at the ticker×section level, extraction has moderate overlap and json has no equivalent concept. This overlap means extraction's preservation may partly reflect familiarity with the same entities rather than solely format-robustness; router, which has zero domain overlap with the preference data, provides the cleanest independence test.

Suite hygiene was re-derived from source files and committed: degeneracy floors measured empirically by scoring a fixed constant output against all 160 golds (extraction 0.000, json 0.006, router 0.200), class balance (router: 32 items × 5 classes; json: 20 items × 8 tool types), zero duplicate prompts, and gold self-consistency (160/160 per task).

## 3.4 Pre-Registered Analysis Plan

The primary endpoint is per-task semantic accuracy at the final checkpoint (step 180) versus baseline\_v2, tested via TOST equivalence at ±5 points. We chose ±5 as an a priori bound that is large enough to avoid trivially passing (n = 160 items with 8 seeds gives reasonable power to detect 5-point differences) yet small enough that passing is substantively meaningful (a 5-point drop on a 160-item suite corresponds to 8 items flipping, which we judged non-negligible — for instance, on json with baseline 98.75, a 5-point drop means losing 8 of ~158 currently-correct items, a detectable change in task coverage). The bound was frozen before any matrix data existed. We acknowledge this is researcher judgment, not derived from an external standard.

Confidence intervals are constructed at two levels: seed-level (mean per-seed accuracy difference, t-based CI with df = 7) and item-level (bootstrap over items with seeds pooled). Both must lie within ±5 for the task to pass. Additionally, ≥7 of 8 seeds must individually fall within ±5. If all three tasks pass, the bounded-null headline holds; if any task fails, the task-selective branch governs. No multiple-comparison correction is applied: the headline requires all tasks to pass (an intersection-union test), which is inherently conservative.

Step20 is saved at a fixed optimizer step; final is saved at step 180 (fixed across all seeds). Margin1, margin3, and margin5 are saved when the 20-sample smoothed training margin first crosses the threshold (1.0, 3.0, 5.0 respectively); the step at which this occurs varies by seed (margin1: steps 24–30, margin3: steps 84–98, margin5: steps 112–138). The margin value reported alongside each checkpoint is the smoothed margin at the save step. The pre-registered design specified fixed-step checkpoints with margin as a post-hoc label; the implementation used margin-crossing triggers for intermediate checkpoints. Cross-seed averages at these checkpoints pool seeds at matched margin but different optimizer steps; reported variance reflects both sources. The primary endpoint (step 180) and diagnostic (step20) are unaffected.

Step20 is excluded from all primary and secondary endpoints. All outcome-to-interpretation mappings were frozen and date-stamped before the decisive runs (Notes 1 and 2, committed 2026-06-11 and 2026-06-18).

## 3.5 Probability-Layer Measurement

To assess whether DPO training selectively protects gold completions relative to incorrect alternatives, we teacher-force three types of completions through each checkpoint: gold (the correct answer), control A (well-formed but with a different section identifier — a semantically wrong but structurally valid answer), and control B (well-formed but with a different ticker — also semantically wrong but structurally valid). The primary statistic is the paired difference Δgold − Δmean(A,B), where Δ denotes the change in per-token log-probability from the SFT baseline. A positive stat indicates gold deflates less than controls — selective protection. The pre-registered threshold is X = 0.05 nats/token, with 95% bootstrap CI required to exclude zero.

Controls A and B are designed for extraction only (the section-identifier and ticker dimensions). On json and router, gold-logp is reported as absolute Δgold without a paired control baseline. Two backends are used: fp16 (Mac, same-backend for SFT baseline and all final checkpoints, 8 seeds) and nf4 (matrix gold-logp records; nf4 uses step20 as the earliest available reference because the SFT checkpoint was not scored for gold-logp under nf4). The task-selective ordering is assessed under both.

A complete reconciliation of all pre-registered secondaries (trajectory-min, FORMAT\_BREAK trajectory, held-out margin per category, chosen/rejected reward decomposition, gold-logp ×3 tasks, probe-stat replication, register-shift tracking, KL-from-reference) against their status (reported, filled post-hoc, or implementation gap) is provided in Appendix C.
