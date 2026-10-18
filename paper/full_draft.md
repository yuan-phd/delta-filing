# The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention

# Abstract

**Setting.** We fine-tune Qwen3.5-4B with sparse LoRA and length-normalized DPO plus an SFT retention term (the RPO objective) on structured financial-filing tasks, and track downstream capability as the implicit reward margin grows to ~7 β-scaled units (nf4 training backend; ~3 under fp16 dequantized-base measurement). **Question.** Does margin growth degrade capability uniformly, as displacement theory and IPO's design motivation predict? **Findings.** In a pre-registered 8-seed study with artifact-aware scoring and TOST equivalence testing, we find the alignment tax is graded and task-selective: extraction is preserved (Δ = −0.55, equivalent at ±5), json loses little (Δ = −3.52, equivalence fails on a clustered item-level interval), and router degrades severely (Δ = −30). Router's loss concentrates in a format-break window at margin 1–3. The margin is accounted for entirely by rejected-completion suppression with chosen held flat by the retention term; the source of the behavioral damage is not attributed. Gold-logp deflation is task-selective (extraction −0.15, json −1.04, router −1.47 nats/token under fp16), with gold selectively protected relative to incorrect controls on extraction (8/8 seeds). Implicit reward margins under QLoRA differ up to 3.4× between the nf4 training backend and fp16 dequantized measurement, axis-dependently. **Scope.** Length-normalized rewards with SFT retention, sparse LoRA on 8/32 layers, one 4B model, easily-separable synthetic preferences, three structured tasks. A no-retention control and a measured format-alignment metric are left to future work.

# §1 Introduction

DPO's implicit reward margin grows large under length-normalized rewards, reaching ~7 β-scaled units in our experiments. This echoes the unbounded growth IPO was designed to prevent in the standard formulation (our margins are per-token; see §3.1 for the conversion). The dominant framing in the literature treats this growth as pathological. Razin et al. (2025) identify likelihood displacement, where chosen log-prob decreases and probability mass may flow to out-of-distribution completions. IPO (Azar et al., 2024) was explicitly designed to bound the margin, treating unbounded growth as the core problem. Cal-DPO (Xiao et al., 2024) observes absolute likelihood deflation and proposes a calibration fix. Yet the empirical basis for these concerns comes primarily from judge-based evaluation or full fine-tuning at large scale. The common practical regime is understudied: small model, LoRA, length-normalized rewards, and structured exact-format tasks. A methodological gap compounds the empirical one: the literature lacks equivalence testing for capability preservation. Studies sometimes report "no significant difference" without the statistical power to distinguish "preserved" from "underpowered to detect a real drop."

This is an empirical characterization study: we do not propose a new training algorithm, but measure how an existing one (length-normalized DPO with an SFT regularization term) affects downstream capability as its implicit reward margin grows. This paper presents a pre-registered 8-seed study of a 4B model fine-tuned with LoRA. We measure behavioral capability (free-generation with artifact-aware scoring) and probability-layer dynamics (per-token gold/control log-probabilities) as the implicit reward margin grows to ~7 β-scaled units. That value is from the nf4 training backend; the margin is ~3 under fp16 dequantized-base measurement. Our training objective combines length-normalized DPO with an SFT retention term (weight 0.5) that anchors chosen-completion likelihood. This is a common practical choice. As we show, the full margin is then accounted for by rejected-completion suppression.

The alignment tax is real but graded. It ranges from undetectable on one task (extraction, Δ = −0.55, TOST equivalence PASS) through small but statistically genuine on a second (json, Δ = −3.52) to catastrophic on a third (router, Δ = −30). The gradient tracks format-alignment to the preference data's dominant style axis, consistently across all 8 seeds. With three tasks, we report this co-occurrence without claiming causation.

The mechanism is asymmetric. The retention term keeps chosen reward approximately stationary while DPO gradients drive rejected reward to −14.2 (nf4). As a result, margin growth comes entirely from rejected-completion collapse. The style axis dominates this suppression in aggregate under both backends, but per-pair ordering is backend-dependent (§4.3). The pre-registered prediction that style would dominate held-out per-pair margin at every checkpoint was falsified under both backends (fp16 0/4; nf4 2/4). Gold-logp deflation is itself task-selective and directionally consistent with behavioral outcomes, though non-linearly related (§4.3–4.4). A naive strict-match scorer masked this picture by producing a false capability collapse (see the scorer artifact in §3.2). Artifact-aware evaluation and pre-registered TOST equivalence testing prevented that error. Here TOST serves as the primary evaluation framework for capability assessment.

Our contributions are:

1. A graded, task-selective alignment tax, with margin growth accounted for entirely by rejected-completion suppression; the source of the behavioral damage is not attributed pending a no-retention control; the format-alignment hypothesis for which tasks are affected is consistent with the data but not directly measured (§4.1–4.3). It includes non-monotonic format-break dynamics at margin ≈ 3 and a falsified pre-registered prediction on held-out per-pair margin.
2. Task-selective gold-logp deflation, with selective protection demonstrated on extraction (8/8 seeds, paired statistic) and robust across quantization backends (§4.4).
3. An artifact-aware evaluation protocol with pre-registered TOST equivalence testing (§3). Its falsification chronology (three falsified hypotheses, two refined characterizations) documents how pre-registration forced successive refinement of initially simpler claims (§5.2).

This study operates within explicit scope conditions: sparse LoRA on 8/32 layers (0.034% of parameters; not full fine-tuning), a single 4B model, length-normalized rewards with an SFT retention term (weight 0.5), easily-separable synthetic preferences, and three structured exact-format tasks. We do not study safety or refusal tasks, where Razin et al. demonstrated catastrophic displacement. We also do not study full fine-tuning, naturalistic human-written preferences, or models above 4B parameters. These are stated boundaries, not implicit weaknesses.

# §2 Related Work

**Likelihood displacement.** Razin et al. (ICLR 2025) identify likelihood displacement under DPO: chosen log-prob decreases while probability mass may flow to out-of-distribution or harmful completions. They introduce the CHES score (centered hidden embedding similarity) to identify preference pairs whose embedding proximity drives catastrophic displacement. Their own analysis notes displacement is "not necessarily problematic"; it is catastrophic only when mass flows to opposite-meaning responses. Our study includes an SFT retention term that anchors chosen likelihood. This term is equivalent to RPO (Liu et al., 2024) and is the mitigation Razin et al. themselves propose. The chosen-decline dynamic their framework describes is therefore largely suppressed (§4.3). Whether displacement would emerge without the retention term in our setup is untested. Pipano et al. (2026) characterize which *f*-divergence losses are displacement-resistant by construction. Our approach instead retains the standard DPO loss and relies on the SFT term for anchoring.

**The squeezing effect.** Ren and Sutherland (ICLR 2025) show that DPO's gradient is dominated by negative gradients on rejected completions. These gradients concentrate probability mass onto the highest-probability token at each position, which they call the squeezing effect. Their analysis demonstrates the effect at the single-token level on Pythia and Qwen. We observe dynamics consistent with the same mechanism on multi-token structured completions, though the transfer across settings is assumed rather than proven. Our reward decomposition (§4.3) is textbook squeezing: rejected reward falls to −14.2 while chosen stays within −0.072 to +0.052 (nf4). The retention term contributes to this chosen stability. The squeezing framework also provides a candidate explanation for task-selective gold deflation (§4.4). Extraction outputs involve entities also present in training data, so extraction items may benefit from mass concentration onto gold tokens. Router outputs use a terse format maximally dissimilar from the verbose training pairs, so router items may not.

**Ordering preservation and calibration.** Cal-DPO (Xiao et al., NeurIPS 2024) observes that DPO preserves preference ordering while absolute likelihoods deflate. It proposes calibrating the implicit reward to match ground-truth reward scale, preventing chosen-reward decline. Our regime includes an SFT retention term that achieves a similar anchoring effect on chosen completions (§3.1). The finding that chosen is flat does not demonstrate Cal-DPO's concern is unfounded. It demonstrates the concern is already addressed in our training objective. The more informative observation is that the behavioral alignment tax remains on off-target tasks (§4.1) even with chosen explicitly anchored. The margin is accounted for by rejected suppression (§4.3), but the source of the behavioral damage is not attributed.

**Bounding the margin.** IPO (Azar et al., AISTATS 2024) replaces DPO's log-sigmoid with a squared loss to bound the implicit reward margin, treating unbounded growth as the core pathology. Our data shows margin growing to ~7 (β-scaled) with graded, format-selective behavioral effects (§4.1, §4.2). These effects are not the uniform catastrophe the bounding was designed to prevent. This does not invalidate IPO generally; it characterizes a regime where the damage is concentrated in a margin 1–3 window and is task-selective.

**Chosen-reward predictions.** Rafailov et al. (COLM 2024) predict from a Q-function framework that chosen log-prob should decrease under DPO. Our training objective includes an SFT retention term that directly counteracts this. We discuss the implications for interpreting our chosen-flatness result in §5.3.

**Equivalence testing and scoring methodology.** Manzoni (arXiv 2603.24044, 2026 preprint) uses TOST equivalence testing with 8 seeds for SFT capability preservation in MoE fine-tuning. A concurrent paper on task-oriented dialogue (Bao and Abisado, "Pareto-Aware Dual-Preference Optimization," Symmetry 2026) applies TOST as incidental validation in a DPO context. To our knowledge, pre-registered TOST as the primary evaluation framework for DPO capability assessment is unprecedented in the DPO literature. TOST itself is well-established in bioequivalence and related fields. On scoring, Baxi (arXiv 2512.17920, 2025 preprint) separates constraint compliance from semantic accuracy in LLM evaluation, establishing the conceptual decomposition. Our contribution is the trajectory dynamics: a transient format-break spike versus a monotonic content-error rise (§4.2). Minerva (Lewkowycz et al., 2022) provides precedent for answer normalization preventing false negatives in exact-match evaluation. Our canonicalization result is a concrete alignment-specific instance (see the scorer artifact in §3.2).

**DPO dynamics and alignment-tax context.** Li (arXiv 2603.19335, 2026 preprint) presents OXRL, which evaluates 20 DPO variants under Bonferroni correction and finds none outperforms vanilla DPO. This is consistent with their finding that loss-function modifications produce minimal behavioral differences at matched compute. On the alignment tax more broadly, Young (arXiv 2603.00047, 2026 preprint) frames the alignment tax geometrically as tracking the projection of the safety direction onto the capability subspace. Our format-axis-alignment finding (§4.1) is an empirical instantiation. Pollanen (arXiv 2601.17260, 2026 preprint) reports that margin can anticorrelate with reasoning capability (r = −0.91 for LLaMA on logic tasks when sweeping β across a dense grid). Their study varies β rather than observing within-run margin growth. The result is therefore related to, but distinct from, our trajectory analysis.

# §3 Method

## 3.1 Setup

We fine-tune Qwen3.5-4B with LoRA (r = 8, α = 16, dropout = 0.05) using vanilla length-normalized DPO (β = 0.5) with an SFT retention term. Let $\pi_\theta$ denote the policy and $\pi_{\text{ref}}$ the SFT reference. For a prompt $x$ and completion $y$ of length $|y|$ tokens, the length-normalized log-probability is

<!-- eq:lnlogp -->
$$\ell_\theta(y \mid x) = \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}). \tag{1}$$

Length normalization follows the per-token formulation discussed by Meng et al. (2024). The implicit reward of a completion is $r_\theta(x, y) = \ell_\theta(y \mid x) - \ell_{\text{ref}}(y \mid x)$, the policy log-probability minus the reference log-probability. Given a preference pair $(x, y_w, y_l)$ with chosen $y_w$ and rejected $y_l$, the β-scaled preference margin and the DPO preference loss are

<!-- eq:pref -->
$$m = \beta\,\big[r_\theta(x, y_w) - r_\theta(x, y_l)\big], \qquad \mathcal{L}_{\text{pref}} = -\log \sigma(m), \qquad \beta = 0.5. \tag{2}$$

Because Eq. (1) is per-token and β = 0.5, a reported margin of 7 corresponds to a per-token log-ratio difference of 14 nats. Under the policy, each rejected token is on average $e^{14} \approx 1.2 \times 10^{6}$ times less likely, relative to the reference, than each chosen token. This is near-total suppression of the rejected text. Displacement and IPO analyses typically reason about sequence-level log-ratios, so our margins are not directly comparable in scale.

The total training objective adds an assistant-only cross-entropy on the chosen completion, with system and user tokens masked:

<!-- eq:total -->
$$\mathcal{L} = \mathcal{L}_{\text{pref}} + \lambda\, \mathcal{L}_{\text{sft}}, \qquad \mathcal{L}_{\text{sft}} = -\ell_\theta(y_w \mid x), \qquad \lambda = 0.5. \tag{3}$$

Equation (3) is the Regularized Preference Optimization (RPO) objective of Liu et al. (2024); a related DPO+NLL objective appears in Pang et al. (2024). We refer to the $\lambda \mathcal{L}_{\text{sft}}$ term as the SFT retention term throughout. Razin et al. (2025) identify this SFT regularization as a mitigation for likelihood displacement. The retention term anchors chosen-completion likelihood during preference optimization.

Qwen3.5-4B is a hybrid architecture with 24 linear-attention layers and 8 full-attention layers (every 4th: layers 3, 7, 11, 15, 19, 23, 27, 31). LoRA adapts only the 8 full-attention layers (q/k/v/o\_proj; 1.57M trainable parameters, 0.034% of 4.66B total). The 24 linear-attention layers and all MLPs are untouched. Training uses AdamW (weight decay 0.01) with learning rate 5 × 10⁻⁵ and a constant schedule (no warmup or decay). We use gradient clipping at 1.0, batch size 8 (gradient accumulation), and 180 fixed optimizer steps with no loss-based early stop. A divergence guard aborts training if the loss becomes NaN or the smoothed margin exceeds 20 (~3× the calibration maximum). Eight seeds (0–7) are run from the same SFT checkpoint (hash-verified 29a3794b…585f02 across all runs). The chat template is byte-identical across sources, and tokenization is identical on all study inputs: the SFT adapter, DPO adapters, and base tokenizer configs differ in metadata (added-token count, pre-tokenizer regex) but produce identical token IDs on all 6,024 evaluation and preference strings, verified by direct token-ID comparison.

The preference data consists of 647 pairs (549 train, 98 held-out; deterministic split) on two axes. The correctness axis has 200 tool\_calling pairs, 171 of them in the 549-pair training split. Its rejected completions are wrong-tool swaps: identical arguments with a randomly substituted tool name. Chosen and rejected are therefore length-matched at ~7.5 words each. The style axis has 447 pairs, 378 of them in the training split, across four prose categories. Its chosen completions are verbose analytical responses (~190–226 words). Its rejected completions are GPT-4o-mini degradations: numerical values, dates, and section references are stripped, specific claims are replaced with vague generalizations, and text is truncated to ~86–100 words. Length normalization is applied because without it, the 447 style pairs would be winnable on length alone. Seeds 0–4 were run in the first quota window and seeds 5–7 in the second.

## 3.2 Tasks and Evaluation Suite

We evaluate on three structured exact-format tasks, each with 160 items. Each task presents a user query about SEC filings and requires a specific structured output. Json asks the model to select and parameterize a tool call (examples in the evaluation-tasks table below). Router asks it to classify the query into one of five categories. Extraction asks it to extract structured metadata (ticker, section, filing date) from a filing passage. Free generation uses greedy decoding with no constrained decoding and no LLM judge. All behavioral generations are produced under nf4 quantization, matching the training backend. The TOST verdicts therefore describe the deployed-quantization regime end-to-end.

**[Table: Evaluation tasks: output format, example, and preference-data coverage.]**

| Task | Output format | Example | Preference coverage |
|---|---|---|---|
| Json | Structured JSON tool call (function name + typed arguments) | `{"tool":"search_filings","arguments":{"ticker":"AAPL"}}` | On-target (200 tool\_calling pairs) |
| Router | Bare UPPER\_SNAKE\_CASE category string from a fixed 5-label set (FILING\_ANALYSIS, FILING\_DIFF, COMPANY\_DEEP\_DIVE, SIMPLE\_QUERY, OUT\_OF\_SCOPE) | `COMPANY_DEEP_DIVE` | Off-target (zero) |
| Extraction | Three-field JSON: ticker, section, filing\_date | `{"ticker":"MSFT","section":"Item 7","filing_date":"2024-02-15"}` | Off-target (zero) |

All tasks produce short outputs; greedy decoding uses max\_new\_tokens = 16 (router) or 128 (json, extraction). The 16-token router limit truncates a small number of FORMAT\_BREAK outputs (6/251 at margin3). Truncation does not affect FORMAT\_BREAK classification, which depends on the leading token.

The baseline is the step-0 SFT checkpoint, evaluated on the same 160-item suite with the same scorer. Baseline uses greedy decoding and is therefore deterministic. All comparison variance comes from training seeds and item-level variation.

Scoring uses scorer v2, a frozen four-tier classification: FORMAT\_BREAK (unparseable), PASS (semantically correct), SURFACE\_FORM (correct content, surface-level format difference such as "Item 1" rendered as "1"), and SEMANTIC\_ERROR (wrong content). The primary metric is semantic accuracy (PASS + SURFACE\_FORM). FORMAT\_BREAK rate is also tracked separately.

The canonicalization rule for extraction was frozen before the 8-seed confirmatory run. It matches section identifiers via regex `\d+[A-Z]?`, case-folded, ignoring appended titles. This rule prevents false capability signals. Under strict string matching, extraction accuracy at margin3 appears to crash, but canonicalized scoring recovers it: seed0: 0.550 → 0.981 (69/72 = 96%); 8-seed mean: 0.616 → 0.988. The flipped items are surface-form register shifts ("Item 1" vs "1"), not semantic failures.

## 3.3 Suite Independence and Domain Overlap

No eval prompt appears verbatim in the preference or SFT training data (exact-string match). However, both the eval suite and the preference data derive from EDGAR ticker×section fetches, producing domain overlap by design. We report the most specific overlap level available for each task. For extraction, at least 81/160 items share a ticker×section combination with a preference pair's source filing. This is a lower bound, since ticker×section provenance is recoverable for 209/647 pairs. For json, at least 131/160 items share a ticker. Section is not a json output field, so ticker-only is the meaningful comparison. At the ticker level, both tasks have high overlap; at the ticker×section level, extraction has moderate overlap and json has no equivalent concept. This overlap means extraction's preservation may partly reflect familiarity with the same entities rather than solely format-robustness. Router has zero domain overlap with the preference data and provides the cleanest independence test.

Suite hygiene was re-derived from source files and committed. Degeneracy floors are the maximum accuracy achievable by any fixed constant output: extraction 0.006, json 0.0125 (2/160), router 0.200. Extraction golds are unique, so the constant-output floor is 1/160 ≈ 0.006. Router has 5 balanced classes (32/160 = 0.20). Json has 18 duplicate tool-call tuples (2/160 = 0.0125). Classes are balanced (router: 32 items × 5 classes; json: 20 items × 8 tool types), there are zero duplicate prompts, and gold self-consistency is 160/160 per task.

## 3.4 Pre-Registered Analysis Plan

The primary endpoint is per-task semantic accuracy at the final checkpoint (step 180) versus the step-0 SFT baseline, tested via TOST equivalence at ±5 points. Let $\Delta = \text{acc}_{\text{final}} - \text{acc}_{\text{baseline}}$ in percentage points. TOST tests the composite null $H_0: |\Delta| \geq \delta$ against $H_1: |\Delta| < \delta$ with $\delta = 5$. The confidence-interval criterion requires the 90% confidence interval for $\Delta$ to lie entirely within $(-\delta, \delta)$:

<!-- eq:tost -->
$$\text{CI}_{90\%}(\Delta) \subset (-\delta, \delta) = (-5, 5). \tag{4}$$

We chose ±5 as an a priori bound that is large enough to avoid trivially passing: a post hoc calculation using the observed per-seed standard deviations (2.28 points on extraction, 1.56 on json) gives power above 0.99 for the seed-level TOST criterion at ±5 when the true Δ is 0. The bound is also small enough that passing is substantively meaningful: a 5-point drop on a 160-item suite corresponds to 8 items flipping, which we judged non-negligible. For instance, on json with baseline 98.75, a 5-point drop means losing 8 of ~158 currently-correct items, a detectable change in task coverage. The bound was frozen before any data from the 8-seed confirmatory run existed. We acknowledge this is researcher judgment, not derived from an external standard.

Confidence intervals are constructed at two levels: seed-level (mean per-seed accuracy difference, t-based CI with df = 7) and item-level (bootstrap over items with seeds pooled). Both must satisfy Eq. (4) for the task to pass. Additionally, ≥7 of 8 seeds must individually fall within ±5. If all three tasks pass, the bounded-null headline holds; if any task fails, the task-selective branch governs. No multiple-comparison correction is applied. The headline requires all tasks to pass (an intersection-union test), which is inherently conservative.

Step20 is saved at a fixed optimizer step; final is saved at step 180 (fixed across all seeds). Margin1, margin3, and margin5 are saved when the 20-sample smoothed training margin first crosses 1.0, 3.0, and 5.0 respectively. The step at which this occurs varies by seed (margin1: steps 24–30, margin3: steps 84–98, margin5: steps 112–138). The margin value reported alongside each checkpoint is the smoothed margin at the save step. The pre-registered design specified fixed-step checkpoints with margin as a post-hoc label. The implementation used margin-crossing triggers for intermediate checkpoints. Cross-seed averages at these checkpoints pool seeds at matched margin but different optimizer steps, so reported variance reflects both sources. The primary endpoint (step 180) and diagnostic (step20) are unaffected.

Step20 is excluded from all primary and secondary endpoints. All outcome-to-interpretation mappings were frozen and date-stamped before the decisive runs (Notes 1 and 2, committed 2026-06-11 and 2026-06-18). The pre-registration was informed by pilot runs on seeds 0 and 1 and frozen (commit 62ab7c3) before seeds 2–7 were run. Training is bitwise deterministic, so the confirmatory seeds 0 and 1 reproduce the pilot adapters: seed 0 at every checkpoint, seed 1 through margin5 (the pilot early-stopped at step 156; the confirmatory run continued to step 180). The confirmatory set therefore contains six seeds (2–7) whose outcomes were not visible at freeze time. All 8 seeds are reported; results restricted to seeds 2–7 are in Appendix A. The notes were committed in e3be4d5 (Note 1) and 62ab7c3 (Note 2, frozen), and will be released with the code. The repository was first pushed publicly on 2026-10-18; the commit dates reflect local authorship and are not independently timestamped.

## 3.5 Probability-Layer Measurement

This study uses two quantization backends. nf4 is the training and behavioral-evaluation backend. All training, checkpoint saving, behavioral generation (§4.1–4.2), and the reward decomposition (§4.3) were performed under nf4. fp16 is the dequantized-base measurement: the same adapters loaded onto an unquantized base. Because the adapters were trained against the nf4 base, neither backend is privileged; nf4 reports what the optimizer experienced, fp16 reports what the adapter weights encode when the base is not quantized. Gold-logp measurement (§4.4), the GOLD\_PROTECTED paired statistic, KL trajectory, and the held-out margin recomputation (Table 12; 69 style, 29 tool_calling) were performed under fp16 on Mac. Cross-backend magnitude comparisons are not valid, so within-backend comparisons are used throughout. Cross-backend ratios appear only in the §4.5 sensitivity analysis.

To assess whether DPO training selectively protects gold completions relative to incorrect alternatives, we teacher-force three types of completions through each checkpoint. Gold is the correct answer. Control A is well-formed but has a different section identifier. Control B is well-formed but has a different ticker. Both controls are semantically wrong but structurally valid answers. For each item $i$, let $\Delta_{\text{gold}}^{(i)}$ denote the change in $\ell_\theta$ (Eq. 1) for the gold completion from SFT baseline to final checkpoint, and $\Delta_A^{(i)}, \Delta_B^{(i)}$ the same for controls A and B. The paired statistic is

<!-- eq:goldstat -->
$$S = \frac{1}{N} \sum_{i=1}^{N} \left[ \Delta_{\text{gold}}^{(i)} - \tfrac{1}{2}\big(\Delta_A^{(i)} + \Delta_B^{(i)}\big) \right], \tag{5}$$

with $N = 160$. A positive $S$ indicates gold deflates less than the controls on average, i.e., selective protection. The pre-registered threshold is $S \geq X = 0.05$ nats/token, with a 95% bootstrap confidence interval excluding zero.

Controls A and B are designed for extraction only (the section-identifier and ticker dimensions). On json and router, gold-logp is reported as absolute Δgold without a paired control baseline. Two backends are used. fp16 (Mac) is same-backend for the SFT baseline and all final checkpoints, over 8 seeds. nf4 uses the gold-logp records from the 8-seed confirmatory run, with step20 as the earliest available reference because the SFT checkpoint was not scored for gold-logp under nf4. The task-selective ordering is assessed under both.

A complete reconciliation of all pre-registered secondaries against their status (reported, filled post-hoc, or implementation gap) is provided in Appendix C. The secondaries are trajectory-min, FORMAT\_BREAK trajectory, held-out margin per category, chosen/rejected reward decomposition, gold-logp ×3 tasks, probe-stat replication, register-shift tracking, and KL-from-reference.

**Reported quantities.** Semantic accuracy for a task is the mean over its 160 items of an indicator that the output is PASS or SURFACE\_FORM under scorer v2. FORMAT\_BREAK rate and content-error (SEMANTIC\_ERROR) rate are the corresponding means of their indicators. The smoothed margin used for checkpoint triggers is the mean of the per-pair margin $m$ (Eq. 2) over the 20 most recent training pairs (about 2.5 optimizer steps at batch size 8). Aggregate axis margin is $\sum_{i \in \text{axis}} m_i$ over the training pairs in that axis; mean per-pair margin is the same sum divided by the pair count. The seed-level confidence interval is a $t$ interval over the 8 per-seed accuracy differences ($df = 7$). The item-level interval is a bootstrap over items with seeds pooled (10,000 resamples), which treats the 8 per-item observations as a cluster. Token-level KL is $\sum_v \pi_\theta(v) [\log \pi_\theta(v) - \log \pi_{\text{ref}}(v)]$ averaged over gold-token positions.

**Reproducibility.** Code, the 647 preference pairs, the three 160-item evaluation suites, scorer v2, and all 40 checkpoint adapters will be released at https://github.com/yuan-phd/delta-filing. The base model is Qwen3.5-4B (Hugging Face `Qwen/Qwen3.5-4B`). nf4 quantization uses bitsandbytes (4-bit NF4, double quantization, fp16 compute; installed unpinned via pip on Kaggle in June 2026, exact version not recorded). fp16 measurements use PyTorch MPS on Apple Silicon with torch.float16 (no bf16 fallback).

# §4 Results

All 8 seeds completed 180 fixed optimizer steps without triggering the divergence guard. Under the nf4 training backend (β = 0.5), final margins range from 6.08 to 7.68 (mean 6.73). Checkpoint margin labels are training-time smoothed values. The pooled value derived from Table 11 (all 549 train pairs) differs slightly (e.g. 6.73 smoothed vs 7.12 decomposed at final). Within each backend, held-out preference margin tracks training margin closely (nf4: 7.12 vs 6.98, within 2%; fp16: 2.95 vs 2.78, within 6%; Appendix F). This indicates that margin growth is not driven by train-set memorization. Absolute margins differ 2.4× (train) to 2.5× (held-out) between backends, and up to 3.4× on the style axis. §4.5 analyzes this quantization sensitivity.

## 4.1 Graded, Task-Selective Degradation

Table 1 reports the verdicts of the primary TOST equivalence test (Eq. 4). The pre-registered falsifier fires: router degrades far beyond the ±5-point bound, and the bounded-null headline is dropped. The task-selective branch governs.

**[Table 1: Primary TOST equivalence test (±5 pts, semantic accuracy at step 180 vs baseline)]**

| Task | Baseline | Mean final | Δ | Seed-level 90% CI | Item-level 90% CI | Seeds with |Δ| ≤ 5 | Verdict |
|---|---|---|---|---|---|---|---|
| Extraction | 97.50 | 96.95 | −0.55 | [−2.07, +0.98] | [−1.80, +0.86] | 8/8 | Equivalent |
| Json | 98.75 | 95.23 | −3.52 | [−4.56, −2.47] | [−5.31, −1.95] | 7/8 | Not equivalent |
| Router | 93.75 | 63.59 | −30.16 | [−35.65, −24.66] | [−34.92, −25.47] | 0/8 | Not equivalent |

Extraction passes all three criteria comfortably. The point estimate is −0.55, both confidence intervals lie well within ±5, and all 8 seeds fall within the bound. A per-item flip analysis finds no systematic asymmetry (15 gains vs 22 losses, sign test, p = 0.32). Extraction capability is preserved as the margin grows to ~7.

Json fails the equivalence test, though the failure is marginal. The point estimate (−3.52) and the seed-level CI are within ±5. The test fails because the item-level pooled CI lower bound (−5.31) crosses the −5 threshold. The seed count is not the cause, even though only 7 of 8 seeds fall within the bound. Seed0 at −6.25 is clearly outside; seed4 at −5.00 sits exactly at the inclusive boundary and passes (we apply |Δ| ≤ 5 inclusively; Note 2 did not specify strict versus inclusive). The per-item flip analysis confirms the drop is not noise. Across all 8 seeds and 160 items, there are 0 gains and 45 losses (McNemar's test with continuity correction, p = 5.4 × 10⁻¹¹). With only 2 of 160 items incorrect at baseline, zero gains is structurally near-guaranteed, so the test's significance mainly reflects the number of losses rather than directional asymmetry. The degradation is small in absolute magnitude. The ceiling (baseline 98.75) caps gains, which reduces item variance; the item-level CI is wide because losses cluster on specific items. The 45 losses fall on 17 distinct items; five items account for 28 of them (one item lost on 7 of 8 seeds, two on 6 of 8). Because losses cluster within items, they are not independent across seeds, which further limits the McNemar test. Of the 45 losses, 35 are wrong-ticker errors in the free-text argument field (the tool name is correct), 7 are wrong-tool substitutions (6 on seed 0), and 3 are format breaks.

Router degrades catastrophically: −30.16 points, with both confidence intervals far outside ±5 and 0 of 8 seeds within the bound. The endpoint distribution is approximately symmetric (mean 0.636, median 0.638, range 0.488 to 0.769). Its spread (0.281) is wide, however, reflecting high seed-to-seed variance in outcomes despite consistent damage depth (§4.2).

The alignment tax is real, graded, and task-selective. It ranges from undetectable (extraction, Δ = −0.55) through small but genuine (json, Δ = −3.52) to catastrophic (router, Δ = −30.16). The gradient of degradation is directionally consistent with each task's format-alignment to the preference data across all 8 seeds. Router's terse enum format is the most orthogonal to the preference data's dominant verbose style axis, and router degrades most. Json has matching tool_calling coverage and degrades modestly. Extraction has an orthogonal but non-antagonistic format and is preserved. However, with only three tasks, we report this co-occurrence without claiming the causal link is established. Extraction also lacks direct preference coverage yet is preserved, so coverage absence alone does not predict degradation.

Json has 200 matching tool_calling preference pairs, yet still shows degradation. The per-axis training dynamics (§4.3) explain why: tool_calling coverage provides partial but not complete protection against the dominant style-axis aggregate pressure.

## 4.2 Non-Monotonic Dynamics with Two Distinct Processes

Router's trajectory across checkpoints reveals that the degradation is not monotonic in margin. Table 2 decomposes router accuracy at each checkpoint into its two failure modes: format breaks (malformed output) and content errors (well-formed but incorrect).

**[Table 2: Router trajectory decomposition (mean ± std over 8 seeds; nf4, training backend)]**

| Checkpoint | Margin | Semantic acc | FORMAT_BREAK rate | Content-error rate |
|---|---|---|---|---|
| step20 | 0.75 | 0.852 ± 0.015 | 0.001 ± 0.002 | 0.148 ± 0.016 |
| margin1 | 1.09 | 0.816 ± 0.030 | 0.007 ± 0.010 | 0.177 ± 0.038 |
| margin3 | 3.12 | 0.587 ± 0.061 | 0.196 ± 0.089 | 0.217 ± 0.071 |
| margin5 | 5.13 | 0.626 ± 0.049 | 0.133 ± 0.096 | 0.241 ± 0.083 |
| final | 6.73 | 0.636 ± 0.077 | 0.106 ± 0.070 | 0.258 ± 0.074 |

Two processes with different dynamics are visible. FORMAT_BREAK rate is non-monotonic: it peaks at margin ≈ 3 (0.196), then partly subsides by the final checkpoint (0.106). Content-error rate rises monotonically from 0.148 to 0.258, never recovering. The apparent partial recovery in overall accuracy between margin3 and the final checkpoint is driven by format-break subsiding, not by content correctness improving.

Format breaks at margin ≈ 3 consistently take the form of invented UPPER\_SNAKE\_CASE pseudo-labels derived from query content (e.g., TOTAL\_REVENUES\_2023\_Q2, REPORT\_PER\_ANALYSIS). Notably, the valid router labels are also UPPER\_SNAKE\_CASE (FILING\_ANALYSIS, COMPANY\_DEEP\_DIVE, etc.). The model therefore generates in the correct register but invents labels outside the valid 5-category set. This is a label-vocabulary interference specific to the preference data's financial domain, rather than degeneration, truncation, or style bleed. This pattern replicates across the seeds of the 8-seed confirmatory run.

The margin ≈ 3 instability is not limited to router. On extraction, the surface-form register-shift rate (SURFACE_FORM tier in scorer v2) mirrors the non-monotonic shape. It spikes at margin ≈ 3 (mean 0.37, range 0.03–0.77 across seeds), then partly subsides to 0.09 by final. This confirms that the margin 1–3 damage window produces surface-form instability across tasks. Under the strict scorer, this register shift would have been counted as a capability collapse. Under scorer v2's canonicalization, the apparent errors are surface-form shifts, not semantic failures (see the scorer artifact in §3.2). This is the paper's most concrete demonstration of why artifact-aware scoring is necessary.

Router accuracy reaches its trajectory minimum at the margin3 checkpoint on 6 of 8 seeds (mean 0.587, range 0.444 to 0.650). Two seeds drift slightly lower afterward. Degradation begins before the margin 1–3 interval. Step20 already shows −8.6 pts from baseline (28% of the net −30.2 pt loss), and margin1 accounts for −12.2 pts (40%). Training uses a constant learning rate with no warmup, so some of this early loss may reflect optimization transient rather than margin-driven damage; we did not run a warmup ablation. The resolution of our checkpoint schedule (margin1, margin3, margin5, final) cannot pinpoint the trough more precisely. The damage accumulates between margin ~1 and ~3 and stabilizes thereafter. Despite the consistent trough depth, the degree of subsequent recovery is highly seed-dependent. Endpoint accuracy ranges from 0.488 (seed5, which shows near-zero recovery) to 0.769 (seed1). The per-seed format-break rate at the final checkpoint ranges from 0.019 to 0.256, indicating that the mix of failure modes is itself stochastic across seeds. The damage is reproducible; the recovery is stochastic.

Figure 1 shows per-seed margin-vs-capability trajectories for all three tasks. Extraction forms a tight band near the baseline. Json shows mid-trajectory volatility but converges to a narrow endpoint range. Router forms a wide fan. The variance structure itself communicates the finding.

## 4.3 Mechanism: Rejected Collapse and Axis-Dependent Backend Sensitivity

We decompose the margin $m$ (Eq. 2) into its chosen and rejected reward components. Chosen reward is approximately stationary at the final checkpoint under both backends (nf4: +0.052; fp16: −0.028). This is consistent with the SFT retention term (weight 0.5) anchoring chosen-completion likelihood. Under fp16, chosen dips to −0.173 at margin3 before recovering. This is a transient decline in Rafailov et al.'s predicted direction, which the retention term dampens but does not fully block. nf4 shows a 2.4× shallower dip at the same checkpoint (−0.072 vs −0.173). Rejected reward falls monotonically under both backends. Chosen's contribution is ≤4% in magnitude at every checkpoint under both backends, so the full margin growth is accounted for by rejected-completion collapse alone. The retention term protects chosen likelihood but not rejected likelihood. Even with chosen explicitly anchored, the behavioral alignment tax remains: router loses 30 points and json loses 3.52. The margin decomposition shows where the margin comes from, not where the damage comes from. Both the DPO gradient on chosen and the gradient on rejected move the weights, and attributing the damage to one channel requires a no-retention control (§5.6). Implications for Cal-DPO and Rafailov et al. are discussed in §5.3. Figure 2 (nf4, training backend) shows the per-seed trajectories.

The per-axis decomposition reveals that per-pair ordering is not backend-invariant. Table 11 reports per-axis means at the final checkpoint under both backends.

**[Table 11: Per-axis reward decomposition at final checkpoint (8-seed mean; 549 training pairs: 378 style, 171 tool_calling)]**

| Axis | Pairs | Margin (nf4, training) | Margin (fp16, dequantized) | r_ch (nf4) | r_rej (nf4) | r_ch (fp16) | r_rej (fp16) |
|---|---|---|---|---|---|---|---|
| Style | 378 | 8.119 | 2.466 | +0.012 | −16.226 | −0.024 | −4.955 |
| Tool_calling | 171 | 4.923 | 4.021 | +0.142 | −9.704 | −0.038 | −8.080 |

Under nf4, the training backend, style dominates on both per-pair margin (8.12 vs 4.92) and pair count. The aggregate margin is 3069 vs 842, a 3.65× ratio. These are the margins the optimizer actually experienced. Under fp16 dequantized-base measurement, the per-pair ordering reverses: tool_calling is higher (4.02 vs 2.47). Style still produces more aggregate margin through pair count (932 vs 688, 1.36×). The divergence is concentrated in the style axis's rejected completions (nf4 −16.2 vs fp16 −5.0; tool_calling −9.7 vs −8.1).

Under both backends, style produces more aggregate margin. The behavioral gradient is consistent with this aggregate pressure pattern: the task most format-orthogonal to the verbose style axis (router) absorbs the most damage.

This per-pair distinction is tested on held-out data. On 98 held-out preference pairs (69 style, 29 tool_calling), tool_calling margin exceeds style margin at every checkpoint under fp16 (8/8 seeds; Appendix F, Table 12). Under nf4, tool_calling leads at margin1–3 (8/8 seeds; Table 14) but style overtakes by margin5, consistent with the axis-dependent inflation measured on training pairs at final. The pre-registered prediction that style margin would exceed tool_calling margin at every checkpoint (Note 2, secondary #3) is falsified under both backends: the prediction holds at 0/4 checkpoints under fp16 and 2/4 under nf4. The final-checkpoint per-pair ordering is backend-dependent.

## 4.4 Gold-Logp Dynamics Are Task-Selective

The reward decomposition shows that chosen completions are approximately stationary on training pairs. We now examine whether gold (correct) completions on evaluation items are similarly stable, and whether this pattern generalizes across tasks.

All Δgold values below are computed in fp16 on Mac; the adapters were trained under nf4 on T4. Adapter weights are stored in full precision regardless of base quantization, so loading nf4-trained adapters onto an fp16 base is valid. Within-backend comparisons are internally consistent; absolute magnitudes may differ from nf4 inference, as discussed below.

Table 6 reports the same-backend (fp16) gold-logp change from SFT baseline to final checkpoint across all three tasks.

**[Table 6: Same-backend fp16 Δgold (gold_logp(final) − gold_logp(SFT)), 8-seed mean]**

| Task | Δgold | 95% CI | Behavioral Δ |
|---|---|---|---|
| Extraction | −0.15 | [−0.18, −0.13] | −0.55 pts (equivalent) |
| Json | −1.04 | [−1.20, −0.89] | −3.52 pts (not equivalent) |
| Router | −1.47 | [−1.62, −1.35] | −30 pts (not equivalent) |

Gold deflation is task-selective. Extraction gold deflates modestly (−0.15 nats/token); json and router gold deflate substantially (−1.04 and −1.47 respectively). The ordering (extraction least, json intermediate, router most) is directionally consistent with the behavioral gradient across all 8 seeds, with non-overlapping confidence intervals between tasks. However, the relationship is highly non-linear: router's gold deflates only 1.4× more than json's, but its behavioral degradation is far larger (−30 vs −3.52 points). Gold-logp magnitude alone does not explain the behavioral gradient. Json shows substantial gold deflation (−1.04) but only minor behavioral loss (−3.52 pts). This suggests that structured output formats with redundant constraints (typed fields, function signatures) may maintain correct greedy-decode behavior despite large probability-mass redistribution. The json loss pattern is consistent with this: 35 of 45 losses land on the one free-text argument (ticker), not on the typed tool-selection field. Router's terse single-token enum format does not provide this task-structural buffer.

The task-selective ordering is robust across quantization backends. We use the 8-seed confirmatory run's own nf4 gold-logp records (step20-to-final), as no nf4 SFT baseline was recorded. The ordering extraction ≪ json < router is identical (nf4 Δgold: extraction −0.008, json −0.22, router −1.59; non-overlapping CIs; Appendix G). Absolute magnitudes diverge substantially, reflecting both the quantization difference and the different baselines. fp16 uses the pure SFT adapter; nf4 uses step20, which has absorbed 20 DPO steps. The divergence is itself task-selective: extraction's deflation nearly vanishes under nf4 (from −0.15 to −0.008), json compresses ~5×, and router's magnitude is essentially unchanged. Whether this reflects quantization-dependent sensitivity, baseline differences, or an interaction between quantization and task-specific token distributions is unresolved. The task-selective ordering, which is the load-bearing claim, holds under both backends.

On extraction specifically, we can go further. We teacher-force gold completions alongside control A (well-formed but with a different section identifier) and control B (well-formed but with a different ticker) through each seed's final checkpoint in fp16. This yields the paired GOLD_PROTECTED statistic (Eq. 5): how much gold deflates relative to incorrect alternatives. Table 7 reports the results.

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

All 8 seeds return GOLD_PROTECTED: every stat exceeds the pre-registered threshold (X = 0.05) with confidence intervals far from zero. On extraction, gold is selectively protected. It declines modestly (−0.10 to −0.23 nats/token) while incorrect controls collapse 2.1× to 3.5× more (−0.34 to −0.47). The relative ordering of correct over incorrect completions is preserved and strengthened.

The paired GOLD_PROTECTED statistic requires task-specific control completions; controls were designed for extraction only (control A = different section identifier; control B = different ticker). On json and router, gold deflation is reported as absolute Δgold (Table 6; nf4 values in Appendix G) without a paired control baseline. Whether gold is relatively protected on these tasks, deflating less than incorrect alternatives, is untested.

Two seeds from a preliminary probe run (seeds 0 and 1, nf4) show no evidence of the catastrophic CHES failure mode. On both seeds, a third control (rejected-register text) falls sharply (−4.6 to −5.1 nats/token). Control C was not included in the 8-seed confirmatory run or the Mac inference runs.

Finally, KL(policy || reference) provides context for the magnitude of policy drift. We compute it as proper token-level KL on extraction gold completions (160 items, fp16, 8 seeds). It rises from 0.022 at margin1 to 0.082 at final, confirming mild policy drift that plateaus as margin continues to grow from ~5 to ~7 (Appendix G). KL is computed on extraction gold completions only; drift on router, the most behaviorally affected task, may be larger.

## 4.5 Backend Sensitivity of QLoRA Margins

Implicit reward margins under QLoRA differ up to 3.4× between nf4 and fp16 dequantized-base measurement of the same adapters. Both backends use the same SFT adapter as reference, scored within each backend. The divergence is axis-dependent: style-axis margins inflate 3.3× (train) to 3.4× (held-out), while tool_calling inflates 1.2× on both splits. Style-rejected completions are long degraded-text sequences, while tool_calling-rejected completions are short tool-swaps. Length and content co-vary with axis, and the cause of the differential inflation is not isolated. Final-checkpoint chosen anchoring is backend-robust, but the mid-training trajectory is not: the margin3 transient is 2.4× deeper under fp16 (−0.173 vs nf4's −0.072).

In our measurements, this is directionally opposite to the gold-logp nf4/fp16 discrepancy (§4.4), where nf4 magnitudes are smaller for extraction (−0.008 vs −0.15) and json (~5×) and comparable for router. However, the gold-logp comparison is baseline-confounded (SFT vs step20), while the margin comparison uses the same SFT adapter scored within each backend. The contrast is therefore suggestive, not established. Chosen-flatness at the final checkpoint and aggregate ordering (style > tool under both backends) are backend-invariant; per-pair ordering is not. Margin-based monitoring of QLoRA-trained models is therefore backend-sensitive. Per-pair and per-axis comparisons require same-backend measurement.

# §5 Discussion

## 5.1 A Two-Level Mechanistic Account

The behavioral and probability-layer findings form a coherent account of how length-normalized DPO affects downstream capability, though the account is not causally proven.

At the training level, the SFT retention term (weight 0.5) anchors chosen-completion likelihood, and the full margin is accounted for by rejected suppression. Rejected reward falls to −14.2 while chosen stays within −0.072 to +0.052 (§4.3). The style axis contributes more aggregate suppression than the correctness axis under both backends (3.65× under nf4, 1.36× under fp16). This is consistent with why the task most format-orthogonal to the verbose style axis (router) absorbs the most damage, while the task with direct tool\_calling coverage (json) absorbs less. Per-pair ordering, by contrast, is backend-dependent: the held-out data shows per-pair tool\_calling margin is higher under fp16 (§4.3, Table 5).

At the probability level, gold-logp deflation is task-selective and directionally consistent with the behavioral gradient. Extraction is the best-preserved task, and its gold completions overlap most with training-pair chosen completions (§3.3). On extraction, gold is selectively protected relative to incorrect controls (8/8 seeds). This is consistent with Ren and Sutherland's squeezing effect concentrating mass onto gold tokens that share entity vocabulary with training pairs. Router's dissimilar terse format may not enjoy this effect. Both backends' chosen minima sit at margin3, coinciding with the behavioral instability window (FORMAT_BREAK 0.196, SURFACE_FORM 0.37). Behavioral metrics are nf4-generated, and the fp16 chosen dip is a dequantized-base measurement, not the source of the behavioral instability. No causal claim is made.

However, the two levels do not form a single causal chain. Gold-logp magnitude is non-linearly related to behavioral outcomes: router's gold deflates only 1.4× more than json's, but its behavioral degradation is far larger (−30 vs −3.52 points). The behavioral gradient is better explained by format-alignment to the style-axis aggregate pressure than by gold-logp dynamics. We present the two levels as complementary observations, without claiming one causes the other. Rejected collapse explains which tasks are pressured; gold-logp dynamics describe how probability mass redistributes.

## 5.2 Falsification Chronology

The pre-registered analysis plan forced three falsifications and two refinements over the course of this study. Each produced a more precise characterization than the hypothesis it replaced. Three additional hypotheses were abandoned during the design phase, before the 8-seed confirmatory run (Appendix C). A causal-clamp experiment was dropped as confounded by KL and total optimization. Router was dropped as a primary endpoint after pilot trajectory analysis revealed a non-monotonic transient. Targeted displacement was dropped after control probes pointed to global drift rather than targeted gold deflation.

Three hypotheses were falsified outright. First, the bounded-null headline (that all three tasks would be preserved within ±5 points) was rejected when the pre-registered falsifier fired on both json and router (§4.1). Second, an apparent extraction capability collapse at margin ≈ 3 was traced to a scorer artifact (see the scorer artifact in §3.2): the apparent errors were surface-form register shifts. The measurement was killed, not refined. Third, the pre-registered prediction that style-axis held-out margin would exceed tool\_calling margin at every checkpoint is falsified under both backends (fp16: 0/4 checkpoints; nf4: 2/4). The final-checkpoint per-pair ordering is backend-dependent (style > tool under nf4, tool > style under fp16). The falsification reveals that per-pair ordering is not backend-invariant; aggregate style dominance holds under both.

Two characterizations were refined. The pilot data showed a "margin-independent early drop" pattern, where router degraded early then stayed flat regardless of margin. This was replaced by a more precise description: a damage window at margin ≈ 1–3 with two distinct processes (transient format-break + monotonic content-error), followed by stabilization (§4.2). The initial "gold barely moves" characterization came from nf4 data in the 8-seed confirmatory run (|Δgold| ≤ 0.01). It was refined to "gold is selectively protected but task-selectively deflates" when fp16 same-backend measurement revealed substantial json and router gold deflation (§4.4).

Each pre-registered decision gate forced a more precise characterization of what actually happened. The result is more nuanced than any single prior framing predicted. It is also more informative than either a clean null or a uniform degradation finding would have been.

## 5.3 Engagement with Prior Work

Our data engages three specific claims from the displacement literature. One caveat applies: our training objective includes an SFT retention term (the RPO formulation; Liu et al., 2024; weight 0.5) on chosen completions. The objective is therefore DPO + retention, not vanilla DPO.

Rafailov et al. predict from their Q-function framework that chosen log-prob should decrease under DPO from an SFT-initialized model. Our training objective includes an SFT retention term on chosen completions, which directly counteracts this decline. The chosen-flatness we observe (−0.072 to +0.052 over 7 units of margin growth) is consistent with this design choice rather than a contradiction of their framework. The chosen-flatness reported in §4.3 is therefore at least partly a consequence of the training objective, not purely an empirical finding about DPO dynamics. It confirms the retention term is effective in this regime. It cannot speak to whether chosen would decline under standard DPO without retention.

The fp16 dequantized-base measurement reveals a transient mid-training chosen dip (−0.173 at margin3, recovering to −0.028 by final). This is a residual decline in the direction Rafailov et al. predict, dampened but not fully blocked by the retention term. Under nf4 the dip is 2.4× shallower (−0.072). The retention term largely counteracts the predicted decline; dequantized-base measurement reveals the residual.

Cal-DPO adds a calibration term to prevent chosen-reward decline. Our regime includes an SFT retention term that achieves a similar anchoring effect by a different mechanism. The finding that chosen is anchored at the endpoint does not demonstrate Cal-DPO's concern is unfounded. It demonstrates the concern is largely addressed by the retention term. The more informative observation is that the behavioral alignment tax remains even with chosen explicitly anchored: router degrades −30 points and json loses 3.52 points. The full margin is accounted for by rejected suppression; which gradient channel produces the behavioral damage is untested.

IPO was explicitly designed to bound the implicit reward margin, treating unbounded growth as the core pathology. Our data shows margin growing to ~7 (β-scaled) with graded behavioral effects. The unbounded growth that IPO constrains produces task-selective damage, not the uniform catastrophe the bounding was designed to prevent. This does not invalidate IPO generally. It characterizes a regime where the damage is concentrated in the margin 1–3 window and is format-selective rather than universal.

## 5.4 Format-Alignment: What It Explains and What It Does Not

The behavioral gradient (extraction preserved → json small loss → router catastrophic) is directionally consistent with format-alignment to the preference data's style axis across all 8 seeds. But with only three tasks, we report this co-occurrence without claiming a causal mechanism.

Two specific limits deserve attention. First, extraction has zero direct preference coverage yet is preserved, so coverage absence alone does not predict degradation. The distinguishing factor may be format-orthogonality: extraction's structured JSON is neither aligned with nor antagonistic to the verbose style. It may also be domain overlap (§3.3; same entities appear in training) or the retention term anchoring chosen completions that share extraction's entity vocabulary. All three co-vary with extraction's preservation and cannot be disentangled with three tasks. Second, json has 200 on-target tool\_calling pairs yet still degrades slightly. The per-axis decomposition explains this: tool\_calling coverage protects against the correctness axis but not against the dominant style-axis aggregate pressure. The damage mechanism (style-rejected suppression) and the protection mechanism (tool\_calling coverage) operate on different axes of the preference data.

Json shows substantial gold deflation (−1.04 nats/token) but only minor behavioral loss (−3.52 points). This suggests that structured output formats with redundant constraints (typed fields, function signatures, required keys) may buffer greedy-decode behavior against large probability-mass redistribution (see §4.1 for the loss breakdown). Extraction is also structured JSON yet deflates far less (−0.15), so format redundancy alone does not explain the difference; format-alignment to the preference data co-varies. With two structured tasks and two co-varying variables, this hypothesis is illustrative, not testable.

## 5.5 Practical Implications

Three implications follow for practitioners running length-normalized DPO on small models with structured-output tasks. (1) Monitor format-break rate at margin ≈ 3, not just endpoint accuracy. The transient spike is where damage concentrates, and it may not be visible at the final checkpoint. (2) Audit preference-data composition for style-axis aggregate dominance. The behavioral gradient is consistent with aggregate training pressure under both backends, though per-pair ordering is backend-dependent and per-axis comparisons require same-backend measurement. (3) Consider including an SFT retention term. Its anchoring is endpoint-level: mid-training chosen can dip transiently under dequantized-base measurement. The no-retention counterfactual is untested.

## 5.6 Limitations

This study operates within explicit scope conditions: sparse LoRA on 8/32 layers (0.034% of parameters; not full fine-tuning), a single 4B model, length-normalized rewards with an SFT retention term (weight 0.5), easily-separable synthetic preferences, and three structured exact-format tasks. We do not study safety or refusal tasks, where Razin et al. demonstrated catastrophic displacement. The preference pairs use synthetic degradation with clean chosen-rejected separation; under naturalistic human preferences with higher chosen-rejected similarity (high CHES scores), the displacement dynamics Razin et al. characterize may produce different outcomes. Results may not generalize beyond these conditions. The no-retention counterfactual (λ = 0 in Eq. 3) is untested; whether chosen declines and whether behavioral damage changes without the retention term are open questions.

Three pre-registered secondaries (#3, #4, #8) were computed post-hoc from saved adapters rather than logged during training: #4 under nf4 (Kaggle T4) with an fp16 dequantized-base measurement (Mac); #3 under fp16 (Mac) and nf4 (Kaggle T4); #8 under fp16 (Mac). Separately, the paired GOLD\_PROTECTED statistic is extraction-only by design (controls A/B are task-specific).

The format-alignment mechanism rests on n = 3 tasks. This is insufficient to distinguish format-orthogonality from coverage, domain overlap, or task difficulty as the causal factor. The checkpoint schedule (margin1, margin3, margin5, final) cannot pinpoint the damage window more precisely than "between margin ~1 and ~3." Gold-logp absolute magnitudes differ substantially between fp16 and nf4 backends. The task-selective ordering is robust across both, but absolute values are not directly comparable (§4.4 and Appendix G).

A complete reconciliation of all pre-registered secondaries against their status (reported, filled post-hoc, or implementation gap) is provided in Appendix C.

# Conclusion

The alignment tax under length-normalized DPO with an SFT retention term is real but graded and task-selective. Router's catastrophic loss is temporally structured. Its onset is at step20 (−8.6 pts, 28% of net loss), and it concentrates in a margin 1–3 damage window where transient format-break and persistent content-error operate as distinct processes. The retention term anchors chosen completions, and the full margin growth is accounted for by rejected-completion suppression. The result is a graded behavioral outcome, with a format-alignment hypothesis consistent with but not established by the three-task gradient. Artifact-aware scoring corrected a false-collapse signal. Pre-registered TOST equivalence testing prevented both false-positive and false-negative capability claims. A disclosed chronology of three falsified hypotheses and two refined characterizations documents how pre-registration forced successive refinement of initially simpler claims. Margin growth to ~7 β-scaled units (nf4 training backend) produces graded rather than uniform degradation. Under the format-alignment hypothesis, this finding offers a differentiated picture (preservation where format-alignment is maintained, substantial risk where it is not) and a methodological template for future alignment-tax studies. These findings are established under specific scope conditions: sparse LoRA on 8/32 layers, a single 4B model, length-normalized rewards with an SFT retention term, and easily-separable synthetic preferences. They may not transfer to full fine-tuning, naturalistic preferences, or safety-critical tasks.

# Appendices

## Appendix A: Full Per-Seed TOST Data

This is the complete data behind Table 1 in the main text.

Per-task **semantic_accuracy** (%) at each checkpoint, all 8 seeds, plus the step-0 SFT baseline. Δ = final − baseline (percentage points). TOST per-seed equivalence band is |Δ| ≤ 5 pp.

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
| json | 4 | 98.75 | 91.88 | 93.75 | 76.88 | 95.62 | 93.75 | -5.00 | PASS |
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

Per-seed TOST pass/fail is shown in the final column of each table; item-level bootstrap 90% CIs at the final checkpoint are reported per task in main-text Table 1.

Seeds 0 and 1 are deterministic re-runs of the pilot (§3.4). Restricting to seeds 2–7: extraction mean Δ = −0.21, json mean Δ = −3.33, router mean Δ = −32.50. All three verdicts are unchanged on the 6-seed subset (seed-level t intervals with df = 5; item-level bootstrap as in the main analysis). Extraction remains equivalent: seed-level 90% CI [−2.18, +1.76], item-level [−1.67, +1.35], and 6/6 seeds within ±5. Json remains not equivalent: its seed-level CI [−4.29, −2.38] and 6/6 seeds are within ±5, but the item-level CI [−5.42, −1.46] crosses −5. Router remains not equivalent: seed-level CI [−38.44, −26.56], 0/6 seeds within ±5.

---

## Appendix B: Scorer v2 Rules and Artifact Walkthrough

### Scorer architecture

Scorer v2 applies a deterministic four-tier classification to each generated output:

1. **FORMAT\_BREAK**: the output cannot be parsed by the task-specific format gate. For router, the gate checks whether the output is a bare string matching one of the five valid labels. For json and extraction, it checks whether the output parses as valid JSON with required keys.
2. **PASS**: the parsed output matches the gold answer after canonicalization.
3. **SURFACE\_FORM**: the parsed output contains the correct semantic content but differs in surface form (e.g., "Item 1" vs "1" for section identifiers).
4. **SEMANTIC\_ERROR**: the parsed, well-formed output has incorrect content.

The primary metric (semantic accuracy) counts PASS + SURFACE\_FORM as correct. FORMAT\_BREAK and SEMANTIC\_ERROR are failures.

### Canonicalization rules (extraction)

The section-identifier field is canonicalized before comparison:
- Extract numeric portion via regex `\d+[A-Z]?` (e.g., "Item 7A" → "7A", "1" → "1", "Item 1" → "1")
- Case-fold the alpha suffix
- Ignore appended titles (e.g., "Item 7: Management's Discussion" → "7")

Ticker and filing\_date fields use exact string match (case-insensitive for ticker).

### The artifact: 0.550 → 0.981

At the margin3 checkpoint, extraction accuracy under strict string matching (no canonicalization) is 0.550. Under scorer v2's canonicalization, the same checkpoint scores 0.981.

The 69 items that flip from strict-FAIL to SURFACE_FORM (semantically correct) are exclusively section-identifier surface-form variants: the model outputs "1" where gold says "Item 1", or "7A" where gold says "Item 7A". The semantic content (which section of which filing) is correct in every case. No ticker or filing\_date errors are involved.

This is not a marginal scorer adjustment. It is the difference between "catastrophic capability collapse at margin ≈ 3" and "capability preserved with a transient register shift." The register-shift rate (SURFACE\_FORM tier) peaks at margin ≈ 3 (mean 0.37 across 8 seeds), then subsides to 0.09 by final. This mirrors router's non-monotonic FORMAT\_BREAK trajectory.

The canonicalization rules were frozen before the 8-seed confirmatory run. No rules were added or modified after observing its results.

---

## Appendix C: Pre-Registration Reconciliation and Falsification Chronology

### C.1 Pre-registered secondary reconciliation

Every pre-registered secondary from Note 2 (frozen 2026-06-18) is accounted for below with its final status.

| # | Pre-registered secondary | Status | Location |
|---|---|---|---|
| 1 | Trajectory-min per seed/task | ✅ Reported | §4.2 (router trough at margin3, 6/8 seeds) |
| 2 | FORMAT\_BREAK trajectory | ✅ Reported | §4.2 (Table 2, non-monotonic) |
| 3 | Held-out margin per category (style vs tool\_calling) | ✅ Reported. **Falsified under both backends**: the prediction (style > tool at every checkpoint) holds at 0/4 checkpoints under fp16 and 2/4 under nf4. Final-checkpoint per-pair ordering is backend-dependent. | §4.3 (Appendix F, Tables 12 and 14) |
| 4 | Chosen/rejected reward decomposition | ✅ Reported (filled post-hoc via dedicated Kaggle GPU inference run) | §4.3 (Tables 3–4) |
| 5 | Gold-logp trajectory × 3 tasks | ✅ Reported (fp16 same-backend, 8 seeds, Mac) | §4.4 (Table 6) |
| 6 | Probe statistic replication (paired) | ✅ Extraction 8/8 GOLD\_PROTECTED (fp16 Mac); json/router: paired stat not computable (controls are extraction-specific) | §4.4 (Table 7) |
| 7 | Register-shift tracking | ✅ Reported (SURFACE\_FORM rate from existing matrix results) | §4.2 |
| 8 | KL-from-reference per checkpoint | ✅ Reported (computed post-hoc from saved adapters, fp16 Mac, 8 seeds) | §4.4 (Appendix G) |

Implementation notes: Secondaries #4 and #8 were not logged during training in the 8-seed confirmatory run. #4 was computed via a dedicated GPU inference run on saved adapters (Kaggle T4). #8 was computed via Mac inference on saved adapters (fp16, proper token-level KL using `forward_full` + `kl_over_positions`). #3 was computed via Mac inference on saved adapters (fp16). All three use the same checkpoint adapters as the confirmatory run; no retraining was involved. The primary TOST analysis (§4.1) was unaffected by these post-hoc computations.

### C.2 Pre-registration deviations

**Checkpoint mechanism.** The pre-registered design (Note 2) specified fixed-step checkpoints with margin as a post-hoc label. The implementation used margin-crossing triggers for intermediate checkpoints (margin1, margin3, margin5), so these checkpoints occur at different optimizer steps across seeds (margin1: steps 24–30, margin3: steps 84–98, margin5: steps 112–138). Step20 and final (step 180) are fixed and identical across seeds. The primary endpoint (step 180) is unaffected. Cross-seed averages at intermediate checkpoints pool seeds at matched margin but different optimizer steps.

**Early-stop configuration.** The pre-registered design specified `max_opt_steps=180` and `no_early_stop=True`. The runtime enforcement was applied in `matrix_kernel.py` (passing `--max-opt-steps 180`) and `run_matrix.py` (passing `no_early_stop=True`), and all 8 seeds terminated at exactly step 180. However, `train_config.json` does not record these runtime parameters (it records module-level constants only), which may create a misleading impression when inspecting saved artifacts. The execution matched the pre-registration; the config-file serialization did not.

**Seed overlap.** Seeds 0 and 1 of the confirmatory set are bitwise re-runs of the pilot seeds (seed 0 fully; seed 1 through margin5). Six of eight confirmatory seeds were unseen at freeze time. The repository was first pushed publicly on 2026-10-18; the commit dates reflect local authorship and are not independently timestamped.

**Backend.** The pre-registration did not specify a quantization backend for any secondary. Secondary #3 (held-out per-category margin) was computed under fp16 (Mac) and nf4 (Kaggle T4); the per-pair ordering is backend-dependent at the final checkpoint (§4.3).

### C.3 Falsification chronology

**Design phase (before the 8-seed confirmatory run):**
- **Causal-clamp experiment dropped.** An experiment to clamp the margin at a fixed value and measure capability was designed but dropped as confounded. Clamping the margin also alters the KL penalty and total optimization pressure, making the causal attribution "margin magnitude → capability loss" untestable.
- **Router-as-primary dropped.** Router was initially the primary endpoint. Pilot trajectory analysis revealed a non-monotonic transient (the margin ≈ 3 format-break spike), making the endpoint accuracy an unreliable summary of the trajectory. Router was moved to a secondary with trajectory decomposition.
- **Targeted-displacement hypothesis dropped.** A preliminary register-shift probe was designed to test whether DPO specifically displaces gold probability mass onto incorrect alternatives (targeted displacement). The probe found that controls fell more than gold (global drift, not targeted displacement). The hypothesis was reformulated as "selective deflation" (gold declines less than controls) rather than "gold doesn't decline."
- **Dissolved numbers.** Three numbers from the exploration phase were found to lack on-disk artifacts and were re-measured or dropped. (a) The 80-item baseline accuracies were never saved and were re-derived from the 160-item suite. (b) The router baseline 0.950 was traced to a conflation between DPO json accuracy and router baseline, and corrected to 0.938. (c) The 160-item gate floor values were unpersisted and were re-derived from source files: extraction 0.006, router 0.200, json 0.0125 (2/160).
- **80 → 160 suite expansion.** The evaluation suite was expanded from 80 to 160 items per task between the pilot and confirmatory phases to improve TOST power. The 160-item suite was frozen before the 8-seed confirmatory run.
- **X = 0.05 threshold.** The GOLD\_PROTECTED threshold (X = 0.05 nats/token) was frozen in the decision-mapping note before a preliminary 12-item probe existed (timestamp-verifiable from commit history).

**Confirmatory-run phase:**
- **Falsified: bounded-null headline.** The pre-registered falsifier fired when json and router both failed the ±5 TOST equivalence test. The task-selective branch governed all subsequent analysis. (§4.1)
- **Falsified: extraction collapse at margin ≈ 3.** Extraction accuracy appeared to crash to 0.550 at margin3 under the strict scorer. Canonicalized scoring revealed this was a surface-form register shift, not a semantic failure (0.550 → 0.981). The strict-scorer measurement was killed, not refined. (§3.2, §4.2)
- **Falsified: style margin > tool\_calling on held-out.** Falsified under both backends: the prediction (style > tool at every checkpoint) holds at 0/4 checkpoints under fp16 and 2/4 under nf4. Final-checkpoint per-pair ordering is backend-dependent. (§4.3)
- **Refined: margin-independent drop → damage window.** The pilot observation of a "margin-independent early drop" on router was refined to a damage window at margin ≈ 1–3 with two distinct processes: transient format-break (non-monotonic) and persistent content-error (monotonic). (§4.2)
- **Refined: gold absolutely stable → gold task-selectively deflates.** The initial characterization from nf4 data in the confirmatory run (|Δgold| ≤ 0.01 on extraction) was refined when fp16 same-backend measurement revealed substantial gold deflation on json (−1.04) and router (−1.47), with extraction deflating least (−0.15). Gold protection is real on extraction (8/8 GOLD\_PROTECTED) but gold deflation is task-selective and non-linearly related to behavioral outcomes. (§4.4)

---

## Appendix D: Suite Hygiene

**Source files:** `data_eval/track1_{router,json,extraction}.jsonl`; leakage reference `kaggle_upload/training_data/dpo_pytorch.jsonl`; degeneracy floors `kaggle_upload/audit_eval.py:91`; gold self-consistency via `kaggle_upload/scorer.py` (gold-as-output).

| Property | extraction | json | router |
|----------|------------|------|--------|
| n items | 160 | 160 | 160 |
| Unique prompts | 160 | 160 | 160 |
| Duplicate prompts | 0 | 0 | 0 |
| Verbatim leakage (eval∩train) | 0 | 0 | 0 |
| Gold self-consistency (gold-as-output PASS) | 160/160 | 160/160 | 160/160 |
| Degeneracy floor (always-one-answer) | ~0.006 (1/160) | 0.0125 (2/160; 18 duplicate tool-call tuples, maximum multiplicity 2) | 0.20 (1/5 classes) |

**Class balance (items per class):**

- **router** (5 classes): COMPANY_DEEP_DIVE=32, FILING_ANALYSIS=32, FILING_DIFF=32, OUT_OF_SCOPE=32, SIMPLE_QUERY=32.
- **json** (8 tool classes): analyst_ratings=20, company_metrics=20, company_news=20, diff_filing_sections=20, get_filing_section=20, insider_trades=20, search_filings=20, stock_price=20.
- **extraction** (6 section classes): Item 1=27, Item 1A=27, Item 3=26, Item 7=27, Item 7A=26, Item 8=27.

**Factual statements:**

- Zero verbatim leakage: 0/160 eval prompts appear as a training `question` (exact-string match) for all three tasks. Source: `kaggle_upload/audit_eval.py:77` logic, `dpo_pytorch.jsonl` vs `data_eval/*.jsonl`.
- Gold self-consistency: 160/160 gold answers pass their own scorer for all three tasks. Source: `kaggle_upload/scorer.py`.
- Zero duplicate prompts: 160 unique prompts / 160 items for all three tasks. Source: `data_eval/*.jsonl`.
- Degeneracy floors (a constant always-one-answer policy): extraction ~1/160≈0.006, json 0.0125 (2/160); router 0.20 (5 balanced classes). Source: `kaggle_upload/audit_eval.py:91`.

---

## Appendix E: Training Configuration and Hash Verification

**SFT checkpoint:** hash 29a3794b…585f02, verified identical across all 8 seed runs. Adapter weights loaded from `adapters/sft_pytorch/adapter_model.safetensors`.

**Tokenizer and chat template:** the chat template is byte-identical across sources (SHA-256 hash comparison). For the tokenizer, tokenization is identical on all study inputs: the SFT adapter, DPO adapters, and base tokenizer configs differ in metadata (added-token count, pre-tokenizer regex) but produce identical token IDs on all 6,024 evaluation and preference strings, verified by direct token-ID comparison.

**LoRA configuration:**
- r = 8, α = 16, dropout = 0.05
- target\_modules = [q\_proj, k\_proj, v\_proj, o\_proj] (explicitly configured, not peft default)
- bias = none
- peft version 0.19.1
- Injected layers: 3, 7, 11, 15, 19, 23, 27, 31 (the 8 full-attention layers in Qwen3.5-4B's hybrid architecture)

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

Source: `kaggle_result/reward_decomposition_kernel/reward_decomposition/summary.json` (β=0.5, reference=sft_baseline). r = policy_logp − sft_logp (length-normalized); margin = β·(r_chosen − r_rejected).

### F.1: r_chosen, r_rejected, margin per checkpoint

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

### F.2: Per-axis margin at final checkpoint (style vs tool_calling)

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

### F.3: Held-out margin by category per checkpoint (Table 12; fp16; 69 style, 29 tool_calling)

Source: `analysis_ws/supplementary/heldout_margin_per_category.json` (n_pairs=98 held-out: 69 style, 29 tool_calling; β=0.5, reference=sft_baseline).

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

### F.4: Held-out margin by category per checkpoint (Table 14; nf4; 8-seed mean)

Source: `kaggle_result/val_nf4_98.json` (8-seed mean, 98 held-out pairs, β=0.5, reference=sft_baseline).

**[Table: Held-out per-category margin under nf4 (98 pairs: 69 style, 29 tool_calling). Tool_calling leads through margin3; style overtakes by margin5 under axis-dependent nf4 inflation.]**

| Checkpoint | Style (nf4) | Tool (nf4) |
|------------|-------------|------------|
| margin1 | 0.780 | 1.466 |
| margin3 | 2.620 | 3.426 |
| margin5 | 5.435 | 4.020 |
| final | 7.919 | 4.753 |

### F.5: fp16 train decomposition, pooled per checkpoint (8-seed mean)

Source: `kaggle_result/fp16_train_decomposition.json` (fp16, Mac; β=0.5, reference=sft_baseline; pooled over all 549 training pairs, then averaged across seeds 0–7).

| Checkpoint | r_chosen | r_rejected | margin |
|------------|----------|------------|--------|
| margin1 | −0.045 | −1.785 | 0.870 |
| margin3 | −0.173 | −4.661 | 2.244 |
| margin5 | −0.126 | −5.743 | 2.809 |
| final | −0.028 | −5.928 | 2.950 |

*fp16 train reward decomposition, pooled over 549 pairs (378 style, 171 tool_calling). Chosen dips transiently at margin3 (−0.173) before recovering; rejected falls monotonically.*

---

## Appendix G: Gold-Logp Per-Seed Data, nf4 Cross-Check, and KL Trajectory

### G.1 fp16 same-backend Δgold (per-seed)

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

### G.2 nf4 cross-check (moved from main text)

nf4 Δgold is computed as gold\_logp(final) − gold\_logp(step20), using the gold-logp records of the 8-seed confirmatory run. No nf4 SFT baseline gold-logp was recorded; step20 is the earliest available reference.

**[Table: nf4 Δgold (step20→final), 8-seed mean]**

| Task | Δgold (nf4) | 95% CI | Δgold (fp16) for comparison |
|---|---|---|---|
| Extraction | −0.008 | [−0.014, −0.003] | −0.15 |
| Json | −0.22 | [−0.34, −0.11] | −1.04 |
| Router | −1.59 | [−1.99, −1.19] | −1.47 |

The task-selective ordering (extraction ≪ json < router) is identical under both backends, with non-overlapping CIs. Absolute magnitudes diverge, reflecting both the quantization difference and the different baselines (fp16 uses the pure SFT adapter; nf4 uses step20, which has absorbed 20 DPO steps). The divergence is itself task-selective: extraction's deflation nearly vanishes under nf4 (from −0.15 to −0.008), json compresses ~5×, while router's magnitude is essentially unchanged.

### G.3 GOLD\_PROTECTED paired statistic (fp16, extraction, per-seed)

Source: `analysis_ws/gold_controls/gold_controls_summary.json` (method: fp16 unquantized, paired stat Dgold-Dmean(A,B), reference=sft_baseline, X=0.05). stat = Δgold − mean(ΔctrlA, ΔctrlB); GOLD_PROTECTED verdict when stat ≥ X.

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

**8-seed mean val_margin at step 150 = 6.087** (rounds to 6.1).

> **Val-subset note:** the training-time `val_margin` logged in the margin files is computed over a fixed 20-pair subset (15 style, 5 tool\_calling) of the 98 held-out pairs under the nf4 training backend, not the full 98-pair set. Post-hoc fp16 recomputation on the same 20 pairs yields a mean of 2.987 at the final checkpoint, versus 7.519 under nf4: a 2.52× backend ratio. The logged `val_margin` values are therefore not comparable to the fp16 held-out margins reported in Table 12.
