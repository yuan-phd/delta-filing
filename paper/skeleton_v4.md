# FINAL Paper Skeleton (v4): Selective Deflation, Not Displacement — Characterizing the Alignment Tax Under Length-Normalized DPO

*Working title (alt: "The Alignment Tax Is Graded: Selective Deflation Under Length-Normalized DPO"). Incorporates all review rounds. Ready to draft.*

---

## Abstract (~150 words, one paragraph, no bullets)

Length-normalized DPO drives the implicit reward margin to ~7 nats/token in a 4B+LoRA model trained on structured exact-format tasks. The dominant framing — displacement theory, IPO's design motivation — predicts this should degrade capability. In a pre-registered 8-seed study with artifact-aware scoring, we find the alignment tax is real but graded: ranging from undetectable (extraction, Δ = −0.5, TOST PASS) through small but statistically genuine (json, Δ = −3.5, one-directional loss, TOST boundary FAIL) to catastrophic (router, Δ = −30), tracking format-alignment to the preference data rather than margin magnitude. The mechanism: margin growth is driven entirely by rejected-completion collapse (to −14.2), not chosen displacement (+0.05 flat); selective deflation protects gold completions (8/8 seeds) while incorrect controls collapse 2–3× more, preserving the ordering greedy decoding needs. A naive scorer masked this picture, manufacturing a false capability collapse (0.550 vs 0.969 canonical) that artifact-aware scoring and pre-registered TOST equivalence testing — to our knowledge unprecedented as the primary evaluation framework for DPO capability preservation — prevented. Scope: LN rewards, LoRA, 4B model, easily-separable synthetic preferences.

---

## 1. Introduction (~1 page)

**Opening:** DPO's implicit reward margin grows without bound under length-normalized rewards. Displacement theory (Razin et al.) predicts chosen-likelihood decline and mass flow to OOD completions. IPO was explicitly designed to bound the margin, treating unbounded growth as the core problem. Cal-DPO observes absolute deflation and assumes performance degradation. Yet empirical evidence in the common practical regime — small model, LoRA, structured tasks — is thin, and the literature lacks equivalence testing to distinguish "preserved" from "underpowered."

**This paper:** 8-seed pre-registered study measuring behavioral capability (free-generation, artifact-aware scoring) and probability-layer dynamics (gold/control log-probs) as margin grows to ~7 nats. We find the tax is real but graded, driven by rejected suppression not chosen displacement, and survivable with minor erosion where format-alignment is maintained.

**Contributions:**
1. Graded, task-selective degradation with non-monotonic dynamics (behavioral)
2. Rejected-collapse-driven margin growth with style-axis dominance (training dynamics)
3. Selective deflation with gold protection, 8/8 seeds (probability layer)
4. Artifact-aware scorer + pre-registered TOST framework (methodology)

**Scope (what we study AND what we do not):** LN rewards, LoRA, 4B model, easily-separable synthetic preferences, three structured exact-format tasks. We do NOT study safety/refusal tasks (where Razin et al. demonstrated catastrophic displacement), full fine-tuning, naturalistic human-written preferences, or models above 4B. These are explicit boundaries.

---

## 2. Related Work (~0.75 page — 5 essential citations + "Additional" paragraph)

**Core 5 (explicit engagement):**

1. **Razin et al. (ICLR 2025) — displacement.** Chosen log-prob decreases, mass may flow to harmful completions. Their caveat: "not necessarily problematic." Our data: chosen is flat; the mechanism is rejected collapse, not chosen displacement.

2. **Ren & Sutherland (ICLR 2025) — squeezing.** Negative gradients on rejected dominate, concentrating mass onto highest-probability token. Our data is textbook squeezing (rejected −14.2, chosen +0.05). This is the mechanistic bridge connecting rejected collapse (§4.3) to gold protection (§4.4).

3. **Cal-DPO (NeurIPS 2024) — ordering preservation.** Observed ordering preserved + absolute deflation, assumed harm (OOD mass). Our counter-evidence: chosen is flat, deflation is selective, the ordering preservation Cal-DPO observes is behaviorally benign in our regime.

4. **IPO (Azar et al. 2024) — bounding the margin.** Explicitly designed to fix unbounded margin growth via regression target. Our data: margin grows to ~7 with graded effects and chosen flat. The problem IPO addresses does not manifest here (not invalidated generally, unnecessary in this regime).

5. **Rafailov et al. (2024) — Q-function view.** Predicts chosen log-prob should decrease from SFT start. Our data: chosen is approximately stationary. "Whether this reflects length-normalization, LoRA capacity constraint, or SFT data composition is an open question."

**Additional (one paragraph, inline cites):** OXRL study (20 variants, none beats vanilla — our protocol explains why); Chen et al. NeurIPS 2024 (preference rankings preserved); MoE-Sieve (TOST precedent in SFT, not DPO); Pareto-Aware Dual-Preference Optimization (Feb 2026, concurrent TOST use in DPO as incidental validation — we use it as pre-registered primary framework); "What Is the Alignment Tax?" (2603.00047, theoretical frame); Viscosity of Logic (2601.17260, margin anticorrelates with reasoning, single-author preprint — supports our finding from a different angle, varies β not within-run margin); format/content decomposition (2512.17920); Minerva answer-normalization.

---

## 3. Method (~1 page for workshop)

### 3.1 Setup
- Model: Qwen3.5-4B + LoRA (r=8, α=16, q/k/v/o_proj, peft 0.19.1)
- Training: vanilla LN-DPO, β=0.5, batch-8, 180 fixed steps, no early stop
- Preference data: 647 pairs (549 train / 98 held-out), 31% correctness (200 tool_calling pairs, length-matched ~28 tok) + 69% style (447 pairs, chosen ~188 tok verbose vs rejected ~84 tok). LN prevents style pairs from being winnable on length alone.
- SFT checkpoint: hash-verified identical across all runs (29a3794b…585f02); tokenizer/chat-template byte-identical (hash-verified)
- 8 seeds, pre-registered split (0–4 window 1, 5–7 window 2, no peeking), divergence guard (NaN or margin>20; 0/8 triggered)

### 3.2 Tasks and evaluation suite

**Task descriptions (one sentence each — essential for understanding format-alignment):**

| Task | Format | Output example | Preference coverage |
|---|---|---|---|
| **json** | Structured JSON tool call (function name + typed arguments) | `{"name":"get_filing","args":{"ticker":"AAPL",...}}` | On-target (200 tool_calling pairs) |
| **router** | Terse single-word enum from {filing_type_A, ..., filing_type_E} | `filing_type_C` | Off-target (zero coverage) |
| **extraction** | Three-field JSON: {ticker, section, filing_date} | `{"ticker":"MSFT","section":"Item 7","filing_date":"2024-02-15"}` | Off-target (zero coverage) |

- 160 items per task, free generation (greedy decode, no constrained decoding, no judge)
- **Baseline is deterministic:** baseline_v2 (step-0 SFT, greedy decode) has no eval variance; all comparison variance comes from training seeds + item-level variation.
- Scorer v2 (frozen canonicalization): three-tier PASS / SURFACE_FORM / SEMANTIC_ERROR. Primary metric: semantic accuracy.
  - **The artifact rescue:** "Under strict matching, extraction at margin3 appears to crash to 0.550; under canonicalized scoring, it is 0.969 — 93% of apparent errors are surface-form register shifts ('Item 1' vs '1'), not semantic failures." *(Callout box in paper, not a separate figure.)*

### 3.3 Suite independence and domain overlap
- Zero verbatim leakage (exact-string match)
- Domain overlap by design (both derive from EDGAR): ≥81/160 extraction items share ticker×section with a preference pair (lower bound; provenance recoverable for 209/647); ≥131/160 json share a ticker
- "This overlap means extraction's preservation cannot be attributed solely to format-robustness. Router (zero domain overlap) provides the cleanest independence test."
- Hygiene: degeneracy floors re-derived (extraction 0.0063, router 0.20, json 0.013), class balance verified, gold self-consistency 160/160

### 3.4 Pre-registered analysis plan
- TOST ±5 pts, semantic accuracy at final step vs baseline_v2
- **±5 justification (honest):** "We chose ±5 pts as an a priori bound that is large enough to avoid trivially passing (n=160 items × 8 seeds gives adequate power to detect 5-pt differences) yet small enough that passing is substantively meaningful (a 5-pt drop on 160 items = 8 item flips, which we judged non-negligible). The bound was frozen before any matrix data existed. We acknowledge this is researcher judgment, not derived from an external standard."
- CI: seed-level t-based (df=7) + item-level bootstrap; both within ±5
- ≥7/8 seeds within bound; 3/3 tasks → bounded-null; any fail → task-selective
- No multiple-comparison correction: headline requires ALL tasks (intersection-union test, inherently conservative)
- Step20 diagnostic-only; all mappings frozen pre-data (Notes 1+2, date-stamped)

### 3.5 Probability-layer measurement
- Gold + control A/B teacher-forced through checkpoints; primary stat = Δgold − Δmean(A,B); X=0.05
- Two backends: nf4 (8-seed absolute gold-logp from matrix) + fp16 (8-seed paired stat from inference)
- Frozen decision mapping (DECISION_NOTE_register.md)

---

## 4. Results (~1.5 pages for workshop)

### 4.1 Finding 1: Graded, task-selective degradation

| task | baseline | mean final | Δ | seed 90% CI | item 90% CI | seeds ≤±5 | verdict |
|---|---|---|---|---|---|---|---|
| extraction | 97.50 | 96.95 | −0.55 | [−2.07, +0.98] | [−1.80, +0.86] | 8/8 | **EQUIVALENT** |
| json | 98.75 | 95.23 | −3.52 | [−4.56, −2.47] | [−5.31, −1.95] | 6/8 | **NOT EQUIV** |
| router | 93.75 | 63.59 | −30.16 | [−35.65, −24.66] | [−34.92, −25.47] | 0/8 | **NOT EQUIV** |

- Falsifier triggered → task-selective. Json: real one-directional (0/45 gains, p=5.4e-11), ceiling-flagged. seed0 −6.25 outside, seed4 −5.00 exactly at bound.
- Framing: "The alignment tax is real, graded, and task-selective. The gradient is *consistent with* the task's format-alignment to the preference data, but with three tasks we report the co-occurrence without claiming the causal link is established."
- Json's partial coverage (200 tool_calling pairs) provides partial but not complete protection — explained by style-axis aggregate dominance (§4.3).

### 4.2 Finding 2: Non-monotonic dynamics with two processes

| ckpt | margin | sem_acc | FB rate | content-error |
|---|---|---|---|---|
| step20 | 0.75 | 0.852±0.015 | 0.001±0.002 | 0.148±0.016 |
| margin1 | 1.09 | 0.816±0.030 | 0.007±0.010 | 0.177±0.038 |
| margin3 | 3.12 | 0.587±0.061 | 0.196±0.089 | 0.217±0.071 |
| margin5 | 5.13 | 0.626±0.049 | 0.133±0.096 | 0.241±0.083 |
| final | 6.73 | 0.636±0.077 | 0.106±0.070 | 0.258±0.074 |

- **Observation (findings):** FB non-monotonic (peaks margin≈3), content-error monotonic. Trough 8/8 seeds at margin≈3 (mean 0.587, range 0.444–0.650). Endpoint: mean 0.636, median 0.638, range 0.487–0.769. Checkpoint resolution: "cannot pinpoint more precisely than margin ~1–3."
- **Interpretation (discussion only):** transient destabilization + persistent content erosion.

**Figure 1:** Per-seed margin-vs-capability trajectories. Extraction/json tight bands, router wide fan.

### 4.3 Training dynamics: rejected collapse and style-axis dominance

| ckpt | r_chosen | r_rejected | margin |
|---|---|---|---|
| margin1 | −0.052 | −2.113 | +1.031 |
| margin3 | −0.072 | −5.990 | +2.959 |
| margin5 | −0.004 | −10.156 | +5.076 |
| final | +0.052 | −14.194 | +7.123 |

- Chosen flat; entire 7-unit margin = rejected collapse.
- Cal-DPO contrast: "Cal-DPO assumes chosen-reward decline is the problem; in our regime, chosen is flat and the entire margin is rejected collapse."

**Per-axis (framing sentence BEFORE the table):** "The relevant quantity for behavioral impact is aggregate suppression pressure (total margin contributed by each axis), not per-pair potency, because weight updates accumulate over all pairs."

| axis | pairs | margin | r_chosen | r_rejected |
|---|---|---|---|---|
| style | 447 | +8.119 | +0.012 | −16.226 |
| tool_calling | 200 | +4.923 | +0.142 | −9.704 |

- Style dominates in aggregate (65% more total margin) due to 2.2× more pairs. Per-pair, tool_calling is slightly more potent. Behavioral gradient tracks aggregate pressure.
- Json's small degradation despite coverage: "tool_calling coverage provides partial but not complete protection against the dominant style-axis aggregate pressure."

**Figure 2:** Chosen (flat) vs rejected (collapsing) trajectories, with axis split.

### 4.4 Finding 3: Selective deflation with gold protection (8/8 seeds)

| seed | stat | 95% CI | Δgold | Δctrl |
|---|---|---|---|---|
| range | +0.213 to +0.295 | all exclude zero | −0.10 to −0.23 | −0.34 to −0.47 |
| mean | +0.247 | — | −0.153 | −0.400 |

- 8/8 GOLD_PROTECTED; all stats far above X=0.05.
- "Gold is selectively protected, not absolutely stable. Gold declines modestly; controls collapse 2–3× more. The finding is the differential, not absolute immunity."
- Quantization: "Gold-logp absolute change is ≤0.01 in nf4 and −0.10 to −0.23 in fp16. Whether the difference reflects quantization-dependent protection or measurement resolution is open; the relative finding is consistent across both backends."
- No CHES evidence.

**Figure 3:** Paired stat bar chart, 8 seeds, all above X=0.05 line.

---

## 5. Discussion (~0.5 page for workshop)

### 5.1 Unified mechanistic account
Squeezing (Ren & Sutherland) → rejected collapse → selective deflation → ordering preserved → graded behavioral gradient. **Language: "consistent with," "accounts for" — not "proves," "demonstrates that."** The individual pieces are each well-evidenced; the chain connecting them is interpretive.

### 5.2 Falsification chronology (narrative spine)
- **Falsified:** bounded-null (falsifier fired); extraction margin3 collapse (scorer artifact killed)
- **Refined:** margin-independent drop → damage window at margin ~1–3; gold absolutely stable → gold selectively protected
"Each pre-registered decision gate forced a more precise characterization."

### 5.3 Disagreements with prior work
- Rafailov: chosen should decrease → flat. Open question.
- Cal-DPO: chosen decline is the problem → rejected collapse is the mechanism.
- IPO: unbounded margin harmful → not in this regime.

### 5.4 Format-alignment: what it explains and what it doesn't
Coverage absence alone doesn't predict degradation (extraction is the counterexample). Three tasks, correlation only.

### 5.5 Practical implications
(1) Monitor FB rate at margin≈3; (2) audit preference-data composition for style-axis aggregate dominance.

### Limitations (end of Discussion)
- Single model/scale (4B, LoRA)
- Easily-separable synthetic preferences
- n=3 tasks, correlation not causation
- Domain overlap (extraction); router = cleanest independence test
- Implementation gap: three secondaries staged but not connected to matrix kernel; two filled post-hoc (inference-only); one (per-category held-out margin) limitation. "Implementation gap, not post-hoc exclusion."
- Checkpoint resolution: margin ~1–3, not more precise
- nf4 vs fp16 gold-logp discrepancy unresolved

---

## 7. Conclusion (~3 sentences)

The alignment tax under length-normalized DPO is real but graded, task-selective, and mechanistically explicable through selective deflation driven by rejected-completion collapse. Careful measurement — artifact-aware scoring and pre-registered TOST equivalence testing — prevented false conclusions at every stage, and a disclosed chronology of two falsified hypotheses and two refined characterizations documents how pre-registration forced discovery of a more nuanced picture than any single prior framing predicted. The finding that margin growth to ~7 nats/token is largely survivable where format-alignment is maintained offers both a practical reassurance and a methodological template for future alignment-tax studies.

---

## Figures (3 main body)
1. Per-seed margin-vs-capability trajectories (signature image)
2. Reward decomposition: chosen flat vs rejected collapse, axis split
3. Gold-logp paired stat across 8 seeds (bar chart + X=0.05 line)

Scorer artifact (0.550→0.969) = callout box in §3.2.

## Appendix
A: Full per-seed TOST tables | B: Scorer v2 rules + artifact walkthrough | C: Falsification chronology (dated) | D: Suite hygiene | E: Config + hash chain | F: Reward decomposition per-seed | G: Gold-logp per-seed (fp16 + nf4) | H: Val margin trajectories
