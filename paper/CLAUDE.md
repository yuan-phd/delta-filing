# claude.md — Paper Editing Guide

## Project

This is a workshop paper (target: NeurIPS 2026 workshop, 4–6 pages + references + appendix) studying alignment tax under length-normalized DPO. The content is authored and finalized before it reaches you. Your job is **style, formatting, and consistency** — not content creation or claim modification.

## Paper Structure (reference)

1. Abstract (~150 words, one flowing paragraph)
2. §1 Introduction (~1 page)
3. §2 Related Work (~0.75 page: 5 core citations + 1 "Additional" paragraph)
4. §3 Method (~1 page: setup, tasks, suite independence, analysis plan, probability measurement)
5. §4 Results (~1.5 pages: 4.1 graded degradation, 4.2 non-monotonic dynamics, 4.3 training dynamics, 4.4 gold protection)
6. §5 Discussion + Limitations (~0.75 page)
7. §7 Conclusion (~3 sentences)
8. Appendix (A–H)

3 figures in main body. Total main body ≤ 6 pages excluding references and appendix.

## Style Rules (critical — apply uniformly)

### Claim calibration
- The mechanistic chain (squeezing → rejected collapse → selective deflation → ordering preserved → graded behavioral gradient) is **interpretive, not proven causal**. 
- Use: "consistent with," "accounts for," "explains," "suggests"
- Never use: "demonstrates that," "proves," "establishes," "confirms" (for the chain as a whole)
- Individual data points CAN use strong language: "8/8 seeds show GOLD_PROTECTED" is a fact, not interpretation.

### Hedge discipline
- n=3 tasks → "we report the co-occurrence without claiming the causal link is established"
- Format-alignment mechanism → "consistent with," never "driven by" or "caused by"
- Gold protection → "selectively protected," never "gold barely moves" or "gold is immune" (fp16 shows real decline)
- nf4 vs fp16 discrepancy → "open question," not "nf4 is stronger"

### Terminology (use these exact terms consistently)
| Concept | Correct term | Do NOT use |
|---|---|---|
| The ±5 TOST result | "TOST equivalence test" | "significance test," "hypothesis test" |
| Router's worst checkpoint zone | "damage window at margin ≈3" | "danger zone" |
| FB spike then partial recovery | "non-monotonic" (findings), "transient destabilization" (discussion only) | "collapse and recovery" |
| Gold logp finding | "selectively protected" | "barely moves," "immune," "stable" |
| Margin growth mechanism | "rejected-completion collapse" | "displacement" (unless referring to Razin's framework) |
| Style axis effect | "aggregate suppression pressure" | "style is stronger" (per-pair, tool_calling is slightly more potent) |
| Json's status | "small loss on a clustered item-level interval" | "preserved," "marginally preserved" |
| The chronology | "three falsifications and two refinements" | "four failures," "four overturned hypotheses" |
| TOST novelty | "unprecedented as pre-registered primary framework for DPO capability preservation" | "novel in DPO literature" (a Feb 2026 paper uses TOST incidentally) |

### Tone
- Confident on data, humble on interpretation.
- No hedging on things that ARE established (8/8 GOLD_PROTECTED, 0/45 json gains, extraction TOST PASS).
- No apologizing for limitations — state them factually. "We acknowledge" not "unfortunately."
- No performative modesty ("our humble contribution"). State what the paper does.

### Formatting
- Tables: clean, minimal borders, aligned decimals.
- Figures: referenced as "Figure 1/2/3" in text, each introduced with a sentence before the reference.
- CIs: always in brackets with ± or [lower, upper] notation, consistent throughout.
- Numbers: percentages as "97.50" not "97.5%"; nats/token always written out; margin values without units (they are β-scaled, dimensionless).
- p-values: scientific notation for small values (p = 5.4e-11, not p < 0.001).
- Seeds: "seed0" through "seed7" or "seeds 0–7" (not "Seed 0" or "seed #0").

### What NOT to change
- Do not soften hedges that are already calibrated (they were reviewed multiple times).
- Do not add interpretation beyond what the content provides.
- Do not remove caveats or scope conditions.
- Do not rephrase the ±5 justification — the wording ("researcher judgment, not derived from an external standard") is deliberate.
- Do not merge the falsification chronology's "2 falsifications + 2 refinements" into a uniform category.
- Do not change the per-axis framing sentence ("aggregate suppression pressure, not per-pair potency") — it prevents a specific reviewer misread.

## LaTeX

- Template: NeurIPS 2026 workshop (neurips_2026.sty or equivalent; check the accepted workshop's requirements when available)
- BibTeX for references
- Figures: PDF or PNG, placed at top of columns
- Callout box for the scorer artifact (0.550 → 0.969) in §3.2 — use a tcolorbox or framed environment, not a separate figure
