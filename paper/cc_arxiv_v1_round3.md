# arXiv v1 Round 3: Language and Equations

## Context

Final language and formalization pass before posting. No claims, numbers, or hedges change. Read `paper/CLAUDE.md` first. Apply to both `paper/full_draft.md` and the .tex files.

## Part A: Global search-and-replace

Apply throughout main text and appendices unless noted.

**A1. Bare "chosen"/"rejected" as nouns.** On first use in each section (and always in abstract and §1), write "chosen reward" / "chosen likelihood" / "rejected reward" as appropriate. After first use in a section, bare "chosen"/"rejected" is acceptable.

**A2. "gradient" for task ordering.** The paper uses "behavioral gradient" and "gradient of degradation" for the cross-task ordering while also discussing optimization gradients. Replace with "degradation ordering" or "task ordering" in: abstract, §4.1, §4.4, §5.1, §5.4. Keep "gradient" only for optimization gradients.

**A3. Stacked hyphenated modifiers.** Unpack these wherever they appear:
- "dequantized-base measurement" → "measurement on the dequantized base"
- "rejected-completion suppression" → "suppression of rejected completions"
- "style-axis aggregate pressure" → "aggregate pressure from the style axis"
- "orthogonal-but-non-antagonistic format" → "a format that is orthogonal to but not antagonistic with the style axis"
- "task-structural buffer" → "a buffer provided by task structure"
- "format-axis-alignment finding" → "the format-alignment finding"
- "chosen-completion likelihood" → "likelihood of the chosen completion"
Keep two-word compounds (e.g., "held-out", "per-pair", "pre-registered").

**A4. Inconsistent forms.** Standardize: "json", "router", "extraction" (lowercase, except sentence-initial); "tool_calling" (lowercase, underscore); "step 20" (with space); "nf4" (lowercase); "versus" in prose, "vs" only in tables.

**A5. "8-seed confirmatory run".** Define once in §3.1 as "the confirmatory run (8 seeds)" and thereafter use "the confirmatory run".

**A6. Log-style verdicts.** In §1, replace "TOST equivalence PASS" and "TOST boundary FAIL" with the abstract's wording: "equivalent at ±5" and "equivalence fails on the item-level interval".

**A7. Scare quotes.** Remove quotation marks around ordinary phrases: "between margin ~1 and ~3", "gold barely moves", "margin-independent early drop". Keep quotes only when naming a hypothesis or quoting a source.

## Part B: Specific sentence fixes

**B1. Abstract, last Findings sentence.** "axis-dependently" → "and the gap depends on the preference axis".

**B2. Abstract.** "the source of the behavioral damage is not attributed" → "we do not attribute the behavioral damage to either gradient channel".

**B3. §1 paragraph 4.** The sentence "the pre-registered prediction that style would dominate held-out per-pair margin at every checkpoint was falsified under both backends (fp16 0/4; nf4 2/4)" appears before backends, held-out pairs, or checkpoints are defined. Replace with: "a pre-registered prediction about which preference axis dominates per-pair margin was falsified (§4.3)".

**B4. §1 contribution (1).** Split the three-semicolon sentence into three: the finding; what is accounted for; what is left open.

**B5. §1 last sentence.** Delete "These are stated boundaries, not implicit weaknesses."

**B6. §2 Cal-DPO.** "does not demonstrate Cal-DPO's concern is unfounded, but that it is already addressed" → "does not show that Cal-DPO's concern is unfounded; it shows that the concern is already addressed by the retention term".

**B7. §2 squeezing.** "is textbook squeezing" → "is consistent with squeezing".

**B8. §2 equivalence.** Replace the "unprecedented in the DPO literature ... for DPO capability assessment" sentence with: "We are not aware of prior DPO work that uses pre-registered TOST as its primary capability test."

**B9. §3.1 units paragraph.** "each rejected token is on average e^14 times less likely, relative to the reference, than each chosen token" → "relative to the reference, the policy assigns each rejected token on average e^14 ≈ 1.2 × 10^6 times less probability than each chosen token (a geometric mean over positions)".

**B10. §3.1 tokenizer sentence.** Split after "study inputs".

**B11. §3.4 first sentence.** Replace the ±5 justification with: "We chose ±5 a priori. It is large enough that a true Δ of 0 passes with power above 0.99 (post hoc, from the observed per-seed SDs of 2.28 on extraction and 1.56 on json), and small enough that passing is meaningful: 5 points is 8 of 160 items. We acknowledge this is researcher judgment, not derived from an external standard."

**B12. §4.1 json paragraph.** Split into three paragraphs: (i) the verdict and why the item-level CI fails; (ii) the clustering (17 items, five items carry 28 losses); (iii) the loss anatomy (35 wrong-ticker, 7 wrong-tool, 3 format breaks). Replace "The seed count is not the cause, even though only 7 of 8 seeds fall within the bound" with "The per-seed criterion is met (7 of 8 within the band): seed 0 is outside at −6.25 and seed 4 sits on the inclusive boundary."

**B13. §4.2 final sentence.** "The variance structure itself communicates the finding" → "The per-seed spread is itself the result: damage depth is consistent, recovery is not."

**B14. §4.3 first paragraph.** Three sentences say the same thing (rejected falls monotonically; chosen contributes ≤4%; retention protects chosen but not rejected). Keep the ≤4% sentence, cut the other two. Also: "Both the DPO gradient on chosen and the gradient on rejected move the weights" → "The DPO gradient on chosen, the DPO gradient on rejected, and the SFT gradient on chosen all move the weights".

**B15. §4.4.** "the catastrophic CHES failure mode" → "the catastrophic displacement mode that Razin et al. flag with high CHES scores".

**B16. §5.1.** "consistent with why the task most format-orthogonal" → "consistent with the observation that the task most orthogonal in format". "may not enjoy this effect" → "may not benefit from this effect".

**B17. §5.2.** "The measurement was killed, not refined" → "The measurement was discarded, not refined."

**B18. Conclusion.** Cut to the three findings plus scope, roughly half the current length. Do not add new claims.

## Part C: Promote reported quantities to displayed equations

Replace the prose "Reported quantities" paragraph in §3.5 (or wherever it lives) with the following displayed, numbered equations. Continue numbering from Eq. (5). In LaTeX use `\begin{equation}` with labels; in markdown use `$$...$$` with tags.

**C1. Note on Eq. (1).** Add after Eq. (1): "Here |y| counts completion tokens only; the same completion-only masking applies to ℓ in Eq. (2) and (3)."

**C2. TOST as two one-sided tests.** Insert immediately before Eq. (4):

> TOST tests two one-sided nulls, $H_{01}: \Delta \leq -\delta$ and $H_{02}: \Delta \geq \delta$, each at $\alpha = 0.05$; both are rejected jointly if and only if Eq. (4) holds. Eq. (4) uses an open interval; the per-seed inclusion rule is $|\Delta| \leq 5$ inclusive.

**C3. Primary metric and rates.**

$$\text{acc}_T(\theta) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\big[s(g_\theta(x_i)) \in \{\text{PASS}, \text{SURFACE\_FORM}\}\big], \tag{6}$$

where $g_\theta$ is greedy generation and $s$ is scorer v2. $\text{FB}_T$ and $\text{CE}_T$ are defined identically with FORMAT_BREAK and SEMANTIC_ERROR, so that $\text{acc}_T + \text{FB}_T + \text{CE}_T = 1$. The behavioral change is $\Delta_T = 100\,[\text{acc}_T(\theta_{180}) - \text{acc}_T(\theta_{\text{ref}})]$ in percentage points.

**C4. Per-item difference and cluster bootstrap.**

$$\Delta_i = \frac{1}{8} \sum_{s=1}^{8} c_i^{(s)} - c_i^{(0)}, \qquad \Delta = \frac{1}{N} \sum_{i=1}^{N} \Delta_i, \tag{7}$$

where $c_i^{(s)} \in \{0, 1\}$ is correctness of item $i$ under seed $s$ and $c_i^{(0)}$ under the reference. The item-level interval bootstraps over items (10,000 resamples), treating the eight per-item observations as a cluster.

**C5. Seed-level interval.**

$$\bar{\Delta} \pm t_{0.95, 7} \cdot \frac{s_\Delta}{\sqrt{8}}, \tag{8}$$

where $s_\Delta$ is the standard deviation of the eight per-seed $\Delta$ values. With $s_\Delta = 2.28$ the half-width is 1.53.

**C6. Margin decomposition and chosen share.**

$$m = \beta\, r_\theta(x, y_w) - \beta\, r_\theta(x, y_l), \qquad \rho_{\text{ch}} = \frac{|\beta\, r_\theta(x, y_w)|}{|m|}. \tag{9}$$

The claim in §4.3 that chosen contributes at most 4% refers to $\rho_{\text{ch}}$.

**C7. Aggregate axis margin and backend ratio.**

$$M_a = \sum_{i \in a} m_i, \qquad \kappa_a = \frac{\bar{m}_a^{(\text{nf4})}}{\bar{m}_a^{(\text{fp16})}}, \tag{10}$$

where $a \in \{\text{style}, \text{tool\_calling}\}$. §4.5 reports $\kappa_{\text{style}} \approx 3.3$ and $\kappa_{\text{tool}} \approx 1.2$.

**C8. Smoothed trigger margin.**

$$\tilde{m}_k = \frac{1}{20} \sum_{j=k-19}^{k} m_j, \tag{11}$$

over the 20 most recent training pairs (about 2.5 optimizer steps at batch size 8). Checkpoints fire when $\tilde{m}_k$ first exceeds 1, 3, 5.

**C9. Gold change.** Insert immediately before Eq. (5):

$$\Delta_{\text{gold}}^{(i)} = \ell_{\theta_{180}}(y_i^{\text{gold}} \mid x_i) - \ell_{\theta_{\text{ref}}}(y_i^{\text{gold}} \mid x_i), \tag{12}$$

with $\Delta_A^{(i)}$ and $\Delta_B^{(i)}$ defined identically for controls A and B.

**C10. Token-level KL.**

$$\text{KL}_t = \sum_{v} \pi_\theta(v \mid x, y_{<t}) \log \frac{\pi_\theta(v \mid x, y_{<t})}{\pi_{\text{ref}}(v \mid x, y_{<t})}, \tag{13}$$

averaged over gold positions $t$ under teacher forcing.

**Renumber.** Since C9 inserts an equation before the old Eq. (5), renumber so equations are sequential by appearance. Update every "Eq. (N)" cross-reference accordingly. Report the final mapping.

## Part D: Verify and compile

1. Grep for: "behavioral gradient", "gradient of degradation", "textbook squeezing", "unprecedented", "stated boundaries", "axis-dependently", "CHES failure mode", "was killed", "Json", "step20", "NF4". All zero.
2. Count equations: should be 13 displayed, numbered sequentially.
3. Confirm acc + FB + CE = 1 holds for each row of the router trajectory table (compute from the table values).
4. Compile with tectonic. 0 errors, 0 undefined.
5. `pdffonts` — no Type 3.
6. Rebuild `arxiv_submission.zip`. Commit and push.
7. Report: page count, equation numbering map, any item you could not apply.

## Do NOT change

- Any number or table value
- Hedges: "consistent with", "directionally consistent", "suggests", "may", "not established", "untested"
- The 3+2 falsification count
- Section structure (other than the §4.1 paragraph split)
