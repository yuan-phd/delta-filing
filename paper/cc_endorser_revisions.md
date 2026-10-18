# Endorser Revisions: Equations + Readability

## Context

A paper endorser reviewed `paper/latex/main.pdf` and gave two pieces of feedback:
1. The English is hard to read: too many em-dash asides and long nested clauses.
2. The paper has zero equations. For an ML paper this reads as lacking rigor, and §2's literature discussion led the reader to expect a proposed method with a formula.

This is an empirical characterization paper, not a method paper. We do not propose a new algorithm. Both fixes below preserve every claim, number, and hedge word. Read `paper/CLAUDE.md` first for the mandatory terminology rules.

Files to edit: `paper/full_draft.md` (authoritative source) and `paper/latex/main.tex` (compiled). Apply every change to both. The .tex is a compressed rendering, so locate the corresponding passage rather than assuming line-by-line correspondence.

## Part 1: Clarify paper type in §1

Find the paragraph in §1 that begins "This paper presents a pre-registered 8-seed study..." Insert as its first sentence:

> This is an empirical characterization study: we do not propose a new training algorithm, but measure how an existing one (length-normalized DPO with an SFT regularization term) affects downstream capability as its implicit reward margin grows.

## Part 2: Add equations to §3

Add the following displayed equations in §3.1 and §3.4–3.5. Use `\begin{equation}...\end{equation}` with `\label{}` in LaTeX; in markdown use `$$...$$` with a label comment. Number them sequentially.

### §3.1, after "We fine-tune Qwen3.5-4B..." paragraph, before the architecture description

Replace the inline loss description ("The total loss is L = L\_pref + 0.5 × L\_sft, where...") with:

> Let $\pi_\theta$ denote the policy and $\pi_{\text{ref}}$ the SFT reference. For a prompt $x$ and completion $y$ of length $|y|$ tokens, the length-normalized log-probability is
>
> $$\ell_\theta(y \mid x) = \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}). \tag{1}$$
>
> The implicit reward is $r_\theta(x, y) = \beta\,[\ell_\theta(y \mid x) - \ell_{\text{ref}}(y \mid x)]$ with $\beta = 0.5$. Given a preference pair $(x, y_w, y_l)$ with chosen $y_w$ and rejected $y_l$, the DPO preference loss is
>
> $$\mathcal{L}_{\text{pref}} = -\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big), \tag{2}$$
>
> and the preference margin is $m = r_\theta(x, y_w) - r_\theta(x, y_l)$. The total training objective adds an assistant-only cross-entropy on the chosen completion:
>
> $$\mathcal{L} = \mathcal{L}_{\text{pref}} + \lambda\, \mathcal{L}_{\text{sft}}, \qquad \mathcal{L}_{\text{sft}} = -\ell_\theta(y_w \mid x), \qquad \lambda = 0.5. \tag{3}$$
>
> Equation (3) is the Regularized Preference Optimization (RPO) objective of Liu et al. (2024); we refer to the $\lambda \mathcal{L}_{\text{sft}}$ term as the SFT retention term throughout. Razin et al. (2025) identify this SFT regularization as a mitigation for likelihood displacement.

Then keep the existing sentence "The retention term anchors chosen-completion likelihood during preference optimization." and delete the now-redundant "This formulation corresponds to Regularized Preference Optimization (RPO; Liu et al., 2024)..." sentence (it is absorbed into the text above).

### §3.4, in the TOST description

After the sentence defining the primary endpoint ("The primary endpoint is per-task semantic accuracy at the final checkpoint..."), add:

> Let $\Delta = \text{acc}_{\text{final}} - \text{acc}_{\text{baseline}}$ in percentage points. TOST tests the composite null $H_0: |\Delta| \geq \delta$ against $H_1: |\Delta| < \delta$ with $\delta = 5$. Equivalence is declared when the 90% confidence interval for $\Delta$ lies entirely within $(-\delta, \delta)$:
>
> $$\text{Equivalent} \iff \text{CI}_{90\%}(\Delta) \subset (-5, 5). \tag{4}$$

### §3.5, in the probability-layer description

After "The primary statistic is the paired difference Δgold − Δmean(A,B)...", replace that prose with:

> For each item $i$, let $\Delta_{\text{gold}}^{(i)}$ denote the change in $\ell_\theta$ for the gold completion from SFT baseline to final checkpoint, and $\Delta_A^{(i)}, \Delta_B^{(i)}$ the same for controls A and B. The paired statistic is
>
> $$S = \frac{1}{N} \sum_{i=1}^{N} \left[ \Delta_{\text{gold}}^{(i)} - \tfrac{1}{2}\big(\Delta_A^{(i)} + \Delta_B^{(i)}\big) \right], \tag{5}$$
>
> with $N = 160$. A positive $S$ indicates gold deflates less than the controls on average. The pre-registered threshold is $S > 0.05$ nats/token with a 95% bootstrap confidence interval excluding zero.

### Elsewhere: reference the equations

- §4.3, where "margin = β × (r_chosen − r_rejected)" appears in prose or captions: replace with "margin $m$ (Eq. 2)".
- §4.4, where the paired GOLD_PROTECTED statistic is first mentioned: add "(Eq. 5)".
- Table 1 caption or §4.1 opening: add "(Eq. 4)" after "TOST equivalence test".

## Part 3: Readability pass — all sections

Apply these rules to every paragraph in Abstract, §1–§5, and Conclusion. Do not touch appendices.

**Rules:**
1. **Em-dashes:** Replace every em-dash aside with either (a) a separate sentence, (b) a parenthetical in commas, or (c) a colon if introducing an explanation. Target: zero em-dashes in main text. Exception: an em-dash before a numeric range is fine (e.g., "margin 1–3" uses en-dash, keep).
2. **Sentence length:** Any sentence over 40 words must be split. Any sentence with more than one subordinate clause must be split.
3. **Nested parentheticals:** A parenthetical inside a parenthetical is never allowed. Flatten by moving one to a separate sentence.
4. **Front-load the claim:** If a sentence starts with a long qualifier before reaching the verb, reorder so the main claim comes first and the qualifier follows.

**What must NOT change:**
- Any number, table value, or figure reference
- Any hedge word: "consistent with", "directionally consistent", "suggests", "may", "not established", "untested", "at least partly"
- Any scope qualifier: "under nf4", "under fp16", "in our regime", "at the final checkpoint"
- The falsification count (three falsified, two refined)
- Any cross-reference (§ numbers, table numbers)
- Terminology fixed in CLAUDE.md

**Worked example** (from §4.3):

Before:
> Under nf4 (the training backend — the margins the optimizer actually experienced), style dominates via both higher per-pair margin (8.12 vs 4.92) and pair count (aggregate: 3069 vs 842, 3.65×).

After:
> Under nf4, the training backend, style dominates on both per-pair margin (8.12 vs 4.92) and pair count. The aggregate margin is 3069 vs 842, a 3.65× ratio. These are the margins the optimizer actually experienced.

Before:
> Even with chosen explicitly anchored, the behavioral alignment tax remains — router −30 points, json one-directional loss — driven entirely by the rejected-suppression channel the retention term does not protect (implications for Cal-DPO and Rafailov et al. in §5.3).

After:
> Even with chosen explicitly anchored, the behavioral alignment tax remains: router loses 30 points and json shows one-directional loss. This damage is driven entirely by the rejected-suppression channel, which the retention term does not protect. Implications for Cal-DPO and Rafailov et al. are discussed in §5.3.

Go paragraph by paragraph. For each one, report the before/after only if the sentence structure changed materially (do not list trivial comma edits).

## Part 4: Verify and compile

1. Grep main.tex for `—` (em-dash, U+2014) in sections 1–5 and abstract. Report count. Target is zero outside appendices.
2. Confirm equations (1)–(5) render and are numbered sequentially.
3. Grep for "Eq. 2", "Eq. 4", "Eq. 5" to confirm cross-references exist.
4. Compile: `cd paper/latex && tectonic main.tex` (or pdflatex sequence if available). 0 errors, 0 undefined refs.
5. Report: page count (expect 20–21; equations add about half a page, readability edits are roughly neutral), em-dash count in main text, and the five equation numbers as rendered.

## Report format

- Part 1: confirm inserted, quote the sentence
- Part 2: list all five equations as rendered in the PDF, and all "Eq. N" cross-references added
- Part 3: per-section count of sentences restructured; show 3–5 representative before/after pairs
- Part 4: compile results
