# arXiv v1 Round 2 Fixes

## Context

Reviewer round 2 identified items that must be fixed before posting. Two CC checks produced new facts (seed provenance, json loss structure) that feed into these fixes. Read `paper/CLAUDE.md` first.

Apply every change to both `paper/full_draft.md` and `paper/latex/main.tex` or `appendix.tex`. The .tex is a compressed rendering; locate the corresponding passage.

Two one-line fixes carried over from the previous round are included (Part G).

## Part A: Title

Replace the title everywhere (markdown header, `\title{}`, any running header):

> The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention

Rationale: "Not Displacement" claimed a refutation; the paper only shows displacement is suppressed by the retention term, with the no-retention counterfactual untested.

## Part B: Accounting vs attribution

The margin decomposition (chosen flat, rejected collapses) is an accounting identity for the margin. It does not show that the parameter drift damaging router comes from the rejected-term gradient: with ℓ_θ(y_w) ≈ ℓ_ref(y_w), the SFT push-up and DPO push-down on chosen balance, but both gradients still move the weights. Sentences that say damage is "driven entirely by the rejected-suppression channel" convert the identity into a causal claim. Keep the decomposition claim; reword the damage claim as unattributed.

Find and replace in these locations:

**B1. §1 contribution 1.** "driven by rejected-collapse dynamics" → "with margin growth accounted for entirely by rejected-completion suppression; the source of the behavioral damage is not attributed pending a no-retention control".

**B2. §4.3.** "Even with chosen explicitly anchored, the behavioral alignment tax remains: router loses 30 points and json shows one-directional loss. This damage is driven entirely by the rejected-suppression channel, which the retention term does not protect." → "Even with chosen explicitly anchored, the behavioral alignment tax remains: router loses 30 points and json loses 3.5. The margin decomposition shows where the margin comes from, not where the damage comes from; both the DPO gradient on chosen and the gradient on rejected move the weights, and attributing the damage to one channel requires a no-retention control (§5.6)."

**B3. §5.1.** "the retention term anchors chosen-completion likelihood and channels DPO's pressure entirely into rejected suppression" → "the retention term anchors chosen-completion likelihood, and the full margin is accounted for by rejected suppression".

**B4. §5.3.** Find "driven entirely by the rejected-suppression channel that the retention term does not protect" → "with the full margin accounted for by rejected suppression; which gradient channel produces the behavioral damage is untested".

**B5. Conclusion.** Find "DPO's gradient pressure channels entirely into rejected-completion suppression" → "the full margin growth is accounted for by rejected-completion suppression".

**B6. Grep for "channels" and "channeled" in all files.** Any remaining instance that attributes damage (not margin) to a channel must be reworded.

## Part C: Json paragraph (§4.1)

**C1. Delete the self-contradiction.** The paragraph now contains both "zero gains is structurally near-guaranteed... reflects the number of losses rather than directional asymmetry" and "Every json flip is a regression, never offset by a compensating gain. The degradation is statistically genuine, one-directional, and consistent." Delete the second set (the "Every json flip..." sentence and "one-directional" wording). Keep the structural caveat.

**C2. Remove "one-directional" from the abstract and §1.** In the abstract, "(json, Δ = −3.52, one-directional loss, TOST boundary FAIL)" → "(json, Δ = −3.52, TOST boundary FAIL)". In §1, same change.

**C3. Fix "ceiling-driven".** Find "The TOST failure is partly an artifact of testing near the ceiling (baseline 98.75 leaves minimal room), but the underlying drop is real." Replace with:

> The ceiling (baseline 98.75) caps gains, which reduces item variance; the item-level CI is wide because losses cluster on specific items. The 45 losses fall on 17 distinct items; five items account for 28 of them (one item lost on 7 of 8 seeds, two on 6 of 8). Because losses cluster within items, they are not independent across seeds, which further limits the McNemar test. Of the 45 losses, 35 are wrong-ticker errors in the free-text argument field (the tool name is correct), 7 are wrong-tool substitutions (6 on seed 0), and 3 are format breaks.

**C4. §4.4 and §5.4 redundant-constraint hypothesis.** After "structured output formats with redundant constraints (typed fields, function signatures) may maintain correct greedy-decode behavior despite large probability-mass redistribution", add: "The json loss pattern is consistent with this: 35 of 45 losses land on the one free-text argument (ticker), not on the typed tool-selection field." Apply in §4.4 only; in §5.4 add a cross-reference "(see §4.1 for the loss breakdown)".

## Part D: Seeds 0 and 1 disclosure

CC verified: confirmatory seed 0 is bitwise identical to pilot seed 0 at every checkpoint. Confirmatory seed 1 is bitwise identical to pilot seed 1 through margin5; only the final adapter differs (the pilot early-stopped at step 156, the confirmatory ran to 180). Seeds 2–7 started after the pre-registration freeze (commit 62ab7c3). Seed 0's confirmatory run started 26 minutes after the freeze if log timestamps are UTC.

**D1. §3.4.** Find "The pre-registration was informed by pilot runs on seeds 0 and 1 and frozen before the 8-seed confirmatory run" and the nearby "inspected only after all 8 seeds completed". Replace both with:

> The pre-registration was informed by pilot runs on seeds 0 and 1 and frozen (commit 62ab7c3) before seeds 2–7 were run. Training is bitwise deterministic, so the confirmatory seeds 0 and 1 reproduce the pilot adapters: seed 0 at every checkpoint, seed 1 through margin5 (the pilot early-stopped at step 156; the confirmatory run continued to step 180). The confirmatory set therefore contains six seeds (2–7) whose outcomes were not visible at freeze time. All 8 seeds are reported; results restricted to seeds 2–7 are in Appendix A.

**D2. Appendix A.** Add a one-line note after the per-seed TOST table: "Seeds 0 and 1 are deterministic re-runs of the pilot (§3.4). Restricting to seeds 2–7: extraction mean Δ = [compute], json mean Δ = [compute], router mean Δ = [compute]." Compute these three means from the existing per-seed Appendix A values and fill them in. State whether each verdict (extraction PASS, json FAIL, router FAIL) is unchanged on the 6-seed subset.

**D3. Appendix C.2.** Add under pre-registration deviations: "Seed overlap. Seeds 0 and 1 of the confirmatory set are bitwise re-runs of the pilot seeds (seed 0 fully; seed 1 through margin5). Six of eight confirmatory seeds were unseen at freeze time."

## Part E: Rendering

**E1. §3.2 task table.** Give it a `\begin{table}` with `\caption{Evaluation tasks: output format, example, and preference-data coverage.}` and `\label{tab:tasks}`. Put the Example column in `\texttt{\small ...}` and allow it to wrap with `\\` at a sensible point, or shorten the JSON example to `\{"name":"get\_filing", "args":\{...\}\}`. Reference it in §3.2 text as "Table~\ref{tab:tasks}".

**E2. Figure 1 annotation.** In `make_figures.py`, the damage-window annotation contains a literal `1--3`. Change to `1–3` (en-dash character) or `1$-$3`. Regenerate Figure 1.

**E3. Scorer artifact box.** The tcolorbox in §3.2 is titled "Scorer artifact (§3.2)" inside §3.2. Change the title to "The scorer artifact: strict versus canonicalized scoring".

**E4. Figure 2 y-axis.** In `make_figures.py`, change the y-axis label to include "per-token": e.g. "Reward (per-token policy logp − SFT logp)". Regenerate.

## Part F: Definitions block

Add a short subsection or paragraph in §3, after Eq. (5) in §3.5 (or as a new §3.6 "Reported quantities"), defining:

> **Reported quantities.** Semantic accuracy for a task is the mean over its 160 items of an indicator that the output is PASS or SURFACE_FORM under scorer v2. FORMAT_BREAK rate and content-error (SEMANTIC_ERROR) rate are the corresponding means of their indicators. The smoothed margin used for checkpoint triggers is the mean of the per-batch margin $m$ (Eq. 2) over the most recent 20 optimizer steps. Aggregate axis margin is $\sum_{i \in \text{axis}} m_i$ over the training pairs in that axis; mean per-pair margin is the same sum divided by the pair count. The seed-level confidence interval is a $t$ interval over the 8 per-seed accuracy differences ($df = 7$). The item-level interval is a bootstrap over items with seeds pooled (1000 resamples), which treats the 8 per-item observations as a cluster. Token-level KL is $\sum_v \pi_\theta(v) [\log \pi_\theta(v) - \log \pi_{\text{ref}}(v)]$ averaged over gold-token positions.

Verify the smoothing window definition (20 steps vs 20 pairs) against `train_pref.py` before writing; adjust the sentence to match the code.

## Part G: Consistency and wording

**G1. Power number.** §3.4, replace "n = 160 items with 8 seeds gives reasonable power to detect 5-point differences" with: "a post hoc calculation using the observed per-seed standard deviation (about 2.2 points on extraction and json) gives power near 1 for TOST at ±5 when the true Δ is 0". Compute the actual per-seed SD from Appendix A for extraction and json at final, and use the real values.

**G2. Rounding consistency.** Table 4 (fp16 Δgold, "Behavioral Δ" column) and §4.4 text use "−0.5 pts" and "−3.5 pts". Change to "−0.55 pts" and "−3.52 pts" to match Table 1 and the abstract.

**G3. Task name casing.** Standardize throughout: lowercase "json", "router", "extraction" in running text; capitalize only at sentence start or in table headers. Grep for "Json" and "JSON" (the latter only when referring to the data format, not the task, is acceptable).

**G4. §4 intro.** "Table 3 reports the reward decomposition over all 549 train pairs" — Table 3 is per-axis. Change to "the pooled value derived from Table 3".

**G5. §2 "precedents".** Find "Minerva ... precedents answer-normalization" or similar verb use. Change to "Minerva (Lewkowycz et al., 2022) provides precedent for answer normalization".

**G6. §3.1 last sentence.** Find "with the assignment pre-registered and results inspected only after all 8 seeds completed" or the grammatically broken variant. This sentence is superseded by D1; delete it.

**G7. §1 IPO caveat.** After "echoing the unbounded growth IPO was designed to prevent", add "(our margins are per-token; see §3.1 for the conversion)".

**G8. Scorer artifact mentions.** Currently in §1, §2, §3.2 box, §4.2, §5.2, Appendix B. Keep §3.2 box and Appendix B. In §1, §2, §4.2, §5.2: replace the full "0.550 → 0.981" statement with a short pointer: "(see the scorer artifact in §3.2)". Keep any surrounding claim; only remove the repeated numbers.

**G9. §5.6.** Add: "The no-retention counterfactual (λ = 0 in Eq. 3) is untested; whether chosen declines and whether behavioral damage changes without the retention term are open questions."

**G10. Carried over.** Abstract: "The gradient tracks format-alignment to the preference data more than margin magnitude." → "The gradient is consistent with a format-alignment hypothesis (not directly measured) rather than margin magnitude." Table 11 header: "Margin (fp16, audit)" → "Margin (fp16, dequantized)".

## Part H: Abstract rewrite

Replace the full abstract with (keep all numbers as they are now in the paper; adjust only structure):

> **Setting.** We fine-tune Qwen3.5-4B with sparse LoRA and length-normalized DPO plus an SFT retention term (the RPO objective) on structured financial-filing tasks, and track downstream capability as the implicit reward margin grows to ~7 β-scaled units (nf4 training backend; ~3 under fp16 dequantized-base measurement).
>
> **Question.** Does margin growth degrade capability uniformly, as displacement theory and IPO's design motivation predict?
>
> **Findings.** In a pre-registered 8-seed study with artifact-aware scoring and TOST equivalence testing, we find the alignment tax is graded and task-selective: extraction is preserved (Δ = −0.55, equivalent at ±5), json loses little (Δ = −3.52, equivalence fails on a clustered item-level interval), and router degrades severely (Δ = −30). Router's loss concentrates in a format-break window at margin 1–3. The margin is accounted for entirely by rejected-completion suppression with chosen held flat by the retention term; the source of the behavioral damage is not attributed. Gold-logp deflation is task-selective (extraction −0.15, json −1.04, router −1.47 nats/token under fp16), with gold selectively protected relative to incorrect controls on extraction (8/8 seeds). Implicit reward margins under QLoRA differ up to 3.4× between the nf4 training backend and fp16 dequantized measurement, axis-dependently.
>
> **Scope.** Length-normalized rewards with SFT retention, sparse LoRA on 8/32 layers, one 4B model, easily-separable synthetic preferences, three structured tasks. A no-retention control and a measured format-alignment metric are left to future work.

Format the bold labels as inline (not separate paragraphs) if the NeurIPS style abstract environment does not support paragraph breaks.

## Part I: Placeholders

**I1. bitsandbytes version.** Find the exact version: check `requirements.txt`, any `pip freeze` output in `kaggle_result/`, Kaggle notebook logs, or `train_config.json`. If only a range (`>=0.43.0`) is recorded, check the Kaggle notebook's first-cell output for the installed version. If none can be found, write "bitsandbytes 0.43+" and flag it.

**I2. Repository URL.** The user has a GitHub repo `delta-filing`. Run `git remote -v` to get the URL. Then run `git status` and `git log --oneline -5` and report: (a) the remote URL, (b) whether `paper/` is tracked and pushed, (c) what is uncommitted. Fill the URL into the reproducibility paragraph as `https://github.com/<user>/delta-filing`. Do not push anything; just report the state so the user can decide what to publish.

## Part J: Compile and verify

1. Grep all files for: "Not Displacement", "driven entirely", "channels entirely", "one-directional", "ceiling-driven", "reasonable power", "precedents answer", "[URL to be added]", "[version]". All should return zero (except the last two if the user has not yet supplied values).
2. Grep for "Selective Deflation" — should appear only if the user wants it in a subtitle; otherwise zero.
3. Confirm D2 numbers are filled and verdicts stated.
4. Confirm F definitions match code (smoothing window).
5. Compile: `cd paper/latex && tectonic main.tex`. 0 errors, 0 undefined.
6. `pdffonts` — no Type 3. Regenerated figures must also be Type 3-free.
7. Rebuild `arxiv_submission.zip`.
8. Report: page count, grep results, the D2 six-seed means and verdicts, the G1 computed SDs and power statement, and any item you could not complete.

## Do NOT change

- Any number not listed above
- Equations (1)–(5)
- The 3+2 falsification count
- Hedges not listed above
