# Compression & Build Report

**Source:** `paper/full_draft.md` (read-only, unmodified)
**Output:** `paper/latex/{main.tex, appendix.tex, neurips_2026.sty, references.bib}` + `figures/*.pdf`
**Build:** compiles cleanly with `pdflatex → bibtex → pdflatex ×2` (TinyTeX). 0 errors, 0 undefined references/citations. Total PDF = 18 pages (main text + References + Appendices A–H).

## Page count (main text, Abstract → Conclusion)

| Layout | Main-text pages | Notes |
|---|---|---|
| Single-column NeurIPS (delivered) | **~9.3** | References begin on p10; Appendices A–H follow |

The 4-page target in `cc_latex_prompt.md` (and the 6-page target in `CLAUDE.md`) are **not reachable in single-column NeurIPS** while preserving all "never-cut" content. See "Page-limit analysis" below.

## Per-section prose compression (characters, tables/figures excluded)

| Section | Before | After | Cut |
|---|---:|---:|---:|
| Abstract | 1384 | 1378 | 0.4% |
| §1 Introduction | 4312 | 4033 | 6.5% |
| §2 Related Work | 5357 | 3804 | 29.0% |
| §3.1 Setup | 1964 | 1533 | 21.9% |
| §3.2 Tasks & Evaluation Suite | 1894 | 1029 | 45.7% |
| §3.3 Suite Independence | 1462 | 765 | 47.7% |
| §3.4 Pre-Registered Analysis Plan | 2347 | 1733 | 26.2% |
| §3.5 Probability-Layer Measurement | 1617 | 1024 | 36.7% |
| §4 Results (opening) | 443 | 408 | 7.9% |
| §4.1 Graded Degradation | 2864 | 2345 | 18.1% |
| §4.2 Non-Monotonic Dynamics | 3043 | 2273 | 25.3% |
| §4.3 Mechanism | 2572 | 2026 | 21.2% |
| §4.4 Gold-Logp Dynamics | 4590 | 3225 | 29.7% |
| §5.1 Two-Level Account | 2509 | 1524 | 39.3% |
| §5.2 Falsification Chronology | 2277 | 1632 | 28.3% |
| §5.3 Engagement with Prior Work | 2070 | 1410 | 31.9% |
| §5.4 Format-Alignment Caveats | 1835 | 1413 | 23.0% |
| §5.5 Practical Implications | 760 | 735 | 3.3% |
| §5.6 Limitations | 2295 | 1613 | 29.7% |
| Conclusion | 1474 | 1447 | 1.8% |
| **TOTAL** | **47069** | **35350** | **24.9%** |

The overall 24.9% is below the skill's 30–50% norm **by design**: the "never-cut" list (Abstract, Conclusion, §4.1 verdicts, the 3+2 chronology, all scope conditions and the retention-term disclosures) covers a large fraction of the draft and was held nearly verbatim (Abstract 0.4%, Conclusion 1.8%, Intro 6.5%, Practical Implications 3.3%). The freely-compressible sections were cut 25–48%.

## Compression operations applied (skill order)

1. **Sentence shortening** — throat-clearing and subordinate clauses removed throughout (e.g., §2 per-citation paragraphs, §3.1/3.3).
2. **Paragraph merging** — §5.1 two-level account merged and de-duplicated against §4.3/§4.4 (39% cut) without dropping any hedge.
3. **Generic-adjective removal** — n/a (draft was already metric-driven).
4. **Tutorial deletion** — scorer-architecture rules, suite-hygiene floors, and hash-chain detail moved to Appendices B/D/E (already present there); main text keeps a one-line pointer each.
5. **Claim-first conversion** — subsection openings lead with the finding (e.g., §4.1 "the pre-registered falsifier fires").
6. **Takeaway compression** — §4.x prose tightened around tables; numbers stated once, not restated from tables.
7. **Figure/table promotion** — Tables 2, 4, 5 relocated to the appendix (see below); Figure 1 carries the trajectory visually.

## Tables & figures

**In main text:** Table 1 (TOST), Table 3 (reward decomposition), Table 6 (three-task gold-logp), Table 7 (GOLD_PROTECTED) + Figures 1–3.
**Relocated to appendix (per your Option-2 instruction):** Table 4 (per-axis reward → App F), Table 5 (held-out margin → App F), Table 2 (router trajectory → App A). Each is summarized in one sentence in the main text with an appendix pointer; all claims (incl. the falsified style>tool prediction and the FORMAT_BREAK characterization) remain in the main-text prose.

**Figures** (generated from the draft's data; vector PDF):
- `fig1_trajectories.pdf` — per-seed margin-vs-capability, 3 tasks (App A data).
- `fig2_reward_decomposition.pdf` — chosen flat vs rejected collapse, per-seed + mean, inset per-axis (App F).
- `fig3_gold_protected.pdf` — 8-seed paired stat with 95% CI vs X=0.05 (App G.3).

## Content integrity (verified preserved)

- No scientific claim, number, hedge word, or scope qualifier changed. "Consistent with", "directionally consistent", "at least partly a consequence", "co-occurrence without claiming causation" all retained.
- Falsification count unchanged: **three falsifications and two refinements**.
- ±5 justification wording ("researcher judgment, not derived from an external standard") unchanged.
- Retention-term disclosure retained in Abstract, §1, §2, §3.1, §4.3, §5.1, §5.3, §5.5.
- Per-axis framing sentence ("aggregate suppression pressure … not per-pair potency") unchanged.
- All TOST verdicts, the 0.550→0.969 scorer artifact (with tcolorbox callout), the 8/8 GOLD_PROTECTED result, the reward decomposition, and the FORMAT_BREAK UPPER_SNAKE_CASE characterization preserved.

## Page-limit analysis (why single-column cannot reach 4–6 pages)

Single-column NeurIPS packs ~5,000 characters of prose per page. The main text carries ~35,350 prose chars (~7 pages) + 4 tables + 3 figures (~2.3 pages) ≈ 9.3 pages. Reaching 6 pages would require cutting prose to ~17k chars — a further ~50% cut (≈64% from the original), which would force removal of protected caveats. Reaching 4 pages is further out of range. The page count is dominated by **content volume**, not wordiness; the compression skill flags >50% as a framing problem, not a wordiness problem.

**Levers to reach ≤6 pages (each requires a constraint to be relaxed):**
1. **Two-column layout** — roughly halves the page count to ~5–6pp with *no* content cut; deviates from the standard single-column NeurIPS style.
2. **Trim protected caveats / move more of §3 (Method) to appendix** — can reach ~6–7pp but trims items on the "never-cut" list.

Delivered state honors single-column + all protected content; the remaining gap is a presentation decision.
