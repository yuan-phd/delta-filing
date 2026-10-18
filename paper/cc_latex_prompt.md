# Task: LaTeX Conversion + Compression + Figure Generation

You are working on a workshop paper submission. All content is finalized in `paper/full_draft.md`. Your job is formatting, compression, and figure generation — NOT content creation or claim modification.

## Critical rules (read paper/CLAUDE.md first)

Before doing anything, read `paper/CLAUDE.md`. It contains mandatory terminology rules, hedge calibrations, and formatting constraints. Every rule in CLAUDE.md overrides your defaults. Specifically:

- **Do NOT change any scientific claim, number, hedge word, or scope qualifier.** "Consistent with" must stay "consistent with" — never upgrade to "demonstrates." "Directionally consistent" must not become "correlated." "At least partly a consequence" must not become "caused by."
- **Do NOT remove caveats, scope conditions, or limitation statements** even if they seem redundant during compression.
- **Do NOT change the falsification count** (three falsifications and two refinements).
- **Do NOT rephrase the ±5 justification** — the wording about "researcher judgment" is deliberate.
- **Preserve all table numbers and cross-references** (Tables 1–7 in main text, Appendix G/H references).

## Step 1: Read the compression skill

Read `paper/skills/paper-writing/SKILL.md` and `paper/skills/paper-writing/author_profile/compression_patterns.md`. You will use these compression operations in Step 3.

## Step 2: Convert to LaTeX

Convert `paper/full_draft.md` to LaTeX using the NeurIPS 2026 workshop template (neurips_2026.sty or equivalent). If you don't have the template, use the standard NeurIPS style file.

Structure:
- Title: "Selective Deflation, Not Displacement: Characterizing the Alignment Tax Under Length-Normalized DPO"
- Author: [I will fill in]
- Abstract → §1 → §2 → §3 → §4 → §5 → Conclusion → References → Appendix A–H
- Tables 1–7 in main text, formatted as LaTeX tables
- Three figure placeholders (Figure 1, 2, 3) — I will insert generated figures in Step 4
- References in BibTeX — create a `references.bib` with all cited papers (use the verified citation data from the draft: venues, years, arXiv IDs are all confirmed)

Output: `paper/latex/main.tex` + `paper/latex/references.bib` + any style files

## Step 3: Compress main text to ≤4 pages

The main text (Abstract through Conclusion, excluding References and Appendix) must fit within 4 pages in the LaTeX template. Current draft is ~7 pages of content, so significant compression is needed (~40–45% reduction).

Apply the 7 compression operations from the skill in order. Priority guidance:

**What to cut first (low information density):**
- Redundant cross-references between sections ("as discussed in §X" when the reader just read §X)
- Restatements of findings already shown in tables (if a table says it, one sentence summarizing is enough)
- Transitional sentences that don't carry claims
- Methodological detail that can move to appendix (scorer rules → already in Appendix B; hygiene details → already in Appendix D; hash verification → already in Appendix E)

**What to compress (rephrase more tightly):**
- Multi-sentence explanations that can be one sentence
- Parenthetical asides that can be footnotes or cut

**What to NEVER cut:**
- The TOST table (Table 1) and its verdicts
- The three-task gold-logp finding (Table 6) 
- The GOLD_PROTECTED result (Table 7, 8/8 seeds)
- The held-out falsification result (Table 5)
- The reward decomposition finding (chosen flat, rejected collapses)
- The FORMAT_BREAK characterization (UPPER_SNAKE_CASE pseudo-labels)
- The scorer artifact (0.550 → 0.969)
- All scope conditions and caveats
- The retention term disclosure
- The falsification chronology (3+2) in §5.2

**Section-specific guidance:**
- §3 Method: move scorer rules, hygiene details, and hash chain to appendix (they're already there). Keep: loss function, task table, TOST plan, checkpoint specification, probability-layer measurement overview.
- §4 Results: this is the core — compress prose around tables but keep all tables and key numbers. §4.2 (trajectory) can be tightened. §4.3 per-axis table (Table 4) could move to appendix if needed.
- §5 Discussion: §5.2 (chronology) can be shortened to 2–3 sentences pointing to Appendix C. §5.4 (format-alignment caveats) can be tightened. §5.6 (limitations) can be compressed.

Report before/after character counts for each section.

## Step 4: Generate 3 figures

Generate publication-quality matplotlib figures from the data files. Save as PDF in `paper/latex/figures/`.

**Figure 1: Per-seed margin-vs-capability trajectories**
- Data source: Appendix A table in full_draft.md (per-seed semantic accuracy at each checkpoint)
- X-axis: margin (use the mean margins from Table 2: 0.75, 1.09, 3.12, 5.13, 6.73)
- Y-axis: semantic accuracy (%)
- Three panels or one plot with three colors (extraction=green, json=blue, router=red)
- Per-seed lines (thin, alpha=0.3) + mean line (thick)
- Baseline as horizontal dashed line per task
- This is the paper's signature figure

**Figure 2: Reward decomposition — chosen vs rejected**
- Data source: Appendix F table (per-seed r_chosen, r_rejected at each checkpoint)
- X-axis: checkpoint (margin1, margin3, margin5, final)
- Y-axis: reward (policy logp − SFT logp)
- Two lines: chosen (near zero, flat) and rejected (collapsing)
- Per-seed thin lines + mean thick line
- Optional: small inset or annotation showing the per-axis split at final (style vs tool_calling)

**Figure 3: Gold-logp paired stat across 8 seeds**
- Data source: Appendix G.3 table (per-seed stat with 95% CI)
- Bar chart: 8 bars (one per seed), height = stat value
- Error bars: 95% CI
- Horizontal dashed line at X = 0.05 (threshold)
- All bars clearly above the threshold line
- Title or annotation: "GOLD_PROTECTED: 8/8 seeds"

Style for all figures: clean, minimal, publication-quality. Font size readable at column width. No chartjunk.

## Deliverables

1. `paper/latex/main.tex` — complete, compilable
2. `paper/latex/references.bib` — all citations
3. `paper/latex/figures/fig1_trajectories.pdf`
4. `paper/latex/figures/fig2_reward_decomposition.pdf`
5. `paper/latex/figures/fig3_gold_protected.pdf`
6. A compression report: before/after character counts per section, total page count after compression

## What NOT to do

- Do not modify `paper/full_draft.md` — it is the authoritative source, read-only
- Do not change any scientific claim, number, or hedge word
- Do not add content that isn't in full_draft.md
- Do not remove the retention term disclosure from any section where it appears
- Do not merge or reorder sections
