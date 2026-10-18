---
name: paper-writing
description: "Late-stage prose compression for research papers, encoding the SNL-UCSB lab's 7 compression operations. MANDATORY TRIGGERS: Use this skill whenever the user asks to compress, tighten, shorten, condense, or trim a section, paragraph, abstract, or paper; to cut to a page/word limit; or to remove wordiness, filler, throat-clearing, or generic adjectives from a draft. This is a compression-only skill — it does not handle brainstorming, drafting, figure synthesis, or section checklists."
---

# Paper Writing Skill — Compression Module

This is a trimmed, compression-only extract of the [SNL-UCSB paper-writing-skill](https://github.com/SNL-UCSB/paper-writing-skill). It encodes the lab's late-stage compression methodology and nothing else. For the full writing pipeline (brainstorming, drafting, figure synthesis, section checklists), install the complete skill from the upstream repository.

The compression methodology is derived from forensic analysis of 6 papers (8 submissions), 7,600+ Overleaf edits, and 5 peer review processes. It works out of the box — the rules are calibrated defaults.

## When to use

Trigger on any request to compress, tighten, shorten, condense, or trim text; to fit a page or word limit; or to strip wordiness, filler, or generic adjectives. Compression is a late-stage operation: the first draft should be comprehensive, and compression comes after the argument is settled.

## The compression workflow

Read `author_profile/compression_patterns.md` for the 7 specific operations, with before/after examples, quantitative benchmarks, and the principle each encodes. Apply the operations in this order:

1. **Sentence shortening** — remove subordinate clauses, qualifiers, and throat-clearing.
2. **Paragraph merging** — multiple examples of the same point → one best example.
3. **Generic adjective removal** — "significant" → a specific number, a named mechanism, or delete.
4. **Tutorial deletion** — remove explanations the venue audience already knows.
5. **Claim-first conversion** — rewrite buried paragraphs and headings so the claim leads.
6. **Takeaway insertion** — replace detailed result description with a compact synthesis paragraph after experiment clusters.
7. **Figure/table promotion** — move dense numerical comparisons from prose into visuals; let prose interpret patterns.

### Reporting

- Show before/after with **character counts** for each compressed unit.
- **Target: 30–50% reduction** from the first draft (see the Compression Intensity Guide in `author_profile/compression_patterns.md`).
- **Over 50% signals a framing problem, not a wordiness problem** — flag it rather than forcing the cut.

### Do not pad

Do not pad to fill a page limit. If the paper is under the limit after compression, that is fine — a short paper with appropriate content beats a padded one that reaches the limit. Padding introduces filler that weakens the argument.

## Quick command

**"Compress / tighten this section"**
1. Read `author_profile/compression_patterns.md`.
2. Apply the 7 operations in order.
3. Show before/after with character counts.
4. Normal compression is 30–50%; over 50% signals a framing problem, not a wordiness problem.

## Files in this skill

| File | Purpose |
|---|---|
| `author_profile/compression_patterns.md` | The 7 compression operations with before/after examples and quantitative benchmarks. |
