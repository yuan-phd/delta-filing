# Errata

Corrections to the paper *The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention* (arXiv:2611.06476). Each entry names the arXiv version it applies to. Corrections are applied in a future arXiv version; the tagged code and data states are not rewritten.

## arXiv v1 (tag `paper-v1`)

### 1. Section 3.2: router gold-label adjudication

**v1 text:**
> Router gold labels were independently checked by GPT-4o; items it disagreed with or flagged as ambiguous were replaced, and all 160 final router items carry agreed labels.

**What the record shows:** GPT-4o agreed with all 160 router gold labels and flagged none as ambiguous, so no items were replaced. Sources:
- `notes/2026-06-11_note3_chronology_and_disclosures.md`: "Router gold double-adjudicated (gpt-4o): 160/160 agree."
- `data_eval/track1_router.n160_preadj.bak`, the 160-item snapshot taken before adjudication, is byte-identical to the frozen suite `data_eval/track1_router.jsonl`.

**Corrected text:**
> Router gold labels were independently checked by GPT-4o, which agreed with all 160 labels and flagged none as ambiguous; no items were replaced.

**Effect on results:** none. The frozen suite, gold labels and all reported numbers are unchanged; only the description of the adjudication outcome is corrected.

**Status:** to be applied in a future arXiv version.
