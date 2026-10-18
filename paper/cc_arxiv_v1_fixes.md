# arXiv v1 Fixes

## Context

A reviewer identified factual errors and writing issues. These are the fixes required before arXiv v1 submission. Items requiring new experiments are deferred to v2. Read `paper/CLAUDE.md` first.

Apply every change to both `paper/full_draft.md` and `paper/latex/main.tex` (or `appendix.tex` where noted). The .tex is a compressed rendering; locate the corresponding passage.

## Part A: Factual errors (4)

**A1. §5.1 unit error.** Find "The style axis contributes 65% more aggregate suppression than the correctness axis because it has 2.2× more pairs." The 65% is the per-pair ratio (8.119/4.923), not aggregate. Replace with:

> The style axis contributes more aggregate suppression than the correctness axis under both backends (3.65× under nf4, 1.36× under fp16).

**A2. Appendix C contradiction.** In Appendix C.1 (reconciliation table, row 3) and C.3 (chronology, third falsification), the text says the held-out prediction was "falsified at every checkpoint on every seed" and "tool pairs are individually more potent." Update both to match §4.3:

> Falsified under both backends: the prediction (style > tool at every checkpoint) holds at 0/4 checkpoints under fp16 and 2/4 under nf4. Final-checkpoint per-pair ordering is backend-dependent.

**A3. Appendix B tier mismatch.** (a) Find "The 69 items that flip from FAIL to PASS" and change to "The 69 items that flip from strict-FAIL to SURFACE_FORM (semantically correct)". (b) Appendix B lists four tiers (FORMAT_BREAK, PASS, SURFACE_FORM, SEMANTIC_ERROR) while §3.2 says "three-tier". Fix §3.2 to say "four-tier classification: FORMAT_BREAK (unparseable), PASS, SURFACE_FORM, and SEMANTIC_ERROR" so the two agree.

**A4. Abstract rounding.** Change "Δ = −0.5" to "Δ = −0.55" and "Δ = −3.5" to "Δ = −3.52" in the abstract to match Table 1. Also check §1 for the same rounded values and fix.

## Part B: Writing fixes (cheap, no new data)

**B1. Units statement.** In §3.1 immediately after Eq. (2), add:

> Because Eq. (1) is per-token and β = 0.5, a reported margin of 7 corresponds to a per-token log-ratio difference of 14 nats: under the policy, each rejected token is on average e^14 ≈ 1.2 × 10^6 times less likely relative to the reference than each chosen token. This is near-total suppression of the rejected text. Displacement and IPO analyses typically reason about sequence-level log-ratios, so our margins are not directly comparable in scale.

**B2. Neutral backend labels.** Replace "faithful-precision" everywhere (§3.5, §4.5, §5.1, §5.3, abstract if present) with "dequantized-base". Replace "re-measurement" with "dequantized-base measurement" where it stands alone. The adapters were co-adapted to nf4; neither backend is ground truth. In §3.5 backend-map paragraph, replace the sentence "fp16 is the faithful-precision audit backend" with:

> fp16 is the dequantized-base measurement: the same adapters loaded onto an unquantized base. Because the adapters were trained against the nf4 base, neither backend is privileged; nf4 reports what the optimizer experienced, fp16 reports what the adapter weights encode when the base is not quantized.

**B3. Format-alignment as hypothesis.** In §1 contributions and in the Conclusion, where format-alignment is mentioned, ensure it is labeled as a hypothesis. §1 contribution 1: change "driven by rejected-collapse dynamics and style-axis aggregate pressure" to "driven by rejected-collapse dynamics; the format-alignment hypothesis for which tasks are affected is consistent with the data but not directly measured". Conclusion: change "with behavioral outcomes tracking format-alignment to the preference data" to "with a format-alignment hypothesis consistent with but not established by the three-task gradient".

**B4. Memorization wording.** §4 opening: "ruling out train-set memorization as the source of margin growth" → "indicating that margin growth is not driven by train-set memorization".

**B5. Undefined internal references.** Replace or define on first use:
- "control-C" → "a third control (rejected-register text)" on first use, then "control C"
- "the two preliminary probe seeds" → "two seeds from a preliminary probe run (seeds 0 and 1, nf4)"
- "baseline_v2" → "the step-0 SFT baseline" on first use; may use "baseline" afterward
- "cc's n = 12 probe" or similar → delete or replace with "a preliminary 12-item probe"
- "the matrix" when referring to the 8-seed run → "the 8-seed confirmatory run" on first use in each section

**B6. Json McNemar qualification.** §4.1, find "0 gains and 45 losses (McNemar's test with continuity correction, p = 5.4 × 10⁻¹¹)". After it add:

> With only 2 of 160 items incorrect at baseline, zero gains is structurally near-guaranteed, so the test's significance mainly reflects the number of losses rather than directional asymmetry.

**B7. Inclusive boundary.** §4.1 where seed4 at −5.00 is discussed: add "(the inclusive bound |Δ| ≤ 5 was the pre-registered criterion)" if Note 2 specifies ≤. If Note 2 says < or is ambiguous, write "(we apply |Δ| ≤ 5 inclusively; Note 2 did not specify strict versus inclusive)". Check Note 2 before choosing.

**B8. Warmup discussion.** §4.2, after the sentence about step20 showing −8.6 pts: add

> Training uses a constant learning rate with no warmup, so some of this early loss may reflect optimization transient rather than margin-driven damage; we did not run a warmup ablation.

**B9. Ratio language.** §4.4 and §4.5: "~18× compression" and "8.6× worse" involve near-zero denominators (extraction nf4 Δgold = −0.008). Change "~18×" to "from −0.15 to −0.008" and "8.6× worse" to "−30 vs −3.5 points". Keep "~5×" for json (denominator −0.22 is not near-zero).

**B10. Additional citations.** Add to references.bib and cite:

```bibtex
@inproceedings{meng2024simpo,
  title={{SimPO}: Simple Preference Optimization with a Reference-Free Reward},
  author={Meng, Yu and Xia, Mengzhou and Chen, Danqi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  note={arXiv:2405.14734}
}

@article{pang2024iterative,
  title={Iterative Reasoning Preference Optimization},
  author={Pang, Richard Yuanzhe and Yuan, Weizhe and Cho, Kyunghyun and He, He and Sukhbaatar, Sainbayar and Weston, Jason},
  journal={arXiv preprint arXiv:2404.19733},
  year={2024}
}
```

§3.1 after Eq. (1): add "Length normalization follows the per-token formulation discussed by Meng et al. (2024)." After the RPO sentence: add "; a related DPO+NLL objective appears in Pang et al. (2024)".

**B11. Reproducibility statement.** Add a short paragraph at the end of §3 (before §4) or as a footnote on the title:

> Code, the 647 preference pairs, the three 160-item evaluation suites, scorer v2, and all 40 checkpoint adapters will be released at [URL to be added]. The base model is Qwen3.5-4B [exact HF checkpoint ID]; nf4 quantization uses bitsandbytes [version], and fp16 measurements use PyTorch MPS on Apple Silicon with torch.float16 (no bf16 fallback).

Fill in bracketed items from `kaggle_result/` configs and `requirements.txt` if available; otherwise leave the brackets for the user to fill.

**B12. Pre-registration note.** §3.4 where Notes 1 and 2 are cited: add "The pre-registration was informed by pilot runs on seeds 0–2 and frozen before the 8-seed confirmatory run; commit hashes are [to be added] and the notes will be released with the code."

**B13. Appendix em-dash sweep.** Apply the same readability rules (no em-dashes, no nested parentheticals, sentences under 40 words) to all appendices. Grep for `—` afterward; target zero in appendix.tex.

## Part C: Compile and verify

1. Grep both files for: "65% more", "faithful-precision", "every checkpoint on every seed", "FAIL to PASS", "three-tier", "18×", "8.6×", "rules out", "ruling out". All should return zero matches.
2. Grep references.bib for "meng2024simpo" and "pang2024iterative" — both present. Reference count should be 16.
3. Compile: `cd paper/latex && tectonic main.tex`. 0 errors, 0 undefined citations.
4. `pdffonts main.pdf` — no Type 3.
5. Rebuild `arxiv_submission.zip` with current sources.
6. Report: page count, reference count, grep results, and the filled-in values for B11 brackets (or which remain unfilled).

## Do NOT change

- Any number not listed above
- Any hedge word not listed above
- The equations
- The 3+2 falsification count
- Section structure
