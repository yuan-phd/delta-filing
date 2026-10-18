# Pre-arXiv Citation Updates

## Context

You are editing a finished workshop paper on DPO alignment tax under QLoRA. The paper is complete and reviewed; these are final citation updates before arXiv submission. Read `paper/CLAUDE.md` first — it contains mandatory terminology and hedge rules.

Files:
- `paper/full_draft.md` — authoritative markdown source
- `paper/latex/main.tex` — compiled LaTeX main body
- `paper/latex/appendix.tex` — compiled LaTeX appendix
- `paper/latex/references.bib` — bibliography

The .tex files are a compressed rendering of full_draft.md (not 1:1). Apply every edit to BOTH full_draft.md and the corresponding .tex file. Do not regenerate the .tex from scratch.

## Three changes

### Change 1: Add RPO citation (critical)

The paper's training loss is L = L_pref + 0.5 × L_sft, where L_sft is assistant-only cross-entropy on the chosen completion. The paper calls this an "SFT retention term." This formulation is exactly Regularized Preference Optimization (RPO), which the paper does not cite. A reviewer will ask "isn't this just RPO?"

**1a. Add to references.bib:**

```bibtex
@inproceedings{liu2024rpo,
  title={Provably Mitigating Overoptimization in {RLHF}: Your {SFT} Loss is Implicitly an Adversarial Regularizer},
  author={Liu, Zhihan and Lu, Miao and Zhang, Shenao and Liu, Boyi and Guo, Hongyi and Yang, Yingxiang and Blanchet, Jose and Wang, Zhaoran},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  note={arXiv:2405.16436}
}
```

**1b. §3.1 — after the loss function description.** Find the sentence that reads approximately "The retention term anchors chosen-completion likelihood during preference optimization." Add immediately after it:

> This formulation corresponds to Regularized Preference Optimization (RPO; Liu et al., 2024), which adds an SFT loss on chosen responses to the DPO objective; Razin et al. (2025) identify this SFT regularization as a mitigation for likelihood displacement.

In LaTeX, use `\citep{liu2024rpo}` and `\citep{razin2025}` (check the existing Razin cite key in references.bib and use that).

**1c. §2 Related Work — Razin paragraph.** Find the sentence "Our study operates in a regime with an SFT retention term that anchors chosen likelihood, so the chosen-decline dynamic their framework describes does not manifest." Change to:

> Our study operates in a regime with an SFT retention term (equivalent to RPO; Liu et al., 2024) that anchors chosen likelihood — the mitigation Razin et al. themselves propose — so the chosen-decline dynamic their framework describes is largely suppressed.

**1d. §5.3 — Rafailov/Cal-DPO paragraph.** Find where the paper says "our training objective includes an SFT retention term" (first occurrence in §5.3). Add "(the RPO formulation; Liu et al., 2024)" after "SFT retention term" on that first occurrence only.

### Change 2: Add Pipano et al. ICLR 2026 (recommended)

**2a. Add to references.bib:**

```bibtex
@inproceedings{pipano2026displacement,
  title={Displacement-Resistant Extensions of {DPO} with Nonconvex $f$-Divergences},
  author={Pipano, Idan and Sabach, Shoham and Asadi, Kavosh and Ghavamzadeh, Mohammad},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  note={arXiv:2602.06788}
}
```

**2b. §2 Related Work — Razin paragraph.** After the sentence "Whether displacement would emerge without the retention term in our setup is untested." add:

> Pipano et al. (2026) characterize which $f$-divergence losses are displacement-resistant by construction; our approach instead retains the standard DPO loss and relies on the SFT term for anchoring.

### Change 3: Fix Young citation title

arXiv 2603.00047 was revised (v2, March 2026). The title changed from "What Is the Geometry of the Alignment Tax?" to "What Is the Alignment Tax?"

**3a. references.bib:** Find the Young entry and change the title to `What Is the Alignment Tax?`

**3b. §2 Related Work:** Find any inline reference to this paper. If the text paraphrases the title with "geometry" (e.g., "frames the alignment tax geometrically"), that paraphrase is still accurate (the paper's content is geometric) — leave the prose, only fix the bib title.

## Verification

After all changes:
1. Grep both full_draft.md and main.tex for "RPO" — should appear in §2, §3.1, §5.3
2. Grep references.bib for "liu2024rpo" and "pipano2026displacement" — both present
3. Grep references.bib for "Geometry of the Alignment" — zero matches
4. Recompile: `cd paper/latex && pdflatex main && bibtex main && pdflatex main && pdflatex main`
5. Confirm 0 errors, 0 undefined citations
6. Report: reference count (should be 14, was 12), and the compiled PDF page count

## Do NOT change

- Any numbers, tables, or figures
- Any hedge language ("consistent with", "directionally consistent", etc.)
- The 3+2 falsification count
- Any section structure
