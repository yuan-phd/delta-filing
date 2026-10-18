# Pre-registration — capability/margin decoupling study (Track 1)

## Thesis (current)
Does **free-generation capability** on rigid exact-format tasks stay **preserved**
as the DPO preference margin grows **unbounded**?

The earlier hypothesis ("unbounded DPO causes format-break collapse on rigid
short-format tasks") was **falsified** by our clean run (margin grew to ~6.7, no
collapse). The current claim is the **decoupling**: capability and margin growth
are dissociated.

## Both outcomes are reported honestly
- **Preserved** (capability flat/high as margin grows) → decoupling, our main claim.
- **Collapsed** (capability falls as margin grows) → revives the old format-break claim.

We do **NOT** tune tasks toward producing a collapse. Doing so would be p-hacking.

## Pre-registered task-inclusion gate (decided on the BASELINE only)
A task enters the final suite **iff**, measured on the untrained SFT baseline and
**before seeing any DPO result**:
1. **Competence** — baseline free-generation accuracy **≥ 0.80**.
2. **Headroom** — baseline accuracy **≤ ~0.97** (not pinned at 1.0, so a drop is observable).
3. **Format-fragility** — leading prose / prose-instead-of-structure **breaks** the format gate.
4. **Degeneracy-visible** — a constant/degenerate answer scores well **below** baseline
   (router floor 1/5; extraction/json ~1/n) so collapse-to-constant is detectable.
5. **Hygiene** — balanced classes, n ≈ 80, zero overlap with training, gold-as-output → PASS.

Inclusion is **independent of whether DPO later changes the task.**

## Candidate tasks (chosen before DPO results)
| task | format family | status |
|------|---------------|--------|
| json (tool-call) | nested JSON object | baseline 0.975 (prior run) → passes |
| router | closed-set 5-class enum | new; gate via baseline audit |
| extraction | flat JSON schema + date regex | new; gate via baseline audit |
| ~~binary (Review)~~ | 2-token label | **DROPPED** (can't register collapse; floored at 0.50) |

The final suite = the candidate tasks that clear the baseline gate
(`run_baseline_audit.py`). Target: ≥3 tasks spanning ≥2 format families.

## Measurement (frozen)
- Capability = **free greedy generation** (`do_sample=False`) + exact-format
  scorer on the generated string. **Never teacher-forced.** (Teacher-forced
  log-probs are used only for the DPO margin / training trajectory.)
- Measured **along the margin trajectory**: adapter checkpoints saved at smoothed
  margin ≈ 1, 3, 5, and final; capability evaluated at each.
- Conditions: {dpo, dpo_clamp, ipo, cdpo} × seeds {0..4}. Hyperparameters identical
  across conditions; only the loss differs.

## Analysis
Per condition, plot **capability vs margin** across checkpoints (mean ± std over
seeds). Decoupling = capability stays high/flat as margin grows; collapse =
capability declines with margin. Report the curve regardless of direction.

## Frozen before the matrix runs
Task suite (post baseline-gate), scorer + gates, seeds, checkpoint margins,
generation config. Changes after this point are logged as deviations.
