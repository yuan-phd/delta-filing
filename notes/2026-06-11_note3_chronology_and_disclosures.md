# Note 3 — Chronology & disclosures through 2026-06-11

**Methods-integrity backbone:** every number without an on-disk artifact was re-measured
or dropped. This note records the full self-falsification arc and every disclosure so the
paper's claims are traceable to artifacts, not memory.

## Overturned headlines (each replaced by the weaker, supported claim)
1. **"Unbounded DPO causes format-break collapse on rigid tasks; bounded losses prevent it"**
   → **FALSIFIED.** The clean length-normalized run grew the margin to ~6.7 with no collapse.
   Reframed to the **decoupling** thesis (capability preserved as margin grows).
2. **Causal-clamp control ("clamp prevents collapse")** → **dropped as KL-confounded.** With
   no collapse to prevent, the clamp arm was removed; the apparent dynamics are **global
   distribution drift** (KL rises). Matrix is now single-condition vanilla DPO.
3. **"Router is the primary degradation casualty"** → **non-monotonic transient, not a clean
   casualty.** Across checkpoints/both seeds router is flat-but-low (seed0 ~0.76–0.81,
   margin-independent) or chaotic (seed1, format-break-driven mid-run, recovers). The seed1
   mid-run FORMAT_BREAKs are **invented enum-register pseudo-labels** (a closed-set vocabulary
   collapse), **not** verbose style-bleed — hypothesis falsified.
4. **"Targeted likelihood displacement (gold specifically deflated)"** → **control-probe
   preview points to GLOBAL_DRIFT / GOLD_PROTECTED.** Controls A/B fall *more* than gold
   (fp16 n=12 preview, directional); nf4 run is decisive under the frozen X=0.05 mapping.
- Related disclosure: **extraction "degradation" was a section surface-form scorer artifact**
  — `Item N → "N"` register mismatches; canonicalized accuracy ≈ 1.000. Not semantic drift.

## Dissolved "remembered" numbers (no on-disk source → not cited)
1. **Old-suite (80-item) router/extraction baselines** — never saved; no kernel output or log
   contains the `run_baseline_audit` table. Unrecoverable.
2. **"router baseline ~0.950"** — no baseline source; `0.950` is the **DPO json matrix result**
   (conflation). Discarded.
3. **160-item competence gate (json 0.988 / router 0.938 / extraction 0.950, INCLUDE)** — run
   interactively without Save Version; survives nowhere (disk, saved kernels, git). The `0.938`
   on disk is a DPO `margin1 json` result (0.9375). Provisional/unverifiable → **baseline_v2 is
   the first and only verifiable gate**.
4. **Extraction degeneracy floor 0.0 → 0.0063** — re-derived: every gold unique, best constant
   strategy = 1/160 = 0.00625. Corrected and propagated everywhere it is recorded.

## Leakage scoping (so "zero leakage" can't be over-read)
- **Verbatim overlap = 0** (exact string match after `.strip()`) vs both the 647 preference
  questions and the 921 SFT questions.
- **Source-domain overlap is by design (EDGAR ticker×section fetches) and reported as LOWER
  BOUNDS:** ticker×section provenance recoverable for only 209/647 pairs (ticker-only 597/647),
  so true overlap can only be higher. At least 81/160 extraction items and 131/160 json items
  share a ticker (×section where available) with a preference pair's source filing.

## Methodology evolutions (versioned, not silent)
- **Generation fix:** un-terminated greedy outputs (no EOS at `<|im_end|>`) → added eos +
  turn-boundary truncation; output lengths became realistic (the hinge variable).
- **Scorer v1 → v2:** asymmetric format gates → **symmetric begins-with gates**; single-tier →
  **three-tier** (FORMAT_BREAK / SURFACE_FORM / SEMANTIC_ERROR); semantic accuracy primary,
  strict secondary. Frozen canonicalization rule (section id `\d+[A-Z]?`, case-fold, titles
  ignored), unit-tested (24 surface / 1 semantic on the 25 known cases).
- **Control renderer fix:** `control_A_section` prefix collision (`Item 1` ⊂ `Item 1A`) →
  pick a different *base number*; field spans now non-empty 160/160.
- **Probe additions inside existing forwards** (runtime unchanged, asserted): mass-logging
  (entropy / top-5 / gold rank) + field-span deltas.

## Exploration → confirmation discipline
- **X = 0.05 nats/token frozen from the fp16 probe BEFORE the nf4/control preview** (no
  threshold tuning after seeing the controls).
- **Absolute-Δgold decline** noticed in exploration, flagged as under-reported, then **promoted
  to a load-bearing confirmation** — reported *separately* from noise-sensitive gap/contrast
  numbers (which carry bootstrap CIs).
- The pre-registered outcome mappings (register probe, behavioral branches) were written
  **before** their runs; no outcome conditions on memory.

## Suite & identity changes
- **80 → 160-item eval suite** (grown as a superset; binary task dropped as un-fixable for a
  rigid-format collapse claim — 2-token label, degeneracy invisible, floored at 0.50). Router
  gold double-adjudicated (gpt-4o): 160/160 agree.
- **Checkpoint-identity hash closure (2026-06-11):** the actual pilot SFT
  (`yuanmazax/delta-filing-smoke`) downloaded and hashed = `29a3794b…585f02` = repo = upload;
  tokenizer_config + chat_template byte-identical. Pilot init == baseline_v2 init == matrix
  init, byte-for-byte. The last "cannot verify locally" flag is closed.
