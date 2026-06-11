# Claude Code Execution Spec — Format-Dissociation Preference-Optimization Study

**Role:** You are implementing a controlled ML experiment in the existing `delta-filing` repo. Do NOT change the research design. Do NOT fabricate or infer any number — every reported value must come from an actual run and be written to a results file. If something cannot be run, write `NOT RUN` and stop, do not invent it.

**Goal of the experiment (context, do not restate in code):** Show that an unbounded preference loss (DPO) causes a *format-break* collapse on rigid short-format tasks while leaving complex JSON outputs intact, that bounded losses (IPO, cDPO, and DPO+clamp) prevent it, and that this is not explained by output length alone. Two tracks: in-domain (Qwen3.5-4B + SFT, SEC tasks) and public (Llama-3.2-3B-Instruct + UltraFeedback + IFEval/BoolQ/function-calling).

---

## 0. Hard rules

1. Never fabricate metrics. Every number → a results file with the run that produced it.
2. **Save every model generation** to disk (jsonl). The experiment is impossible without raw generations.
3. Set and record the seed for every run (Python, NumPy, Torch, and the data sampler/shuffle).
4. Keep all hyperparameters identical across conditions within a track. The ONLY thing that differs between conditions is the loss function.
5. After writing any training/eval script, dry-run it on 2–3 examples before launching a full run. Trace the loss and the scorer on a tiny batch and print intermediate values.
6. Use 4-bit NF4 QLoRA. Print and assert model memory < 5 GB after load.
7. Commit results files; do not overwrite prior runs (suffix with track/condition/seed).

---

## 1. Global config

```
BASE_MODELS:
  track1: Qwen/Qwen3.5-4B  + existing SFT LoRA adapter (adapters/sft_pytorch)
  track2: meta-llama/Llama-3.2-3B-Instruct  (no SFT; instruct model IS the baseline/reference)

LORA: r=8, alpha=16, dropout=0.05, target=[q_proj,k_proj,v_proj,o_proj], bias=none
QUANT: 4-bit nf4, double quant, fp16 compute
PREF_OPT: beta=0.5, epochs=1, length_normalized_logprobs=True,
          sft_retention_weight=0.5, optimizer=AdamW(wd=0.01), grad_clip=1.0,
          early_stop: min_steps=100, threshold=0.005 (rolling 20-step pref loss)
SEEDS: track1=[0,1,2,3,4]  track2=[0,1,2]
CLAMP_C: 3.0   # for DPO+clamp; record this value
```

---

## 2. Code changes to the existing training code

Work from the ablation notebook logic (`notebook/kaggle/delta-filing-dpo-training-ablation.ipynb`). Port to a parameterized script `train_pref.py` that takes `--track --condition --seed`.

**2a. Fix the SFT-retention mask bug.** Current retention CE is computed over the full sequence. Change to compute cross-entropy **only on assistant tokens** (mask system/user/prompt tokens with label = -100). This applies to all four conditions identically (so the comparison stays controlled). Add an assertion that the number of unmasked label tokens equals the assistant span length.

**2b. Add the DPO+clamp condition.** New loss:
```
margin = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
# DPO:        loss = -logsigmoid(margin)
# DPO+clamp:  loss = -logsigmoid(torch.clamp(margin, -CLAMP_C, CLAMP_C))
```
Everything else identical to DPO. This is the causal control — implement it as a first-class condition, not an afterthought.

**2c. Loss inventory (confirm each matches these exactly):**
```
DPO:       -F.logsigmoid(margin)
DPO+clamp: -F.logsigmoid(clamp(margin, -C, C))
IPO:       (clamp(margin,-10,10) - 1/(2*beta))**2          # target = 1.0 at beta=0.5
cDPO:      -(1-eps)*F.logsigmoid(margin) - eps*F.logsigmoid(-margin)   # eps=0.1
```

**2d. Margin trajectory logging.** At every logging step (e.g. every 10 steps) write to `logs/{track}_{condition}_seed{S}_margin.jsonl`: `{step, train_margin, pref_loss}`. Every 50 steps also log a validation margin computed on a fixed held-out preference subset (same subset across all runs in a track). This fills the gap where IPO/cDPO trajectories were never logged.

**2e. Determinism.** Seed everything; log the seed into every output file's header.

---

## 3. Data preparation

**3a. Track 1 — expand the two pivotal eval tasks.** Using GPT-4o-mini as labeler on real SEC filing content (same method as the existing data), generate held-out test items so that:
- **Binary-classification task (Review):** ~80 items. Output space is exactly {`COMPLETE`, `NEEDS_FOLLOW_UP`}. Balanced classes. Each item has a gold label.
- **Complex-JSON task (Tool calling/parsing):** ~80 items. Gold = the correct tool name + required arguments. Score programmatically (see §6).
- Keep the other existing categories as-is for context, but the two above are the dissociation axis and need adequate n.
- Enforce: no overlap with training data; no `<think>` tags; no placeholders. Reuse `tests/test_data_quality.py`-style checks.

**3b. Track 1 — preference training data.** Reuse existing 647 preference pairs (or regenerate cleanly with the same generator). No change to training data content; only the retention mask (2a) changes.

**3c. Track 2 — preference training data.** `HuggingFaceH4/ultrafeedback_binarized` (or `openbmb/UltraFeedback` binarized). Sample a fixed subset (~3–5k pairs); the SAME subset for all conditions and seeds within track 2. Length-normalize logprobs as in track 1.

**3d. Track 2 — eval sets (all public, all programmatic):**
- **IFEval** (`google/IFEval`): rigid-format end. Use the official instruction-following verifiers for the format gate.
- **BoolQ** (`google/boolq` validation): rigid short end. Gold yes/no. Format gate = normalized output ∈ {yes,no}.
- **Function-calling / JSON** (complex-structured end): sample ~300 prompts with gold calls from a public set (e.g. `glaiveai/glaive-function-calling-v2` held-out split, or BFCL prompts). Gold = function name + arguments. Score = JSON validity + name match + argument match.
- For each eval task, run the **baseline** model first (pre-PO) to get the tax-delta reference.

---

## 4. Training run matrix

```
Track 1:  conditions × seeds = {DPO, DPO+clamp, IPO, cDPO} × {0,1,2,3,4}  = 20 runs
Track 2:  conditions × seeds = {DPO, DPO+clamp, IPO, cDPO} × {0,1,2}      = 12 runs
Baselines (eval only, no training): track1 SFT adapter; track2 Llama instruct as-is
```
Each run saves: the adapter, the margin trajectory (2d), the training config header (incl. seed, clamp C, retention weight).

---

## 5. Eval harness

One script `eval_pref.py --track --condition --seed --task`. For each generation it MUST persist to `generations/{track}_{condition}_seed{S}_{task}.jsonl`:
```
{item_id, prompt, gold, raw_output, output_token_len,
 format_pass (bool), normalization_applied (bool),
 content_correct (bool|null), verdict ∈ {PASS, FORMAT_BREAK, REASONING_ERROR}}
```
No metric is computed without the underlying generations on disk.

---

## 6. Two-stage scorer (the core measurement)

Implement `score(task, raw_output, gold) -> verdict, fields`. Pure functions, deterministic, no LLM.

```
def score(task, raw, gold):
    s = normalize(raw)                      # strip ws; record if changed
    # ---- stage 1: FORMAT GATE ----
    if task == "binary":
        labels = {"COMPLETE","NEEDS_FOLLOW_UP"}     # track1
        # track2 boolq: {"yes","no"}
        format_pass = strip_trailing_punct(s).upper() in labels
    elif task == "json" or task == "function_call":
        obj = try_json_parse(first_json_span(raw))
        format_pass = obj is not None and required_keys_present(obj, task)
    elif task == "ifeval":
        format_pass = ifeval_verifier(prompt, raw)   # official verifier
    if not format_pass:
        return "FORMAT_BREAK", {...}
    # ---- stage 2: CONTENT ----
    correct = matches_gold(task, parsed_or_label, gold)
    return ("PASS" if correct else "REASONING_ERROR"), {...}
```

Rules to pin down and DOCUMENT in the results header:
- `normalize`: strip whitespace; for binary, strip a single trailing `.`; case-insensitive compare. Record `normalization_applied` per item so we can report how many passes needed it.
- `first_json_span`: extract the first balanced `{...}`; if the model wrapped JSON in prose, that still counts as format_pass ONLY IF a valid JSON object is present (record `had_surrounding_prose` too — useful signal).
- `matches_gold` for function_call: function name exact + required args match (value-normalized).
- BoolQ content: map yes/no to the gold boolean.

Per (track, condition, seed, task) write `results/{...}.json`:
```
{accuracy, format_break_rate, reasoning_error_rate, n,
 mean_output_len, median_output_len, pct_normalized}
```

---

## 7. Aggregation + the length-controlled analysis (the hinge)

Script `analyze.py` produces:

**7a. Main results table** (per track): for each condition, mean ± std across seeds of {accuracy, format_break_rate} for each task, plus every per-seed raw value in an appendix table.

**7b. Margin trajectories figure:** overlay DPO / DPO+clamp / IPO / cDPO margin vs step (mean across seeds, light per-seed lines). Expect DPO to climb unbounded; IPO flat near 1.0; clamp flat near C; cDPO bounded.

**7c. Verbosity figure:** mean output length per task per condition.

**7d. THE HINGE — length-controlled format-break:**
- Headline number: DPO format_break_rate on `binary` vs on `json`. (Claim needs binary high, json ≈ baseline.)
- Pool all generations across conditions; bucket by `output_token_len` (e.g. quartiles or fixed-width bins). Within each bucket, plot format_break_rate for `binary` vs `json`. 
- Output an explicit verdict line to `results/hinge_verdict.txt`: does JSON survive at lengths where binary breaks? (YES = dissociation is not pure length; NO = narrow the claim.) Report the actual bucketed numbers regardless of which way it goes.

**7e. Causal control readout:** compare DPO vs DPO+clamp format_break_rate on `binary`. If clamp materially reduces it → boundedness is causal. State the delta.

---

## 8. Output artifacts (deliverables back to the researcher)

```
logs/        margin trajectories (jsonl, per run)
generations/ every raw generation (jsonl, per run×task)   <-- mandatory
results/     per-run scored metrics (json) + main tables + hinge_verdict.txt
figures/     margin trajectory, verbosity, length-controlled format-break
RESULTS.md   terse summary: main table, hinge verdict, clamp causal delta,
             per-seed spread. Numbers only, each with its source file. No narrative spin.
```

---

## 9. Order of execution

1. Code changes §2 (retention mask fix, DPO+clamp, margin logging, generation saving). Dry-run on tiny batch; trace loss + scorer values.
2. Data prep §3 (track1 eval expansion via GPT-4o-mini; track2 datasets + baseline eval).
3. Smoke test: 1 condition × 1 seed, full pipeline end-to-end on a small slice; verify generations + scores land on disk and verdict tags are sane.
4. Launch the full run matrix §4.
5. Eval §5–6 on all adapters + baselines.
6. Analyze §7. Write RESULTS.md.
7. Report back: surface the hinge verdict and the clamp causal delta first — those decide the paper's claim strength.

If anything is ambiguous or a dataset/verifier is unavailable, stop and ask rather than substitute. Do not silently change the design.
