# Kaggle smoke-gate runbook (Track 1)

## 1. Upload dataset
Upload the **contents of `kaggle_upload/`** as ONE Kaggle dataset (preserve folder structure).
It already contains everything the kernel needs:

```
pref_losses.py  scorer.py  train_pref.py  eval_pref.py  run_smoke.py  smoke_kernel.py
data_eval/track1_binary.jsonl   data_eval/track1_json.jsonl
adapters/sft_pytorch/{adapter_config.json,adapter_model.safetensors,tokenizer*.json,chat_template.jinja}
training_data/dpo_pytorch.jsonl
```
UI: Datasets → New Dataset → drag the `kaggle_upload` folder → Create.
(CLI alt, from inside `kaggle_upload/`: `kaggle datasets init -p .` then edit the slug in
`dataset-metadata.json` and `kaggle datasets create -p . --dir-mode zip`.)

## 2. Kernel settings
- New **Notebook** → Add Data → attach the dataset above.
- Accelerator: **GPU** (T4 x2 or P100 — single GPU is used; 4-bit 4B ≈ 3 GB).
- **Internet: ON** (Qwen3.5-4B downloads from HF).

## 3. Run (one cell)
```python
import glob
try:
    exec(open(glob.glob('/kaggle/input/**/smoke_kernel.py', recursive=True)[0]).read())
except SystemExit as e:
    print('exit code:', e.code)
```
Expected exit code: `0` = both gates passed; `2` = GATE 1 failed (baseline format mismatch);
`3` = GATE 2 failed (margin didn't climb). 2/3 mean STOP — bring the log back, do not run more.
(If you hit a transformers/peft error at import, prepend a cell:
`!pip -q install -U transformers peft accelerate bitsandbytes` and re-run.)

## 4. Bring back to the reviewer
- The **full cell stdout** (it prints GATE 1 + GATE 2 verdicts, the margin trajectory,
  the DPO generations with verdict tags, and the scorer intermediates).
- The artifact files from `/kaggle/working/` (Output tab → download, or Save Version):
  - `logs/track1_dpo_seed0_margin.jsonl`
  - `generations/track1_baseline_seed0_binary.jsonl`, `..._baseline_seed0_json.jsonl`
  - `generations/track1_dpo_seed0_binary.jsonl`, `..._dpo_seed0_json.jsonl`
  - `results/track1_baseline_seed0_{binary,json}.json`
  - `results/track1_dpo_seed0_{binary,json}.json`

Do NOT launch the full matrix regardless of exit code — review first.

---

# Track-1 MATRIX runbook (4 kernels, one condition each)

## A. One-time: build the model dataset (zero downloads in the matrix kernels)
1. New **Notebook**, Accelerator **None/CPU**, **Internet ON**.
2. Attach the code dataset, run one cell:
   ```python
   import glob
   exec(open(glob.glob('/kaggle/input/**/make_model_dataset.py', recursive=True)[0]).read())
   ```
   (downloads Qwen3.5-4B into `/kaggle/working/Qwen3.5-4B`).
3. **Save Version (Save & Run All)** → on the output, **Create Dataset / New Dataset from
   Notebook Output**. Name it e.g. `qwen3-5-4b`. (Folder `Qwen3.5-4B/` → auto-detected.)

## B. Upload the code dataset version
Re-drag the whole `kaggle_upload/` as a **New Version** of the code dataset
(`train_pref.py`, `run_matrix.py`, `matrix_kernel.py`, `make_model_dataset.py` are new/changed).

## C. Run one kernel per condition (4 total)
New **Notebook** → **Add Data**: attach BOTH the **code dataset (latest version)** and the
**`qwen3-5-4b`** model dataset. Accelerator **GPU** (T4 x2 / P100), **Internet ON** (for pip).
Launch cell — change only the condition string:
```python
import os, glob
os.environ["DELTA_CONDITION"] = "dpo"     # one of: dpo, dpo_clamp, ipo, cdpo
try:
    exec(open(glob.glob('/kaggle/input/**/matrix_kernel.py', recursive=True)[0]).read())
except SystemExit as e:
    print('exit code:', e.code)
```
Confirm zero model download: the log should print
`[cache] model source for Qwen/Qwen3.5-4B: /kaggle/input/.../Qwen3.5-4B (local_files_only=True)`
with **no download progress bar**.

**Order:** run `dpo` first → bring results back → confirm timing+sanity → run `dpo_clamp`,
`ipo`, `cdpo`.

## D. Bring back per kernel
Download `/kaggle/working/{logs,generations,results}/`. Outputs are no-overwrite and unique
per condition×seed, so the 4 kernels merge cleanly into one `results/` + `logs/` +
`generations/`. (`track1_baseline_seed0_*` is recomputed in each kernel — identical and
deterministic; keep any one copy.)
Per condition you get: 5 adapters, 5 `logs/track1_<cond>_seed{0..4}_margin.jsonl`,
10 `generations/...`, 10 `results/...`, and the condition rows in
`results/track1_matrix_summary.json`.

---

# Script 2 — REWARD DECOMPOSITION (chosen vs rejected) runbook

Closes the #3 gap: decompose the DPO margin into reward_chosen (chosen logp − SFT) and
reward_rejected (rejected logp − SFT) across all 8 seeds × {margin1, margin3, margin5, final},
to show whether the margin grows by chosen↑ or rejected↓ — globally and per axis
(style vs tool_calling). Runs on T4 with the SAME 4-bit NF4 quant the matrix trained under.

## Inputs (already staged in `kaggle_upload/`)
- `kaggle_reward_decomposition_kernel.py` — the kernel (T4×2 launcher + worker + analysis).
- `matrix_adapters/seed{0..7}/{final,ckpt_margin1,ckpt_margin3,ckpt_margin5}/` — the 8 seeds'
  trained adapters (config + safetensors) + per-seed `checkpoints.json` (~193 MB total).
- `adapters/sft_pytorch/` — the SFT reference adapter (already present).
- `training_data/dpo_pytorch.jsonl` — the 647 pairs → 549 train / 98 valid (DATA_SEED=42).
- `train_pref.py`, `pref_losses.py` — imported for the faithful pairs/teacher-forcing.

## Attach + run
New **Notebook** → **Add Data**: attach BOTH the **code dataset (the version that now
includes `matrix_adapters/`)** and the **`qwen3-5-4b`** model dataset. Accelerator
**GPU T4 ×2** (uses both: 4 seeds/GPU), **Internet ON** (pip). One cell:
```python
import glob
exec(open(glob.glob('/kaggle/input/**/kaggle_reward_decomposition_kernel.py',
                    recursive=True)[0]).read())
```
The launcher splits seeds [0,2,4,6]→GPU0, [1,3,5,7]→GPU1 (auto-falls back to all-8-serial
on a single GPU). ~70 min wall on T4×2.

## Outputs (`/kaggle/working/`)
- `reward_decomposition/sft.jsonl` + `seed{0..7}_{margin1,margin3,margin5,final}.jsonl`
  (per-pair `{pair_id, chosen_logp, rejected_logp, category, axis}`) — 33 files.
- `reward_decomposition/summary.json` (per-seed + aggregate decomposition; the printed table).
- `figures/reward_decomposition_chosen_vs_rejected.png` and `_by_axis.png`.

