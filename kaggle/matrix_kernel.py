"""Kaggle kernel for ONE Track-1 run: a single (condition, seed).

Split to one run per kernel (full-epoch train + checkpoint-saving + eval at every
margin checkpoint on all 3 tasks) so each kernel sits far under Kaggle's 9h cap.
10 kernels total for the core matrix: {dpo, dpo_clamp} x seeds {0..4}.

Set in the launch cell:
    DELTA_CONDITION in {dpo, dpo_clamp, ipo, cdpo}
    DELTA_SEED      in {0,1,2,3,4}

Baseline is NOT recomputed here (reuse the baseline-audit results when merging).
Outputs are no-overwrite and unique per (condition, seed, checkpoint, task), so all
10 kernels merge cleanly into one results/ + logs/ + generations/.

Attach the qwen3-5-4b model dataset -> zero downloads. GPU ON.
"""
import glob
import os
import subprocess
import sys

os.environ.setdefault("DELTA_HF_CACHE", "/kaggle/working/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CONDITION = os.environ.get("DELTA_CONDITION", "").strip()
SEED = os.environ.get("DELTA_SEED", "").strip()
assert CONDITION in ["dpo", "dpo_clamp", "ipo", "cdpo"], \
    f"NOT RUN: set DELTA_CONDITION (got {CONDITION!r})"
assert SEED.isdigit(), f"NOT RUN: set DELTA_SEED to an integer (got {SEED!r})"

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes", "peft", "accelerate", "datasets"], check=False)

import torch
assert torch.cuda.is_available(), "NOT RUN: no CUDA GPU (enable GPU on the kernel)"
print("GPU:", torch.cuda.get_device_name(0), "| condition:", CONDITION, "| seed:", SEED)

code = sorted(glob.glob("/kaggle/input/**/pref_losses.py", recursive=True))
assert code, "NOT RUN: upload the repo dataset (must contain pref_losses.py)"
CODE = os.path.dirname(code[0])
sys.path.insert(0, CODE)

sft = sorted(glob.glob("/kaggle/input/**/sft_pytorch/adapter_config.json", recursive=True))
assert sft, "NOT RUN: SFT adapter not found"
SFT = os.path.dirname(sft[0])
dpo = sorted(glob.glob("/kaggle/input/**/dpo_pytorch.jsonl", recursive=True))
PREF = dpo[0] if dpo else os.path.join(CODE, "training_data", "dpo_pytorch.jsonl")

os.chdir("/kaggle/working")
os.makedirs("data_eval", exist_ok=True)
for src_f in glob.glob(os.path.join(CODE, "data_eval", "*.jsonl")):
    dst = os.path.join("data_eval", os.path.basename(src_f))
    if not os.path.exists(dst):
        os.symlink(src_f, dst)

import run_matrix
sys.argv = ["run_matrix", "--sft-adapter", SFT, "--pref-data", PREF,
            "--output-base", "/kaggle/working/adapters/pref",
            "--conditions", CONDITION, "--seeds", SEED,
            "--tasks", "json", "router", "extraction",
            "--skip-baseline"]
run_matrix.main()
print(f"\nMATRIX_RUN_DONE condition={CONDITION} seed={SEED}")
