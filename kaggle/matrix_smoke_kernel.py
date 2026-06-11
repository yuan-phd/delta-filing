"""Kaggle SMOKE kernel for the new matrix pipeline (run before the 10-run matrix).

Exercises the full new path on a small slice:
  train (reduced steps) with checkpoint-saving at LOW margin thresholds
  -> eval capability at EVERY checkpoint on ALL 3 tasks (json, router, extraction)
  -> analyze.py builds the capability-vs-margin curve data on disk.

Verifies checkpoint-eval works on all three tasks and the curve lands correctly
BEFORE committing 10 real runs. GPU ON; attach qwen3-5-4b for zero downloads.
"""
import glob
import os
import subprocess
import sys

os.environ.setdefault("DELTA_HF_CACHE", "/kaggle/working/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes", "peft", "accelerate", "datasets", "matplotlib"], check=False)

import torch
assert torch.cuda.is_available(), "NOT RUN: no CUDA GPU"
print("GPU:", torch.cuda.get_device_name(0))

code = sorted(glob.glob("/kaggle/input/**/pref_losses.py", recursive=True))
assert code, "NOT RUN: upload the repo dataset"
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
            "--conditions", "dpo", "--seeds", "0",
            "--tasks", "json", "router", "extraction",
            "--smoke", "--smoke-n", "60", "--max-steps", "40", "--log-every", "2",
            "--max-items", "8", "--margin-checkpoints", "0.2", "0.4", "0.6",
            "--step-checkpoints", "5"]
run_matrix.main()

print("\n=== analyze.py: capability-vs-margin curve from smoke data ===")
import analyze
sys.argv = ["analyze", "--results-dir", "results", "--logs-dir", "logs",
            "--figures-dir", "figures", "--out", "RESULTS_smoke.md"]
analyze.main()
print("\nSMOKE_PIPELINE_DONE — check results/, logs/, generations/, figures/, RESULTS_smoke.md")
