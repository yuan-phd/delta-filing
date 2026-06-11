"""Kaggle kernel: register-shift CONTROL probe (nf4, faithful). No training.

Computes per-token completion log-prob + KL for gold/A/B/C across 9 checkpoints on
the full 160-item extraction suite, writing results/register_probe_nf4.jsonl, then
applies the FROZEN decision rule (register_decision.py, X=0.05).

Requires the checkpoint adapters (kaggle_runs/seed{0,1}.../adapters/...) and the
frozen control C (data_eval/track1_extraction_controlC.jsonl) in the dataset.
GPU ON; attach qwen3-5-4b for zero downloads.
"""
import glob
import os
import subprocess
import sys

os.environ.setdefault("DELTA_HF_CACHE", "/kaggle/working/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes", "peft", "accelerate", "numpy"], check=False)

import torch
assert torch.cuda.is_available(), "NOT RUN: no CUDA GPU"
print("GPU:", torch.cuda.get_device_name(0))

code = sorted(glob.glob("/kaggle/input/**/probe_register_shift.py", recursive=True))
assert code, "NOT RUN: upload the repo dataset (must contain probe_register_shift.py)"
CODE = os.path.dirname(code[0])
sys.path.insert(0, CODE)
os.chdir(CODE)   # probe reads data_eval/ + kaggle_runs/ relative paths

# the checkpoint adapters must be present (uploaded under kaggle_runs/)
assert glob.glob(os.path.join(CODE, "kaggle_runs", "seed0*", "adapters", "**", "ckpt_margin1"),
                 recursive=True), "NOT RUN: checkpoint adapters (kaggle_runs/.../ckpt_margin*) not found"

import probe_register_shift as P
sys.argv = ["probe", "--quant", "--out", "/kaggle/working/register_probe_nf4.jsonl"]
P.main()

print("\n=== FROZEN decision (X=0.05) ===")
import register_decision as D
sys.argv = ["decide", "--probe", "/kaggle/working/register_probe_nf4.jsonl"]
D.main()
print("\nREGISTER_PROBE_DONE")
