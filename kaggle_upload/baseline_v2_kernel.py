"""Kaggle kernel: baseline re-measurement with scorer v2 (Item 3). No training.

Evaluates the untrained SFT baseline on the SAME 160-item suite (json/router/
extraction) in the identical generation pipeline, scored three-tier (scorer v2),
producing the on-disk reference point for the equivalence test. Stage only.

GPU ON; attach the qwen3-5-4b model dataset for zero downloads.
"""
import glob
import os
import subprocess
import sys

os.environ.setdefault("DELTA_HF_CACHE", "/kaggle/working/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes", "peft", "accelerate"], check=False)

import torch
assert torch.cuda.is_available(), "NOT RUN: no CUDA GPU (enable GPU on the kernel)"
print("GPU:", torch.cuda.get_device_name(0))

code = sorted(glob.glob("/kaggle/input/**/pref_losses.py", recursive=True))
assert code, "NOT RUN: upload the repo dataset (must contain pref_losses.py)"
CODE = os.path.dirname(code[0])
sys.path.insert(0, CODE)
sft = sorted(glob.glob("/kaggle/input/**/sft_pytorch/adapter_config.json", recursive=True))
assert sft, "NOT RUN: SFT adapter not found"
SFT = os.path.dirname(sft[0])

os.chdir("/kaggle/working")
os.makedirs("data_eval", exist_ok=True)
for src_f in glob.glob(os.path.join(CODE, "data_eval", "*.jsonl")):
    dst = os.path.join("data_eval", os.path.basename(src_f))
    if not os.path.exists(dst):
        os.symlink(src_f, dst)

import run_baseline_v2
sys.argv = ["run_baseline_v2", "--sft-adapter", SFT]
run_baseline_v2.main()
print("\nBASELINE_V2_DONE")
