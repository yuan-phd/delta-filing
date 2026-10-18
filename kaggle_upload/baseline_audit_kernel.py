"""Kaggle kernel entrypoint for the baseline-competence gate (run_baseline_audit.py).

No training. Evaluates the untrained SFT baseline (free generation) on all
candidate tasks and prints the INCLUDE/EXCLUDE gate per task. Same self-locating
setup as the other kernels. GPU ON; attach the qwen3-5-4b model dataset for zero
downloads (else Internet ON for the one-time fetch).
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

import run_baseline_audit
sys.argv = ["run_baseline_audit", "--sft-adapter", SFT,
            "--tasks", "binary", "json", "router", "extraction"]
run_baseline_audit.main()
print("\nBASELINE_AUDIT_DONE")
