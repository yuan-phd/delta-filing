"""Kaggle kernel entrypoint for the Track-1 smoke GATE (run_smoke.py).

Self-locating: finds the repo code, SFT adapter, and preference data under
/kaggle/input/* so there are no hardcoded dataset slugs. Attach as datasets:
  - the repo code (must contain pref_losses.py + run_smoke.py + data_eval/*.jsonl)
  - the SFT adapter dir (…/sft_pytorch/adapter_config.json)  [may be same dataset]
  - dpo_pytorch.jsonl                                         [may be same dataset]

Kernel settings: GPU ON, Internet ON (Qwen3.5-4B downloads from HF).
Outputs land in /kaggle/working/{logs,generations,results}.
"""
import glob
import os
import subprocess
import sys

# Persistent HF cache so Qwen3.5-4B downloads once and is reused across every
# build_model() call (set before train_pref is imported).
os.environ.setdefault("DELTA_HF_CACHE", "/kaggle/working/hf_cache")

# 4-bit QLoRA stack (Kaggle base images vary; pin what the study needs).
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes", "peft", "accelerate", "datasets"], check=False)

import torch
assert torch.cuda.is_available(), "NOT RUN: no CUDA GPU (enable GPU on the kernel)"
print("GPU:", torch.cuda.get_device_name(0))


def find(pattern):
    return sorted(glob.glob(pattern, recursive=True))


code = find("/kaggle/input/**/pref_losses.py")
assert code, "NOT RUN: upload the repo as a dataset (must contain pref_losses.py)"
CODE = os.path.dirname(code[0])
sys.path.insert(0, CODE)
print("CODE dir:", CODE)

# eval sets must ship with the code dataset (smoke slice used by the gate)
assert os.path.exists(os.path.join(CODE, "data_eval", "track1_binary_smoke.jsonl")), \
    "NOT RUN: data_eval/track1_binary_smoke.jsonl missing from the code dataset"

sft = find("/kaggle/input/**/sft_pytorch/adapter_config.json")
assert sft, "NOT RUN: SFT adapter (…/sft_pytorch/adapter_config.json) not found"
SFT = os.path.dirname(sft[0])
print("SFT adapter:", SFT)

dpo = find("/kaggle/input/**/dpo_pytorch.jsonl")
PREF = dpo[0] if dpo else os.path.join(CODE, "training_data", "dpo_pytorch.jsonl")
print("preference data:", PREF)

os.makedirs("/kaggle/working", exist_ok=True)
os.chdir("/kaggle/working")
# eval_pref reads EVAL_FILES/SMOKE_EVAL_FILES relative to cwd; symlink ALL eval
# sets (smoke + full) into place.
os.makedirs("data_eval", exist_ok=True)
for src_f in glob.glob(os.path.join(CODE, "data_eval", "*.jsonl")):
    dst = os.path.join("data_eval", os.path.basename(src_f))
    if not os.path.exists(dst):
        os.symlink(src_f, dst)

import run_smoke
sys.argv = ["run_smoke",
            "--sft-adapter", SFT,
            "--pref-data", PREF,
            "--output-base", "/kaggle/working/adapters/pref",
            "--max-items", "5",
            "--smoke-n", "60",
            "--max-steps", "30",
            "--log-every", "2"]
rc = run_smoke.main()
print("\nSMOKE_RETURN_CODE", rc)
sys.exit(rc)
