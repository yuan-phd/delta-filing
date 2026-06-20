"""Kaggle PARALLEL launcher — run TWO matrix seeds at once, one per GPU (T4 x2).

Each seed runs `matrix_kernel.py` as an independent subprocess pinned to one GPU via
CUDA_VISIBLE_DEVICES (0 and 1). Peak ~3.2 GB/seed on a 16 GB T4 -> no interference.
Halves wall-time per seed-pair. NO design change: same fixed 180 steps, scorer v2,
divergence guard, everything -- just two instances in parallel.

Set in the launch cell:
    DELTA_CONDITION in {dpo, dpo_clamp, ipo, cdpo}
    DELTA_SEEDS = "3,4"   (one or two seeds; 1 seed -> GPU 0 only)

GPU x2 ON; attach the qwen3-5-4b model dataset (zero downloads, no cache race).
Per-seed outputs are seed-distinct (no overwrite). The other seed continues if one
aborts (divergence guard) or crashes -- the subprocesses are independent.
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time

CONDITION = os.environ.get("DELTA_CONDITION", "").strip()
SEEDS = [s.strip() for s in os.environ.get("DELTA_SEEDS", "").split(",") if s.strip()]
assert CONDITION in ["dpo", "dpo_clamp", "ipo", "cdpo"], \
    f"NOT RUN: set DELTA_CONDITION (got {CONDITION!r})"
assert 1 <= len(SEEDS) <= 2 and all(s.isdigit() for s in SEEDS), \
    f"NOT RUN: set DELTA_SEEDS to '<seed>' or '<seed>,<seed>' (got {SEEDS!r})"

# install once in the parent so the children's matrix_kernel pip is a fast no-op
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes", "peft", "accelerate", "datasets"], check=False)

import torch
NGPU = torch.cuda.device_count()
print(f"visible GPUs: {NGPU}", flush=True)
assert NGPU >= len(SEEDS), (
    f"NOT RUN: {len(SEEDS)} seeds requested but only {NGPU} GPU(s) visible "
    "(enable GPU T4 x2 accelerator)")

mk = sorted(glob.glob("/kaggle/input/**/matrix_kernel.py", recursive=True))
assert mk, "NOT RUN: matrix_kernel.py not found in the uploaded dataset"
MK = mk[0]
CODE = os.path.dirname(MK)

# pre-create data_eval symlinks once (so the two children don't race on symlink)
os.chdir("/kaggle/working")
os.makedirs("data_eval", exist_ok=True)
for src in glob.glob(os.path.join(CODE, "data_eval", "*.jsonl")):
    dst = os.path.join("data_eval", os.path.basename(src))
    if not os.path.exists(dst):
        os.symlink(src, dst)


def launch(gpu, seed):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)               # pin this child to one GPU
    env["DELTA_CONDITION"] = CONDITION
    env["DELTA_SEED"] = str(seed)
    env["DELTA_HF_CACHE"] = f"/kaggle/working/hf_cache_gpu{gpu}"   # separate cache, no race
    print(f"[LAUNCH] condition={CONDITION} seed={seed} -> physical GPU {gpu} "
          f"(CUDA_VISIBLE_DEVICES={gpu})", flush=True)
    return subprocess.Popen([sys.executable, MK], env=env, cwd="/kaggle/working",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)


def pump(proc, tag):
    for line in proc.stdout:
        sys.stdout.write(f"{tag} {line}")
        sys.stdout.flush()


jobs = []      # (gpu, seed, proc, thread)
for gpu, seed in enumerate(SEEDS):
    p = launch(gpu, seed)
    t = threading.Thread(target=pump, args=(p, f"[g{gpu}/s{seed}]"), daemon=True)
    t.start()
    jobs.append((gpu, seed, p, t))
    time.sleep(8)   # small stagger so model-load/setup of the two children don't collide

# wait for ALL — one aborting/crashing does NOT stop the other (independent procs)
for gpu, seed, p, t in jobs:
    p.wait()
    t.join()

print("\n" + "=" * 60)
print("PARALLEL RUN COMPLETE")
for gpu, seed, p, _ in jobs:
    cj = f"/kaggle/working/adapters/pref/track1_{CONDITION}_seed{seed}/checkpoints.json"
    aborted = "?"
    if os.path.exists(cj):
        m = json.load(open(cj))
        aborted = f"{m.get('aborted')} ({m.get('abort_reason')})" if m.get("aborted") else False
    print(f"  seed {seed} (GPU {gpu}): exit_code={p.returncode}  aborted={aborted}")

# the per-seed results/generations/logs/adapters are seed-distinct (safe). The ONLY
# shared file is results/track1_matrix_summary.json, which both children overwrite ->
# remove it; analyze.py rebuilds the curve from the per-seed results.
summ = "/kaggle/working/results/track1_matrix_summary.json"
if os.path.exists(summ):
    os.remove(summ)
    print("  (removed the shared track1_matrix_summary.json — analyze.py rebuilds the "
          "curve from per-seed results; per-seed result JSONs are intact)")
print("PARALLEL_MATRIX_DONE")
