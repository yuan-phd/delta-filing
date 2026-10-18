"""Kaggle kernel — Script 2: chosen/rejected REWARD DECOMPOSITION across the matrix.

Closes the #3 gap: the matrix never logged chosen/rejected rewards, so we could not
tell whether the DPO margin grows because chosen-logp RISES or rejected-logp FALLS.
This teacher-forces the chosen AND rejected completion of each of the 549 TRAIN pairs
(the exact 85% split training used, DATA_SEED=42) through every checkpoint and logs
per-pair length-normalized completion log-probs, then decomposes the margin.

This is the GPU/T4 (4-bit NF4, EXACTLY the matrix's quantization) counterpart of the
local mac_reward_decomposition.py — same faithful method, same outputs, just run on
the same nf4 base the matrix trained under.

FAITHFUL METHOD (identical to mac_reward_decomposition.py):
  * pairs/split    : train_pref.build_pairs + split_pairs (same tokenization, same
                     system-prompt-by-category, same DATA_SEED) -> the identical 549 pairs.
  * teacher-forcing: pref_losses.compute_log_probs (completion-only, length-normalized) —
                     the SAME function training uses for reference log-probs.
  * base           : 4-bit NF4 double-quant, fp16 compute (train_pref.build_model config).
  * reward         : reward = policy_logp - reference_logp (reference = SFT adapter).
                     margin = beta * (reward_chosen - reward_rejected),  beta = train_pref.BETA.
  * axis           : 'tool_calling' (category==tool_calling) vs 'style' (all other) — the
                     same binary split training uses to pick the system prompt.

PARALLELISM (T4 x2): the 8 seeds are split across the two GPUs (4 per GPU). Each GPU
runs an independent worker subprocess pinned via CUDA_VISIBLE_DEVICES; peak ~3.2 GB/GPU
so no interference. Each worker also computes the shared SFT reference once on its own
GPU (deterministic -> identical jsonl). Launcher then canonicalizes sft.jsonl, runs the
analysis, and writes figures. Falls back to one GPU (all 8 seeds serial) if only 1 visible.

LAUNCH CELL (GPU T4 x2 ON, Internet ON for pip; attach the code dataset that now contains
matrix_adapters/  AND the qwen3-5-4b model dataset):

    import glob
    exec(open(glob.glob('/kaggle/input/**/kaggle_reward_decomposition_kernel.py',
                        recursive=True)[0]).read())

Outputs -> /kaggle/working/reward_decomposition/{sft.jsonl, seed{S}_{ck}.jsonl, summary.json}
           /kaggle/working/figures/reward_decomposition_*.png
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time

# ----------------------------------------------------------------------------
# Shared setup (both launcher and worker run this)
# ----------------------------------------------------------------------------
os.environ.setdefault("DELTA_HF_CACHE", "/kaggle/working/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CKPTS = ["margin1", "margin3", "margin5", "final"]
OUT_DIR = "/kaggle/working/reward_decomposition"
FIG_DIR = "/kaggle/working/figures"

# Locate the uploaded code dataset (pref_losses.py / train_pref.py) and add to path.
_code = sorted(glob.glob("/kaggle/input/**/pref_losses.py", recursive=True))
assert _code, "NOT RUN: upload the code dataset (must contain pref_losses.py / train_pref.py)"
CODE = os.path.dirname(_code[0])
if CODE not in sys.path:
    sys.path.insert(0, CODE)

# SFT reference adapter (the matrix's reference) and the matrix adapters tree.
_sft = sorted(glob.glob("/kaggle/input/**/sft_pytorch/adapter_config.json", recursive=True))
assert _sft, "NOT RUN: SFT adapter (sft_pytorch) not found in the attached datasets"
SFT = os.path.dirname(_sft[0])

_madir = sorted(glob.glob("/kaggle/input/**/matrix_adapters/seed0/final/adapter_config.json",
                          recursive=True))
assert _madir, ("NOT RUN: matrix_adapters/ not found — re-upload the code dataset version "
                "that includes kaggle_upload/matrix_adapters/seed{0..7}/")
MATRIX_ADAPTERS = os.path.dirname(os.path.dirname(os.path.dirname(_madir[0])))  # .../matrix_adapters

# Preference data (the 647 raw pairs -> 549 train / 98 valid).
_dpo = sorted(glob.glob("/kaggle/input/**/dpo_pytorch.jsonl", recursive=True))
PREF = _dpo[0] if _dpo else os.path.join(CODE, "training_data", "dpo_pytorch.jsonl")
assert os.path.exists(PREF), f"NOT RUN: preference data not found ({PREF})"


def ckpt_dir(seed, ck):
    """Adapter dir for (seed, checkpoint). 'final' is the seed's top-level adapter."""
    base = os.path.join(MATRIX_ADAPTERS, f"seed{seed}")
    return os.path.join(base, "final") if ck == "final" else os.path.join(base, f"ckpt_{ck}")


def seed_margin(seed, ck):
    """Smoothed train margin recorded for this checkpoint (for figure labels)."""
    cj = os.path.join(MATRIX_ADAPTERS, f"seed{seed}", "checkpoints.json")
    if not os.path.exists(cj):
        return None
    m = json.load(open(cj))
    return {c["label"]: c["smoothed_margin"] for c in m["checkpoints"]}.get(ck)


# ============================================================
# Worker: teacher-force a set of seeds on ONE GPU
# ============================================================
def build_train_pairs(tok):
    """The exact 549 train pairs (mirrors train_pref.build_pairs/split_pairs) + category/axis."""
    import torch
    import train_pref as T
    raw = [json.loads(l) for l in open(PREF) if l.strip()]
    pairs = []
    for idx, ex in enumerate(raw):
        sp = T.system_prompt_for("track1", ex.get("category", "analysis"))
        prompt_msgs = [{"role": "system", "content": sp},
                       {"role": "user", "content": ex["question"]}]
        ctext = tok.apply_chat_template(
            prompt_msgs + [{"role": "assistant", "content": ex["chosen"]}], tokenize=False)
        rtext = tok.apply_chat_template(
            prompt_msgs + [{"role": "assistant", "content": ex["rejected"]}], tokenize=False)
        cids = tok.encode(ctext, max_length=T.MAX_SEQ_LEN, truncation=True)
        rids = tok.encode(rtext, max_length=T.MAX_SEQ_LEN, truncation=True)
        ptext = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        pids = tok.encode(ptext, max_length=T.MAX_SEQ_LEN, truncation=True)
        cs = T._completion_boundary(cids, pids)
        rs = T._completion_boundary(rids, pids)
        if cs >= len(cids) or rs >= len(rids):
            continue
        cat = ex.get("category", "analysis")
        pairs.append({"pair_id": idx, "category": cat,
                      "axis": "tool_calling" if cat == "tool_calling" else "style",
                      "chosen_ids": torch.tensor(cids, dtype=torch.long), "chosen_start": cs,
                      "rejected_ids": torch.tensor(rids, dtype=torch.long), "rejected_start": rs})
    train, valid = T.split_pairs(pairs)
    print(f"  built {len(pairs)} pairs -> train={len(train)} valid={len(valid)} "
          f"(expect 549/98; header match: {len(train) == 549 and len(valid) == 98})", flush=True)
    return train


def load_base_nf4():
    """Load the base model in 4-bit NF4 (EXACTLY train_pref.build_model's quant config)
    on the single visible GPU (CUDA_VISIBLE_DEVICES pins it), with the SFT adapter
    attached as adapter_name='sft'. Returns (peft_model, tok, device)."""
    import torch
    import train_pref as T
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    device = torch.device("cuda")
    tok = T.build_tokenizer(T.TRACK1)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    src, local_only = T.resolve_model_source(T.TRACK1["model_id"])
    print(f"  loading base {src} (4bit-nf4-doublequant-fp16compute) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        src, quantization_config=bnb, trust_remote_code=True, device_map={"": 0},
        cache_dir=T.HF_HUB_CACHE, local_files_only=local_only)
    base.eval()
    pm = PeftModel.from_pretrained(base, SFT, adapter_name="sft")
    pm.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"  [mem] peak after base+SFT load: "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    return pm, tok, device


def score_checkpoint(model, pairs, device, label, out_path):
    """Teacher-force chosen+rejected for all pairs; write per-pair logp jsonl. Resumable."""
    import torch
    import pref_losses as PL
    if os.path.exists(out_path) and len(open(out_path).readlines()) == len(pairs):
        print(f"  [{label}] cached -> {out_path} (skip)", flush=True)
        return
    t0 = time.time()
    with open(out_path, "w") as f, torch.no_grad():
        for i, p in enumerate(pairs):
            cl = float(PL.compute_log_probs(model, p["chosen_ids"], p["chosen_start"], device))
            rl = float(PL.compute_log_probs(model, p["rejected_ids"], p["rejected_start"], device))
            f.write(json.dumps({"pair_id": p["pair_id"], "chosen_logp": cl, "rejected_logp": rl,
                                "category": p["category"], "axis": p["axis"]}) + "\n")
            if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
                dt = time.time() - t0
                eta = dt / (i + 1) * (len(pairs) - i - 1)
                print(f"    [{label}] {i + 1}/{len(pairs)} pairs  {dt:.0f}s  ETA {eta:.0f}s",
                      flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(f"  [{label}] wrote -> {out_path}", flush=True)


def run_worker(seeds, gpu):
    """Score the SFT reference + all assigned seeds x 4 checkpoints on this GPU."""
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[worker gpu{gpu}] seeds={seeds} | device={torch.cuda.get_device_name(0)}", flush=True)
    pm, tok, device = load_base_nf4()
    pairs = build_train_pairs(tok)
    loaded = {"sft"}

    def use(path, name):
        if name not in loaded:
            pm.load_adapter(path, adapter_name=name)
            loaded.add(name)
        pm.set_adapter(name)
        pm.eval()

    # Shared SFT reference (each GPU writes its own copy; launcher canonicalizes).
    use(SFT, "sft")
    score_checkpoint(pm, pairs, device, "sft", os.path.join(OUT_DIR, f"sft_gpu{gpu}.jsonl"))

    for s in seeds:
        for ck in CKPTS:
            path = ckpt_dir(s, ck)
            assert os.path.isdir(path), f"missing adapter: {path}"
            name = f"seed{s}_{ck}"
            use(path, name)
            score_checkpoint(pm, pairs, device, name, os.path.join(OUT_DIR, f"{name}.jsonl"))
            # free the DPO adapter we no longer need (keep 'sft' resident)
            pm.delete_adapter(name)
            loaded.discard(name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[worker gpu{gpu}] DONE seeds={seeds}", flush=True)


# ============================================================
# Analysis (identical to mac_reward_decomposition.analyze)
# ============================================================
def _load(tag):
    p = os.path.join(OUT_DIR, f"{tag}.jsonl")
    return {r["pair_id"]: r for r in (json.loads(l) for l in open(p))} if os.path.exists(p) else None


def analyze():
    import numpy as np
    import train_pref as T
    beta = T.BETA
    sft = _load("sft")
    if sft is None:
        print("  analyze: no sft.jsonl — workers did not complete."); return
    seeds = [s for s in range(8) if all(_load(f"seed{s}_{ck}") for ck in CKPTS)]
    if not seeds:
        print("  analyze: no seed fully cached yet."); return
    ids = sorted(sft)

    def stats(rows, key, axis=None):
        v = [rows[i][key] for i in ids if (axis is None or sft[i]["axis"] == axis)]
        return float(np.mean(v))

    summary = {"beta": beta, "reference": "sft_baseline", "quant": "4bit-nf4-doublequant-fp16compute",
               "seeds": seeds, "by_seed": {}}
    print("\n" + "=" * 88)
    print("REWARD DECOMPOSITION  (reward = policy_logp - sft_logp; margin = beta*(r_chosen - r_rejected))")
    print(f"beta={beta}  reference=SFT  seeds={seeds}\n")
    agg = {ck: {"r_chosen": [], "r_rejected": [], "margin": [],
                "style_margin": [], "tool_margin": [],
                "style_rc": [], "style_rr": [], "tool_rc": [], "tool_rr": []} for ck in CKPTS}
    for s in seeds:
        summary["by_seed"][s] = {}
        for ck in CKPTS:
            pol = _load(f"seed{s}_{ck}")
            rc = stats(pol, "chosen_logp") - stats(sft, "chosen_logp")
            rr = stats(pol, "rejected_logp") - stats(sft, "rejected_logp")
            margin = beta * (rc - rr)
            d = {"reward_chosen": rc, "reward_rejected": rr, "margin": margin,
                 "mean_chosen_logp": stats(pol, "chosen_logp"),
                 "mean_rejected_logp": stats(pol, "rejected_logp"),
                 "smoothed_margin_log": seed_margin(s, ck)}
            for ax, tag in [("style", "style"), ("tool_calling", "tool")]:
                arc = stats(pol, "chosen_logp", ax) - stats(sft, "chosen_logp", ax)
                arr = stats(pol, "rejected_logp", ax) - stats(sft, "rejected_logp", ax)
                d[f"{tag}_reward_chosen"] = arc; d[f"{tag}_reward_rejected"] = arr
                d[f"{tag}_margin"] = beta * (arc - arr)
                agg[ck][f"{tag}_margin"].append(beta * (arc - arr))
                agg[ck][f"{tag}_rc"].append(arc); agg[ck][f"{tag}_rr"].append(arr)
            agg[ck]["r_chosen"].append(rc); agg[ck]["r_rejected"].append(rr); agg[ck]["margin"].append(margin)
            summary["by_seed"][s][ck] = d

    print(f"  {'ckpt':<9}{'r_chosen':>10}{'r_rejected':>12}{'margin':>9}  driver")
    for ck in CKPTS:
        rc = float(np.mean(agg[ck]["r_chosen"])); rr = float(np.mean(agg[ck]["r_rejected"]))
        mg = float(np.mean(agg[ck]["margin"]))
        driver = ("rejected falls" if abs(rr) > abs(rc) else "chosen rises") if mg else "-"
        print(f"  {ck:<9}{rc:>+10.3f}{rr:>+12.3f}{mg:>+9.3f}  {driver} "
              f"(|Δrej|={abs(rr):.3f} vs |Δcho|={abs(rc):.3f})")
    print("\n  By axis at FINAL (mean over seeds):")
    for tag, nm in [("style", "style"), ("tool", "tool_calling")]:
        rc = float(np.mean(agg["final"][f"{tag}_rc"])); rr = float(np.mean(agg["final"][f"{tag}_rr"]))
        mg = float(np.mean(agg["final"][f"{tag}_margin"]))
        print(f"    {nm:<13} margin={mg:+.3f}  (reward_chosen={rc:+.3f}  reward_rejected={rr:+.3f})")
    sm = float(np.mean(agg["final"]["style_margin"])); tm = float(np.mean(agg["final"]["tool_margin"]))
    print(f"    -> style axis {'DOMINATES' if sm > tm else 'does NOT dominate'} "
          f"(style margin {sm:+.3f} vs tool {tm:+.3f})")
    summary["aggregate_final"] = {"style_margin": sm, "tool_margin": tm}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    print(f"  wrote {OUT_DIR}/summary.json")
    make_figures(summary, seeds, agg)


def make_figures(summary, seeds, agg):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    xs = list(range(len(CKPTS)))

    # Fig 1: chosen vs rejected REWARD trajectory (mean over seeds) — the displacement test
    fig, ax = plt.subplots(figsize=(7, 4.6))
    rc = [np.mean(agg[ck]["r_chosen"]) for ck in CKPTS]
    rr = [np.mean(agg[ck]["r_rejected"]) for ck in CKPTS]
    ax.plot(xs, rc, "o-", color="tab:green", label="reward_chosen (chosen logp - sft)")
    ax.plot(xs, rr, "o-", color="tab:red", label="reward_rejected (rejected logp - sft)")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set(xticks=xs, xlabel="checkpoint", ylabel="reward (policy - reference logp)",
           title="Margin decomposition: chosen↑ vs rejected↓ (mean over seeds)")
    ax.set_xticklabels(CKPTS); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/reward_decomposition_chosen_vs_rejected.png", dpi=130)
    plt.close(fig)

    # Fig 2: per-seed margin + per-axis margin trajectory
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for s in seeds:
        axes[0].plot(xs, [summary["by_seed"][s][ck]["margin"] for ck in CKPTS], "o-", alpha=0.6)
    axes[0].set(xticks=xs, title="reconstructed margin per seed", xlabel="checkpoint",
                ylabel="beta*(r_chosen - r_rejected)"); axes[0].set_xticklabels(CKPTS)
    for tag, nm, col in [("style", "style", "tab:blue"), ("tool", "tool_calling", "tab:orange")]:
        m = [np.mean(agg[ck][f"{tag}_margin"]) for ck in CKPTS]
        axes[1].plot(xs, m, "o-", color=col, label=nm)
    axes[1].set(xticks=xs, title="margin by axis (style vs tool_calling)", xlabel="checkpoint",
                ylabel="margin"); axes[1].set_xticklabels(CKPTS); axes[1].legend()
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/reward_decomposition_by_axis.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {FIG_DIR}/reward_decomposition_chosen_vs_rejected.png + _by_axis.png")


# ============================================================
# Launcher: split 8 seeds across the visible GPUs (T4 x2)
# ============================================================
def launch(gpu, seeds):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)                  # pin this child to one physical GPU
    env["DELTA_RD_WORKER"] = "1"
    env["DELTA_RD_GPU"] = str(gpu)
    env["DELTA_RD_SEEDS"] = ",".join(str(s) for s in seeds)
    env["DELTA_HF_CACHE"] = f"/kaggle/working/hf_cache_gpu{gpu}"   # separate cache, no race
    print(f"[LAUNCH] gpu{gpu} <- seeds {seeds} (CUDA_VISIBLE_DEVICES={gpu})", flush=True)
    return subprocess.Popen([sys.executable, os.path.abspath(__file__)],
                            env=env, cwd="/kaggle/working",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)


def pump(proc, tag):
    for line in proc.stdout:
        sys.stdout.write(f"{tag} {line}")
        sys.stdout.flush()


def main():
    import torch
    assert torch.cuda.is_available(), "NOT RUN: no CUDA GPU (enable GPU on the kernel)"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "bitsandbytes", "peft", "accelerate", "datasets"], check=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    ngpu = torch.cuda.device_count()
    print(f"visible GPUs: {ngpu}", flush=True)
    all_seeds = list(range(8))

    # __file__ may be undefined when launched via exec(open(...).read()); persist a copy.
    global __file__
    if "__file__" not in globals() or not os.path.exists(__file__):
        path = "/kaggle/working/kaggle_reward_decomposition_kernel.py"
        with open(path, "w") as f:
            f.write(open(_self_source()).read())
        __file__ = path

    if ngpu >= 2:
        split = {0: all_seeds[0::2], 1: all_seeds[1::2]}     # GPU0:[0,2,4,6] GPU1:[1,3,5,7]
        jobs = []
        for gpu, seeds in split.items():
            p = launch(gpu, seeds)
            t = threading.Thread(target=pump, args=(p, f"[g{gpu}]"), daemon=True)
            t.start()
            jobs.append((gpu, seeds, p, t))
            time.sleep(8)                                    # stagger model-load
        for gpu, seeds, p, t in jobs:
            p.wait(); t.join()
            print(f"  gpu{gpu} seeds={seeds} exit_code={p.returncode}", flush=True)
    else:
        print("Only 1 GPU visible -> running all 8 seeds serially on GPU 0.", flush=True)
        run_worker(all_seeds, gpu=0)

    # Canonicalize the shared SFT reference (gpu0's copy; identical across GPUs).
    sft_canon = os.path.join(OUT_DIR, "sft.jsonl")
    for cand in [f"sft_gpu{g}.jsonl" for g in (0, 1)]:
        cp = os.path.join(OUT_DIR, cand)
        if os.path.exists(cp) and not os.path.exists(sft_canon):
            import shutil
            shutil.copyfile(cp, sft_canon)
    print("\n" + "=" * 60)
    analyze()
    print("REWARD_DECOMPOSITION_DONE")


def _self_source():
    """Best-effort path to this script's source for re-exec under T4x2 launching."""
    if "__file__" in globals() and os.path.exists(__file__):
        return __file__
    hit = sorted(glob.glob("/kaggle/input/**/kaggle_reward_decomposition_kernel.py", recursive=True))
    assert hit, "cannot locate kaggle_reward_decomposition_kernel.py source for worker launch"
    return hit[0]


# ----------------------------------------------------------------------------
# Entry: worker subprocess vs. launcher
# ----------------------------------------------------------------------------
if os.environ.get("DELTA_RD_WORKER") == "1":
    _seeds = [int(s) for s in os.environ["DELTA_RD_SEEDS"].split(",") if s.strip()]
    _gpu = int(os.environ.get("DELTA_RD_GPU", "0"))
    run_worker(_seeds, _gpu)
else:
    main()
