#!/usr/bin/env python
"""Parameterized preference-optimization training for the format-dissociation study.

    python train_pref.py --track track1 --condition dpo --seed 0
    python train_pref.py --track track1 --condition dpo_clamp --seed 0 --smoke

Implements claude_code_spec.md §2:
  2a  assistant-only SFT-retention CE (mask system/user/prompt with -100) + assertion
  2b  DPO+clamp condition (first-class)
  2c  loss inventory: dpo / dpo_clamp / ipo / cdpo
  2d  margin-trajectory logging -> logs/{track}_{condition}_seed{S}_margin.jsonl
  2e  determinism: seed everything; seed written into every output header

Heavy deps (transformers / bitsandbytes / peft) are imported lazily inside
build_model(), so this file can be imported on a CPU/MPS box (no bitsandbytes)
for the loss-math dry-run. The actual 4-bit NF4 QLoRA training path requires a
CUDA GPU (Kaggle), per the agreed compute plan.
"""
import argparse
import gc
import glob
import json
import os
import random
import time

# Reduce CUDA fragmentation (set before any CUDA allocation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import pref_losses as PL

# ============================================================
# Model caching (download base model ONCE, reuse on every build_model)
# ============================================================
HF_CACHE_DIR = (os.environ.get("DELTA_HF_CACHE")
                or os.environ.get("HF_HOME")
                or "/kaggle/working/hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
HF_HUB_CACHE = os.path.join(HF_CACHE_DIR, "hub")
_RESOLVED = {}   # model_id -> (local_path, local_files_only)


def _norm(s):
    """Lowercase + drop non-alphanumerics, so 'Qwen3.5-4B' / 'qwen35-4b' /
    'qwen3-5-4b' all normalize to the same token for matching."""
    import re
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _detect_local_model(model_id, search_roots=("/kaggle/input",)):
    """Find a base-model dir (one containing config.json) whose path matches the
    model's short name, tolerant of Kaggle slug mangling (dots/hyphens removed).
    Returns the dir or None. (The SFT adapter has adapter_config.json, not
    config.json, so it is never matched here.)"""
    short_n = _norm(model_id.split("/")[-1])
    if not short_n:
        return None
    for root in search_roots:
        for p in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
            d = os.path.dirname(p)
            if short_n in _norm(d):
                return d
    return None


def resolve_model_source(model_id):
    """Return (path, local_files_only) for from_pretrained, downloading the base
    model at most once. Order: explicit DELTA_MODEL_DIR -> attached Kaggle dataset
    (auto-detected) -> snapshot_download into a persistent cache (idempotent).
    """
    if model_id in _RESOLVED:
        return _RESOLVED[model_id]

    env_dir = os.environ.get("DELTA_MODEL_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, "config.json")):
        res = (env_dir, True)
    else:
        hit = _detect_local_model(model_id)
        if hit:
            res = (hit, True)
        else:
            from huggingface_hub import snapshot_download
            res = (snapshot_download(model_id, cache_dir=HF_HUB_CACHE), True)
    _RESOLVED[model_id] = res
    print(f"[cache] model source for {model_id}: {res[0]} (local_files_only={res[1]})")
    return res

# ============================================================
# §1 global config (constants identical across conditions within a track)
# ============================================================
BETA = 0.5
SFT_WEIGHT = 0.5                 # sft_retention_weight (§1)
LR = 5e-5
EPOCHS = 6      # raised from 1: at effective-batch 8 (~69 opt-steps/epoch) the margin
                # only reaches ~2 in 1 epoch; ~6 epochs (~414 opt-steps) lets the DPO
                # margin climb back into the ~5-6 high-margin regime. DPO early-stops
                # once it converges (pref-loss<0.005 <=> margin>~5.3); clamp runs full.
# Effective batch via gradient accumulation (noise reduction): accumulate this
# many per-sample grads, then one optimizer step. One forward at a time, so peak
# memory stays at batch-1 level. `step` below counts OPTIMIZER steps. Loss math is
# unchanged -- we only divide each sample's loss by ACCUM so the accumulated grad
# is the batch MEAN (standard accumulation).
EFFECTIVE_BATCH = 8
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 1024
LOG_EVERY = 10
EVAL_EVERY = 50
MIN_STEPS_BEFORE_EARLY_STOP = 140    # floor raised 100->140 (~2 epochs at batch-8) so a
                                     # fluke low pref-loss can't stop us before the high-
                                     # margin regime; consistent with the longer run.
EARLY_STOP_THRESHOLD = 0.005         # KEPT: -logsigmoid(margin)<0.005 <=> margin>~5.3, so
                                     # this fires ONLY in the high-margin regime (genuine
                                     # DPO convergence) -- changing it would mis-target.
EARLY_STOP_WINDOW = 20
CLAMP_C = PL.DEFAULT_CLAMP_C         # 3.0, recorded in header
CDPO_EPS = PL.DEFAULT_CDPO_EPS       # 0.1
VAL_SUBSET_N = 20                    # fixed held-out subset for val margin (§2d)
DATA_SEED = 42                       # fixes train/valid split + val subset across ALL runs
# Margin-keyed checkpoints: save the adapter the first time the smoothed train
# margin crosses each level, so capability can be evaluated ALONG the margin
# trajectory (decoupling = capability-vs-margin curve, not just the endpoint).
MARGIN_CHECKPOINTS = [1.0, 3.0, 5.0]
MARGIN_SMOOTH_WINDOW = 20
# Early FIXED-STEP checkpoint(s) (not margin-triggered): localize the onset region
# before margin1 fires (~50-100 steps in), to distinguish "dropped before margin1"
# from an actual onset curve. Evaluated on all 3 tasks + gold-logp like the others.
STEP_CHECKPOINTS = [20]

LORA_CFG = dict(r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

SYSTEM_PROMPT = (
    "You are Delta Filing, a financial analyst AI specializing in SEC filing analysis. "
    "You analyze 10-K and 10-Q filings, detect year-over-year changes in risk disclosures, "
    "track management guidance accuracy, and flag potential red flags. "
    "Always reference specific filing sections, cite specific numbers and dates, "
    "provide analytical judgment, and note caveats. "
    "Do not give investment advice or predict stock prices."
)
TOOL_SYSTEM_PROMPT = (
    "You are Delta Filing, a financial analyst AI specializing in SEC filing analysis. "
    "You have access to tools for retrieving SEC filings, financial data, news, and insider trades. "
    "When you need data, call the appropriate tool. When you have data, analyze it thoroughly.\n\n"
    "Available tools:\n\n"
    "1. search_filings(ticker, filing_type, count)\n"
    "2. get_filing_section(ticker, section_id, filing_type)\n"
    "3. diff_filing_sections(ticker, section_id, filing_type)\n"
    "4. stock_price(ticker, period)\n"
    "5. company_metrics(ticker)\n"
    "6. company_news(ticker, days)\n"
    "7. insider_trades(ticker)\n"
    "8. analyst_ratings(ticker)\n\n"
    'To use a tool, respond with JSON: {"tool": "name", "arguments": {...}}\n'
    "Respond with ONLY the JSON, nothing else."
)

TRACK1 = dict(
    model_id="Qwen/Qwen3.5-4B",
    sft_adapter="adapters/sft_pytorch",     # ref + trainable init (override via --sft-adapter)
    pref_data="training_data/dpo_pytorch.jsonl",
    chat_template=SIMPLE_CHAT_TEMPLATE,
)
TRACK2 = dict(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    sft_adapter=None,                        # instruct model IS the reference/baseline
    pref_data="training_data/ultrafeedback_subset.jsonl",  # produced by data_prep; NOT yet downloaded
    chat_template=None,                      # use the model's native chat template
)


# ============================================================
# Data
# ============================================================
def system_prompt_for(track, category):
    if track == "track1":
        return TOOL_SYSTEM_PROMPT if category == "tool_calling" else SYSTEM_PROMPT
    return None  # track2: prompts come from the dataset's own messages


def _completion_boundary(seq_ids, prompt_ids):
    """Length of the prompt-token prefix shared by `seq_ids` (longest common prefix
    with prompt_ids). This is the completion start index for `seq_ids`."""
    m = 0
    for a, b in zip(seq_ids, prompt_ids):
        if a != b:
            break
        m += 1
    return m


def build_pairs(track, tokenizer, raw_data):
    """Tokenize preference pairs, build assistant-only retention labels (§2a) and
    the completion boundaries used for completion-only preference log-probs.

    Returns list of dicts: chosen_ids, rejected_ids, chosen_labels,
    chosen_start, rejected_start, assistant_span_len.
    """
    pairs = []
    for ex in raw_data:
        if track == "track1":
            sp = system_prompt_for(track, ex.get("category", "analysis"))
            user, chosen, rejected = ex["question"], ex["chosen"], ex["rejected"]
            prompt_msgs = [{"role": "system", "content": sp},
                           {"role": "user", "content": user}]
        else:
            # track2 ultrafeedback: expect {"prompt", "chosen", "rejected"}
            user, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]
            prompt_msgs = [{"role": "user", "content": user}]

        chosen_msgs = prompt_msgs + [{"role": "assistant", "content": chosen}]
        rejected_msgs = prompt_msgs + [{"role": "assistant", "content": rejected}]

        # Full sequences.
        chosen_text = tokenizer.apply_chat_template(chosen_msgs, tokenize=False)
        rejected_text = tokenizer.apply_chat_template(rejected_msgs, tokenize=False)
        chosen_ids = tokenizer.encode(chosen_text, max_length=MAX_SEQ_LEN, truncation=True)
        rejected_ids = tokenizer.encode(rejected_text, max_length=MAX_SEQ_LEN, truncation=True)

        # Prompt-only (system+user+generation header). The completion boundary is
        # the longest common prefix of each full sequence with the prompt -- this
        # is both the retention-mask boundary (chosen) and the completion_start for
        # the completion-only preference log-probs (chosen + rejected).
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, max_length=MAX_SEQ_LEN, truncation=True)

        chosen_start = _completion_boundary(chosen_ids, prompt_ids)
        rejected_start = _completion_boundary(rejected_ids, prompt_ids)
        if chosen_start >= len(chosen_ids) or rejected_start >= len(rejected_ids):
            # Completion got truncated away; skip this pathological pair.
            continue

        chosen_ids_t = torch.tensor(chosen_ids, dtype=torch.long)
        labels = PL.build_assistant_labels(chosen_ids_t, chosen_start)
        span = PL.assert_assistant_mask(labels, chosen_ids_t, chosen_start)

        pairs.append({
            "chosen_ids": chosen_ids_t,
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "chosen_labels": labels,
            "chosen_start": chosen_start,
            "rejected_start": rejected_start,
            "assistant_span_len": span,
        })
    return pairs


def split_pairs(pairs):
    """Deterministic 85/15 split using DATA_SEED (identical across all runs)."""
    idx = list(range(len(pairs)))
    rng = random.Random(DATA_SEED)
    rng.shuffle(idx)
    cut = int(len(idx) * 0.85)
    train_idx, valid_idx = idx[:cut], idx[cut:]
    train = [pairs[i] for i in train_idx]
    valid = [pairs[i] for i in valid_idx]
    return train, valid


# ============================================================
# Model (lazy heavy imports; CUDA / 4-bit NF4 QLoRA path)
# ============================================================
def build_tokenizer(cfg):
    from transformers import AutoTokenizer
    src, local_only = resolve_model_source(cfg["model_id"])
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True,
                                        cache_dir=HF_HUB_CACHE, local_files_only=local_only)
    if cfg["chat_template"]:
        tok.chat_template = cfg["chat_template"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(cfg, trainable, device):
    """Load base in 4-bit NF4 QLoRA; attach LoRA. §0 rule 6: assert mem < 5 GB."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig, get_peft_model

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    src, local_only = resolve_model_source(cfg["model_id"])
    base = AutoModelForCausalLM.from_pretrained(
        src, quantization_config=bnb, trust_remote_code=True, device_map={"": 0},
        cache_dir=HF_HUB_CACHE, local_files_only=local_only)

    if cfg["sft_adapter"]:                       # track1: start from SFT adapter
        model = PeftModel.from_pretrained(base, cfg["sft_adapter"], is_trainable=trainable)
    elif trainable:                              # track2: fresh LoRA on instruct base
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **LORA_CFG))
    else:                                        # track2 reference: bare instruct model
        model = base

    # §0 rule 6: print + assert footprint after load.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        # §0 rule 6 footprint print kept for visibility. Assertion removed
        # (researcher-authorized): max_memory_allocated is the cumulative process
        # peak across multiple model loads (ref + trainable + eval), so it ratchets
        # up and is not a meaningful per-load budget check. Model runs fine on the
        # 16 GB T4. Guard only -- no training/loss logic changed.
        print(f"[mem] cumulative peak CUDA mem after load: {mem_gb:.2f} GB")
    return model


def _ref_logprobs(ref_model, pairs, device):
    cache = []
    with torch.no_grad():
        for i, p in enumerate(pairs):
            rc = float(PL.compute_log_probs(ref_model, p["chosen_ids"], p["chosen_start"], device))
            rr = float(PL.compute_log_probs(ref_model, p["rejected_ids"], p["rejected_start"], device))
            cache.append({"ref_chosen": rc, "ref_rejected": rr})
            if (i + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return cache


def reference_logprobs(cfg, train_pairs, valid_pairs, device):
    """Pre-compute reference log-probs for train+valid with ONE ref-model load,
    then free it (gc + empty_cache) so the trainable model has the GPU to itself."""
    ref_model = build_model(cfg, trainable=False, device=device)
    ref_model.eval()
    ref_train = _ref_logprobs(ref_model, train_pairs, device)
    ref_valid = _ref_logprobs(ref_model, valid_pairs, device)
    del ref_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()   # footprint print below reflects training only
    return ref_train, ref_valid


# ============================================================
# Training loop
# ============================================================
def make_header(args, cfg, n_train, n_valid, device):
    return {
        "_header": True,
        "track": args.track, "condition": args.condition, "seed": args.seed,
        "model_id": cfg["model_id"], "sft_adapter": cfg["sft_adapter"],
        "beta": BETA, "sft_retention_weight": SFT_WEIGHT, "clamp_C": CLAMP_C,
        "cdpo_eps": CDPO_EPS, "lr": LR, "epochs": EPOCHS, "grad_clip": GRAD_CLIP,
        "effective_batch": getattr(args, "accum_steps", None) or EFFECTIVE_BATCH,
        "batching": "gradient accumulation (one forward/sample, step every N)",
        "weight_decay": WEIGHT_DECAY, "max_seq_len": MAX_SEQ_LEN,
        "lora": LORA_CFG, "quant": "4bit-nf4-doublequant-fp16compute",
        "early_stop": {"min_steps": MIN_STEPS_BEFORE_EARLY_STOP,
                       "threshold": EARLY_STOP_THRESHOLD, "window": EARLY_STOP_WINDOW},
        "data_seed": DATA_SEED, "n_train": n_train, "n_valid": n_valid,
        "val_subset_n": min(VAL_SUBSET_N, n_valid),
        "device": str(device), "smoke": args.smoke,
        "retention_mask": "assistant-only (-100 on system/user/prompt)",
        "pref_logprobs": "completion-only, normalized by completion length",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def train(args):
    cfg = TRACK1 if args.track == "track1" else TRACK2
    if args.sft_adapter:
        cfg = dict(cfg, sft_adapter=args.sft_adapter)
    if args.pref_data:
        cfg = dict(cfg, pref_data=args.pref_data)

    seed_rec = PL.set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs("logs", exist_ok=True)
    out_dir = os.path.join(args.output_base, f"{args.track}_{args.condition}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(cfg["pref_data"]):
        raise SystemExit(
            f"NOT RUN: preference data {cfg['pref_data']} missing. "
            f"(track2 UltraFeedback subset is not downloaded yet — see data_prep.)")

    print(f"[1] Loading + tokenizing preference data: {cfg['pref_data']}")
    raw = [json.loads(l) for l in open(cfg["pref_data"]) if l.strip()]
    if args.smoke:
        raw = raw[:args.smoke_n]
    tokenizer = build_tokenizer(cfg)
    pairs = build_pairs(args.track, tokenizer, raw)
    train_pairs, valid_pairs = split_pairs(pairs)
    print(f"    pairs={len(pairs)}  train={len(train_pairs)}  valid={len(valid_pairs)}")

    print("[2] Pre-computing reference log-probs (length-normalized)...")
    ref_train, ref_valid = reference_logprobs(cfg, train_pairs, valid_pairs, device)

    log_path = f"logs/{args.track}_{args.condition}_seed{args.seed}_margin.jsonl"
    header = make_header(args, cfg, len(train_pairs), len(valid_pairs), device)
    header["seed_record"] = seed_rec
    with open(log_path, "w") as f:
        f.write(json.dumps(header) + "\n")

    print(f"[3] Training condition={args.condition} seed={args.seed} -> {out_dir}")
    model = build_model(cfg, trainable=True, device=device)
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)

    log_every = getattr(args, "log_every", None) or LOG_EVERY
    eval_every = getattr(args, "eval_every", None) or EVAL_EVERY

    accum = getattr(args, "accum_steps", None) or EFFECTIVE_BATCH
    step = 0                            # OPTIMIZER steps (one per `accum` samples)
    pref_losses, margins = [], []       # per-sample (micro) histories
    indices = list(range(len(train_pairs)))
    early_stopped = False
    margin_ckpts = []                   # [{label, dir, step, smoothed_margin}]
    saved_levels = set()
    ckpt_levels = getattr(args, "margin_checkpoints", None) or MARGIN_CHECKPOINTS
    step_levels = getattr(args, "step_checkpoints", None) or STEP_CHECKPOINTS
    saved_steps = set()

    def optimizer_step_bookkeeping():
        """Run after each optimizer step: checkpoints / logging / val / early-stop."""
        nonlocal step, early_stopped
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        step += 1

        w = min(len(margins), MARGIN_SMOOTH_WINDOW)
        smoothed = sum(margins[-w:]) / w
        for lvl in ckpt_levels:
            if lvl not in saved_levels and smoothed >= lvl:
                cdir = os.path.join(out_dir, f"ckpt_margin{lvl:g}")
                model.save_pretrained(cdir)
                saved_levels.add(lvl)
                margin_ckpts.append({"label": f"margin{lvl:g}", "dir": cdir,
                                     "step": step, "smoothed_margin": smoothed})
                print(f"  [ckpt] smoothed margin {smoothed:.2f} >= {lvl} "
                      f"at step {step} -> {cdir}")
        # early FIXED-STEP checkpoint(s) — onset localization before margin1
        for sl in step_levels:
            if sl not in saved_steps and step >= sl:
                cdir = os.path.join(out_dir, f"ckpt_step{sl}")
                model.save_pretrained(cdir)
                saved_steps.add(sl)
                margin_ckpts.append({"label": f"step{sl}", "dir": cdir,
                                     "step": step, "smoothed_margin": smoothed})
                print(f"  [ckpt] fixed step {step} (>= {sl}) -> {cdir}")

        if step % log_every == 0:
            rec = {"step": step,
                   "train_margin": sum(margins[-w:]) / w,
                   "pref_loss": sum(pref_losses[-w:]) / w}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  step {step:4d} | pref={rec['pref_loss']:.4f} | margin={rec['train_margin']:.3f}")

        if step % eval_every == 0:
            vm = validate(model, valid_pairs, ref_valid, args, device)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "val_margin": vm["val_margin"],
                                    "val_pref_loss": vm["val_pref_loss"]}) + "\n")
            print(f"  step {step:4d} | VAL pref={vm['val_pref_loss']:.4f} | margin={vm['val_margin']:.3f}")
            model.train()

        if (step >= MIN_STEPS_BEFORE_EARLY_STOP and len(pref_losses) >= EARLY_STOP_WINDOW):
            recent = sum(pref_losses[-EARLY_STOP_WINDOW:]) / EARLY_STOP_WINDOW
            if recent < EARLY_STOP_THRESHOLD:
                print(f"  early stop at step {step}: rolling pref {recent:.4f} < {EARLY_STOP_THRESHOLD}")
                early_stopped = True
        if args.smoke and step >= args.max_steps:
            print(f"  smoke: reached max_steps={args.max_steps}")
            early_stopped = True

    optimizer.zero_grad()
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(indices)            # seeded by set_seed(args.seed)
        micro = 0                          # samples accumulated in the current batch
        for i_idx, idx in enumerate(indices):
            p = train_pairs[idx]
            ref = ref_train[idx]
            loss, m = PL.combined_loss(
                model, p["chosen_ids"], p["rejected_ids"], p["chosen_labels"],
                p["chosen_start"], p["rejected_start"],
                ref["ref_chosen"], ref["ref_rejected"], device,
                beta=BETA, condition=args.condition, sft_weight=SFT_WEIGHT,
                clamp_c=CLAMP_C, eps=CDPO_EPS)

            if torch.isnan(loss):          # skip this sample; don't count toward batch
                print(f"  (opt-step {step+1}): NaN loss on a sample, skipping it")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            (loss / accum).backward()      # mean over the accumulated batch
            pref_losses.append(m["pref"])
            margins.append(m["margin"])
            micro += 1

            if micro == accum:             # full accumulated batch -> one optimizer step
                optimizer_step_bookkeeping()
                micro = 0
                if early_stopped:
                    break
        # flush a partial final batch (e.g. epoch tail / skipped samples)
        if micro > 0 and not early_stopped:
            optimizer_step_bookkeeping()
            micro = 0
        if early_stopped:
            break

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    final_margin = (sum(margins[-min(len(margins), MARGIN_SMOOTH_WINDOW):])
                    / min(len(margins), MARGIN_SMOOTH_WINDOW)) if margins else float("nan")
    # final checkpoint = the run dir itself
    margin_ckpts.append({"label": "final", "dir": out_dir, "step": step,
                         "smoothed_margin": final_margin})
    with open(os.path.join(out_dir, "checkpoints.json"), "w") as f:
        json.dump({"run": f"{args.track}_{args.condition}_seed{args.seed}",
                   "checkpoints": margin_ckpts}, f, indent=2)
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump({**header, "final_step": step, "early_stopped": early_stopped,
                   "final_smoothed_margin": final_margin,
                   "margin_checkpoints": [c["label"] for c in margin_ckpts]}, f, indent=2)
    print(f"[done] {args.condition} seed{args.seed}: {step} steps -> {out_dir}")
    print(f"       checkpoints: {[c['label'] for c in margin_ckpts]}")
    print(f"       margin trajectory: {log_path}")

    # free training memory so a following eval session has the GPU to itself
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate(model, valid_pairs, ref_valid, args, device):
    """Validation margin on the FIXED held-out subset (§2d)."""
    model.eval()
    n = min(VAL_SUBSET_N, len(valid_pairs))
    vl, vm = [], []
    with torch.no_grad():
        for vi in range(n):
            vp, rv = valid_pairs[vi], ref_valid[vi]
            l, m = PL.combined_loss(
                model, vp["chosen_ids"], vp["rejected_ids"], vp["chosen_labels"],
                vp["chosen_start"], vp["rejected_start"],
                rv["ref_chosen"], rv["ref_rejected"], device,
                beta=BETA, condition=args.condition, sft_weight=SFT_WEIGHT,
                clamp_c=CLAMP_C, eps=CDPO_EPS)
            if not torch.isnan(l):
                vl.append(m["pref"])
                vm.append(m["margin"])
    return {"val_pref_loss": sum(vl) / len(vl) if vl else float("nan"),
            "val_margin": sum(vm) / len(vm) if vm else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["track1", "track2"], required=True)
    ap.add_argument("--condition", choices=list(PL.CONDITIONS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-base", default="adapters/pref")
    ap.add_argument("--sft-adapter", default=None, help="override track1 SFT adapter path")
    ap.add_argument("--pref-data", default=None, help="override preference data path")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end slice")
    ap.add_argument("--smoke-n", type=int, default=60, help="#pairs loaded in smoke mode")
    ap.add_argument("--max-steps", type=int, default=30, help="smoke-mode step cap")
    ap.add_argument("--log-every", type=int, default=None, help="override margin log interval")
    ap.add_argument("--eval-every", type=int, default=None, help="override val interval")
    ap.add_argument("--margin-checkpoints", nargs="+", type=float, default=None)
    ap.add_argument("--step-checkpoints", nargs="+", type=int, default=None,
                    help="fixed-step checkpoint(s); default [20]")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
