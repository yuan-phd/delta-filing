"""Pure preference-loss + retention-mask primitives for the format-dissociation study.

This module deliberately imports ONLY torch / torch.nn.functional so it can be
imported and unit-tested on CPU/MPS without bitsandbytes or CUDA. All model-loading
(4-bit NF4 QLoRA) lives in train_pref.py.

Implements the loss inventory of claude_code_spec.md §2c exactly:

    margin = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    DPO:       -logsigmoid(margin)
    DPO+clamp: -logsigmoid(clamp(margin, -C, C))            # §2b causal control
    IPO:       (clamp(margin, -10, 10) - 1/(2*beta))**2     # target = 1.0 at beta=0.5
    cDPO:      -(1-eps)*logsigmoid(margin) - eps*logsigmoid(-margin)   # eps=0.1

NOTE on scope of the §2a retention-mask fix: §2a changes ONLY the SFT-retention
cross-entropy term so it is computed over assistant tokens only. The preference
log-probs (compute_log_probs) remain full-sequence, length-normalized average
log-probs exactly as in the existing ablation notebook -- the spec does not ask to
change them, and §0 rule 4 forbids touching anything but the loss function across
conditions. See README_PREF or the delivery notes for this interpretation.
"""

import os
import random

import torch
import torch.nn.functional as F

CONDITIONS = ("dpo", "dpo_clamp", "ipo", "cdpo")

# Defaults from §1 global config.
DEFAULT_BETA = 0.5
DEFAULT_CLAMP_C = 3.0          # §1 CLAMP_C, used by dpo_clamp
DEFAULT_CDPO_EPS = 0.1         # §2c cDPO eps / label smoothing
IPO_MARGIN_CLIP = 10.0         # §2c IPO clamp range (matches existing notebook)


# ============================================================
# Determinism (§2e)
# ============================================================
def set_seed(seed: int) -> dict:
    """Seed Python, NumPy, and Torch (CPU+CUDA). Returns a record for headers."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        numpy_seeded = True
    except Exception:
        numpy_seeded = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Best-effort determinism; harmless on CPU/MPS.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {"seed": seed, "numpy_seeded": numpy_seeded,
            "cuda": torch.cuda.is_available()}


# ============================================================
# §2c Preference loss inventory
# ============================================================
def preference_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected,
                    beta=DEFAULT_BETA, condition="dpo",
                    clamp_c=DEFAULT_CLAMP_C, eps=DEFAULT_CDPO_EPS):
    """Compute the preference loss for one pair.

    Inputs are per-token AVERAGE (length-normalized) log-probs -- scalars (float
    or 0-dim tensor). Returns (loss_tensor, margin_float).

    The four conditions differ ONLY in how `margin` is mapped to a loss; the
    margin itself is identical across conditions (§0 rule 4).
    """
    def _t(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(float(x))

    pi_chosen, pi_rejected = _t(pi_chosen), _t(pi_rejected)
    ref_chosen, ref_rejected = _t(ref_chosen), _t(ref_rejected)

    chosen_reward = pi_chosen - ref_chosen
    rejected_reward = pi_rejected - ref_rejected
    margin = beta * (chosen_reward - rejected_reward)

    if condition == "dpo":
        loss = -F.logsigmoid(margin)
    elif condition == "dpo_clamp":
        loss = -F.logsigmoid(torch.clamp(margin, -clamp_c, clamp_c))
    elif condition == "ipo":
        target = 1.0 / (2.0 * beta)
        margin_clipped = torch.clamp(margin, -IPO_MARGIN_CLIP, IPO_MARGIN_CLIP)
        loss = (margin_clipped - target) ** 2
    elif condition == "cdpo":
        loss = -(1 - eps) * F.logsigmoid(margin) - eps * F.logsigmoid(-margin)
    else:
        raise ValueError(f"Unknown condition: {condition!r}; "
                         f"expected one of {CONDITIONS}")

    return loss, float(margin)


# ============================================================
# Preference log-probs: COMPLETION-ONLY, normalized by completion length
# ============================================================
LOGPROB_CHUNK = 256   # sequence-length chunk for the vocab reduction (memory cap)


def compute_log_probs(model, input_ids, completion_start, device,
                      autocast_dtype=torch.float16, chunk_size=LOGPROB_CHUNK):
    """Per-token AVERAGE log-prob of the COMPLETION tokens ONLY, normalized by
    completion length.

    Prompt tokens (positions < completion_start) are NOT scored, so the
    preference margin carries no prompt-length artifact -- essential because the
    study is about length effects. Applied identically to chosen and rejected and
    across all four conditions (§0 rule 4).

    Memory-frugal: we never materialize a full-vocab [1, L-1, V] fp32 log_softmax
    (V ~= 248k -> ~1 GB each, the OOM cause). Instead we slice to the completion-
    predicting positions only, then compute the target log-prob as
    `target_logit - logsumexp(logits)` in chunks over sequence length. The math is
    identical to log_softmax(.).gather(.) -- only the memory footprint changes.

    input_ids       : 1-D LongTensor for the full prompt+completion sequence.
    completion_start: index of the first completion (assistant) token; equals the
                      retention-mask prompt_len. Tokens [completion_start, L-1] are
                      the completion.
    """
    assert completion_start >= 1, "completion_start must be >= 1 (need a preceding token to predict from)"
    x = input_ids.unsqueeze(0).to(device)
    L = x.shape[1]
    n_comp = L - completion_start
    assert n_comp >= 1, f"empty completion span (L={L}, completion_start={completion_start})"

    use_amp = device.type == "cuda"
    ctx = torch.amp.autocast("cuda", dtype=autocast_dtype) if use_amp \
        else torch.autocast(device_type=device.type, enabled=False)
    with ctx:
        out = model(x)
        logits = out.logits                        # [1, L, V]

    # Positions that predict the completion tokens: logits[cs-1 .. L-2] -> tokens[cs .. L-1]
    comp_logits = logits[:, completion_start - 1:L - 1, :]      # [1, n_comp, V] (view)
    comp_labels = x[:, completion_start:L]                      # [1, n_comp]

    parts = []
    for c0 in range(0, n_comp, chunk_size):
        c1 = min(c0 + chunk_size, n_comp)
        cl = comp_logits[:, c0:c1, :].float()                  # [1, chunk, V] fp32 (only chunk wide)
        lse = torch.logsumexp(cl, dim=-1)                      # [1, chunk]
        tgt = cl.gather(-1, comp_labels[:, c0:c1].unsqueeze(-1)).squeeze(-1)  # [1, chunk]
        parts.append((tgt - lse).sum())
        del cl, lse, tgt
    avg_log_prob = torch.stack(parts).sum() / n_comp
    return avg_log_prob


# ============================================================
# §2a Retention-mask fix: assistant-only cross-entropy
# ============================================================
def build_assistant_labels(input_ids, prompt_len):
    """Return a labels tensor that masks (-100) everything before the assistant
    span, supervising ONLY the assistant response tokens.

    `input_ids` : 1-D LongTensor for the full system+user+assistant sequence.
    `prompt_len`: number of leading tokens that constitute system+user+the
                  generation-prompt header `<|im_start|>assistant\\n` -- i.e. the
                  token count of apply_chat_template([system,user],
                  add_generation_prompt=True). These are masked.
    """
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    return labels


def assert_assistant_mask(labels, input_ids, prompt_len):
    """§2a assertion: #unmasked label tokens == assistant span length.

    Returns the assistant span length so callers can log it.
    """
    total_len = len(input_ids)
    assistant_span_len = total_len - prompt_len
    n_unmasked = int((labels != -100).sum().item())
    assert n_unmasked == assistant_span_len, (
        f"retention mask broken: {n_unmasked} unmasked label tokens != "
        f"assistant span length {assistant_span_len} "
        f"(total={total_len}, prompt_len={prompt_len})")
    return assistant_span_len


def compute_sft_loss(model, input_ids, labels, device, autocast_dtype=torch.float16):
    """§2a retention CE over assistant tokens ONLY.

    HuggingFace CausalLM handles the standard next-token shift and ignores
    label==-100 internally, so passing the masked `labels` yields cross-entropy
    on the assistant span alone.
    """
    x = input_ids.unsqueeze(0).to(device)
    y = labels.unsqueeze(0).to(device)
    use_amp = device.type == "cuda"
    ctx = torch.amp.autocast("cuda", dtype=autocast_dtype) if use_amp \
        else torch.autocast(device_type=device.type, enabled=False)
    with ctx:
        out = model(x, labels=y)
    return out.loss


def combined_loss(model, chosen_ids, rejected_ids, chosen_labels,
                  chosen_start, rejected_start,
                  ref_chosen, ref_rejected, device,
                  beta=DEFAULT_BETA, condition="dpo", sft_weight=0.5,
                  clamp_c=DEFAULT_CLAMP_C, eps=DEFAULT_CDPO_EPS):
    """Combined preference + (assistant-only) SFT-retention loss for one pair.

    Preference log-probs are completion-only (chosen_start / rejected_start mark
    the completion boundary). Retention CE is assistant-only via chosen_labels.
    """
    pi_chosen = compute_log_probs(model, chosen_ids, chosen_start, device)
    pi_rejected = compute_log_probs(model, rejected_ids, rejected_start, device)

    pref, margin = preference_loss(
        pi_chosen, pi_rejected, ref_chosen, ref_rejected,
        beta=beta, condition=condition, clamp_c=clamp_c, eps=eps)

    sft = compute_sft_loss(model, chosen_ids, chosen_labels, device)
    total = pref + sft_weight * sft

    return total, {
        "total": float(total.detach()),
        "pref": float(pref.detach()),
        "sft": float(sft.detach()),
        "margin": margin,
    }
