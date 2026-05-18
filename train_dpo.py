"""
Delta Filing — Hand-written DPO Training (PyTorch)
====================================================
Custom Direct Preference Optimization loss function from scratch.
No TRL DPOTrainer — every step is explicit for interview clarity.

DPO Loss:
  loss = -log(sigmoid(β × (log_π(y_w|x) - log_π(y_l|x)
                            - log_π_ref(y_w|x) + log_π_ref(y_l|x))))

Where:
  π     = policy model (SFT + trainable LoRA)
  π_ref = reference model (SFT, frozen)
  y_w   = chosen (winner) response
  y_l   = rejected (loser) response
  β     = temperature, controls deviation from reference

Usage:
    python train_dpo.py
    python train_dpo.py --beta 0.1 --lr 5e-5 --epochs 2
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ============================================================
#  Chat template (must match SFT training)
# ============================================================

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
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


# ============================================================
#  Dataset
# ============================================================

class DPODataset(Dataset):
    """DPO preference dataset."""

    def __init__(self, data_path, tokenizer, max_length=2048):
        with open(data_path) as f:
            raw = [json.loads(line) for line in f if line.strip()]

        self.pairs = []
        skipped = 0

        for ex in raw:
            category = ex.get("category", "analysis")
            sys_prompt = TOOL_SYSTEM_PROMPT if category == "tool_calling" else SYSTEM_PROMPT

            chosen_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["chosen"]},
            ]
            rejected_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["rejected"]},
            ]

            chosen_text = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

            chosen_tokens = tokenizer.encode(chosen_text, max_length=max_length, truncation=True)
            rejected_tokens = tokenizer.encode(rejected_text, max_length=max_length, truncation=True)

            self.pairs.append({
                "chosen_ids": torch.tensor(chosen_tokens, dtype=torch.long),
                "rejected_ids": torch.tensor(rejected_tokens, dtype=torch.long),
            })

        print(f"  Loaded {len(self.pairs)} DPO pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ============================================================
#  Core: Log probability computation
# ============================================================

def compute_sequence_log_probs(model, input_ids, device):
    """Compute total log probability of a token sequence.

    For a sequence [t0, t1, t2, ..., tn]:
      - Feed the full sequence through the model
      - At position i, the model predicts a distribution over token i+1
      - We look up the log probability of the actual token t_{i+1}
      - Sum all these log probabilities

    Args:
        model: the language model
        input_ids: tensor of shape (seq_len,) — token IDs
        device: 'mps', 'cuda', or 'cpu'

    Returns:
        total_log_prob: scalar tensor
    """
    x = input_ids.unsqueeze(0).to(device)  # (1, seq_len)

    outputs = model(x)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]   # (1, seq_len-1, vocab)
    shift_labels = x[:, 1:]            # (1, seq_len-1)

    # Per-token log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log prob of actual next token at each position
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)  # (1, seq_len-1)

    return token_log_probs.sum()


# ============================================================
#  Core: DPO Loss Function
# ============================================================

def dpo_loss(policy_model, ref_model, chosen_ids, rejected_ids, beta, device):
    """Compute DPO loss for a single preference pair.

    Math:
      chosen_reward  = log_π(chosen)  - log_π_ref(chosen)
      rejected_reward = log_π(rejected) - log_π_ref(rejected)
      loss = -log(sigmoid(β × (chosen_reward - rejected_reward)))

    The loss is minimized when the policy assigns higher relative
    probability to chosen vs rejected, compared to the reference.
    β controls how far the policy can deviate from the reference.
    """
    # Policy model log probs (gets gradients)
    pi_chosen = compute_sequence_log_probs(policy_model, chosen_ids, device)
    pi_rejected = compute_sequence_log_probs(policy_model, rejected_ids, device)

    # Reference model log probs (no gradients)
    with torch.no_grad():
        ref_chosen = compute_sequence_log_probs(ref_model, chosen_ids, device)
        ref_rejected = compute_sequence_log_probs(ref_model, rejected_ids, device)

    # Rewards: how much more the policy likes each response vs the reference
    chosen_reward = pi_chosen - ref_chosen
    rejected_reward = pi_rejected - ref_rejected

    # DPO reward margin
    reward_margin = beta * (chosen_reward - rejected_reward)

    # Loss: -log(sigmoid(margin)) — numerically stable via logsigmoid
    loss = -F.logsigmoid(reward_margin)

    return loss, {
        "loss": loss.item(),
        "reward_margin": reward_margin.item(),
        "chosen_reward": chosen_reward.item(),
        "rejected_reward": rejected_reward.item(),
    }


# ============================================================
#  Training loop
# ============================================================

def train(args):
    print("=" * 60)
    print("  Delta Filing — DPO Training (hand-written PyTorch)")
    print("=" * 60)
    print(f"  β={args.beta}, LR={args.lr}, Epochs={args.epochs}")
    print("=" * 60)

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"\n  Device: {device}")

    # Tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data
    print("\n[2] Loading DPO data...")
    dataset = DPODataset(args.data, tokenizer, args.max_seq_length)
    n = len(dataset)
    split = int(n * 0.85)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    train_idx = indices[:split]
    valid_idx = indices[split:]
    print(f"  Train: {len(train_idx)}, Valid: {len(valid_idx)}")

    # Policy model
    print("\n[3] Loading policy model (SFT + trainable LoRA)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    policy_model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    policy_model.train()
    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy_model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.3f}%)")

    # Reference model
    print("\n[4] Loading reference model (frozen)...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_base, args.sft_adapter, is_trainable=False)
    ref_model.eval()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # Train
    print(f"\n[5] Training...")
    os.makedirs(args.output, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        random.shuffle(train_idx)
        losses, margins = [], []

        for i, idx in enumerate(train_idx):
            pair = dataset[idx]

            optimizer.zero_grad()
            loss, metrics = dpo_loss(
                policy_model, ref_model,
                pair["chosen_ids"], pair["rejected_ids"],
                args.beta, device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            losses.append(metrics["loss"])
            margins.append(metrics["reward_margin"])
            global_step += 1

            if global_step % args.log_every == 0:
                avg_l = sum(losses[-args.log_every:]) / args.log_every
                avg_m = sum(margins[-args.log_every:]) / args.log_every
                print(f"  Step {global_step:4d} | loss={avg_l:.4f} | margin={avg_m:.3f}")

            if global_step % args.eval_every == 0:
                policy_model.eval()
                vl, vm = [], []
                with torch.no_grad():
                    for vi in valid_idx[:30]:
                        vp = dataset[vi]
                        l, m = dpo_loss(
                            policy_model, ref_model,
                            vp["chosen_ids"], vp["rejected_ids"],
                            args.beta, device,
                        )
                        vl.append(m["loss"])
                        vm.append(m["reward_margin"])
                avg_vl = sum(vl) / len(vl)
                is_best = avg_vl < best_val_loss
                if is_best:
                    best_val_loss = avg_vl
                print(f"  Step {global_step:4d} | VAL loss={avg_vl:.4f} | "
                      f"margin={sum(vm)/len(vm):.3f}{'  ★' if is_best else ''}")
                policy_model.train()

            if global_step % args.save_every == 0:
                ckpt = os.path.join(args.output, f"checkpoint-{global_step}")
                policy_model.save_pretrained(ckpt)
                print(f"  Step {global_step:4d} | Saved {ckpt}")

    # Save final
    policy_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n{'='*60}")
    print(f"  Done! Best val loss: {best_val_loss:.4f}")
    print(f"  Saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--sft-adapter", default="./adapters/sft_pytorch")
    parser.add_argument("--data", default="./training_data/dpo_pytorch.jsonl")
    parser.add_argument("--output", default="./adapters/dpo_pytorch")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()
    train(args)
