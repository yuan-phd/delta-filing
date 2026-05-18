"""
Delta Filing — SFT Training (PyTorch, manual loop)
No TRL dependency — direct PyTorch training to avoid TRL bugs on MPS/CPU.
"""

import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=2048):
        with open(path) as f:
            raw = [json.loads(line) for line in f if line.strip()]
        self.examples = []
        skipped = 0
        for ex in raw:
            text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
            tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
            if len(tokens) > max_length:
                skipped += 1
                continue
            self.examples.append(torch.tensor(tokens, dtype=torch.long))
        print(f"  Loaded {len(self.examples)} examples, skipped {skipped}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train(args):
    print("=" * 60)
    print("  Delta Filing — SFT Training (manual PyTorch loop)")
    print("=" * 60)

    device = "cpu"  # CPU is stable; MPS has NaN bugs
    print(f"  Device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify no thinking tokens
    test = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], tokenize=False
    )
    assert "<think>" not in test, "Thinking tokens in template!"

    # Load data
    print("\n  Loading data...")
    train_data = SFTDataset(f"{args.data}/train.jsonl", tokenizer, args.max_seq_length)
    valid_data = SFTDataset(f"{args.data}/valid.jsonl", tokenizer, args.max_seq_length)

    # Load model
    print("\n  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        device_map={"": device}, trust_remote_code=True,
    )

    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # Training
    os.makedirs(args.output, exist_ok=True)
    global_step = 0
    accum_loss = 0
    best_val_loss = float("inf")
    indices = list(range(len(train_data)))

    total_steps = len(indices) * args.epochs // args.grad_accum
    print(f"\n  Epochs: {args.epochs}, Samples: {len(indices)}")
    print(f"  Grad accum: {args.grad_accum}, Total optimizer steps: {total_steps}")
    print(f"  Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(indices)
        epoch_loss = 0
        epoch_tokens = 0

        for i, idx in enumerate(indices):
            input_ids = train_data[idx].unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            accum_loss += outputs.loss.item()
            epoch_tokens += input_ids.shape[1]

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = accum_loss / args.log_every / args.grad_accum
                    print(f"  Epoch {epoch} Step {global_step:4d} | "
                          f"loss={avg:.4f} | tokens={epoch_tokens}")
                    accum_loss = 0

                if global_step % args.eval_every == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vi in range(min(30, len(valid_data))):
                            v_ids = valid_data[vi].unsqueeze(0).to(device)
                            v_out = model(input_ids=v_ids, labels=v_ids)
                            val_losses.append(v_out.loss.item())
                    avg_val = sum(val_losses) / len(val_losses)
                    is_best = avg_val < best_val_loss
                    if is_best:
                        best_val_loss = avg_val
                    print(f"  Epoch {epoch} Step {global_step:4d} | "
                          f"VAL loss={avg_val:.4f}{'  ★' if is_best else ''}")
                    model.train()

                if global_step % args.save_every == 0:
                    ckpt = os.path.join(args.output, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    print(f"  Epoch {epoch} Step {global_step:4d} | Saved {ckpt}")

    # Save final
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n  Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--data", default="./training_data/sft_pytorch")
    parser.add_argument("--output", default="./adapters/sft_pytorch")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()
    train(args)
