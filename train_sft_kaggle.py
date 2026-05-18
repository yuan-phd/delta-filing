# ============================================================
# Cell 1: Install (run once, then restart runtime)
# ============================================================
# !pip install -q git+https://github.com/huggingface/transformers.git
# !pip install -q --upgrade peft trl datasets accelerate bitsandbytes
# import transformers; print(f"transformers: {transformers.__version__}")
# # After install, click "Restart runtime", then skip to Cell 2

# ============================================================
# Cell 2: Training
# ============================================================
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

# --- Verify GPU ---
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)")

# --- Config ---
MODEL_ID = "Qwen/Qwen3.5-4B"
DATA_DIR = "/kaggle/input/datasets/yuanmazax/delta-filing"
OUTPUT_DIR = "/kaggle/working/adapters/sft_pytorch"

# --- Chat template without thinking tokens ---
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

# --- Tokenizer ---
print("\n[1] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Verify no thinking tokens
test_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "test"}],
    tokenize=False, add_generation_prompt=True,
)
assert "<think>" not in test_text, f"Thinking tokens found: {test_text}"
print("  Chat template: OK (no thinking tokens)")

# --- Load model with 4-bit quantization (QLoRA) ---
print("\n[2] Loading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
)

# VERIFY quantization is working
memory_gb = model.get_memory_footprint() / 1e9
print(f"  Model memory: {memory_gb:.1f} GB")
assert memory_gb < 5.0, f"Model using {memory_gb:.1f} GB — quantization NOT working!"
print(f"  4-bit quantization: VERIFIED")

# --- LoRA config ---
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

# --- Dataset ---
print("\n[3] Loading dataset...")
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "valid": f"{DATA_DIR}/valid.jsonl",
    },
)
print(f"  Train: {len(dataset['train'])}, Valid: {len(dataset['valid'])}")

# Verify first example
sample = dataset["train"][0]
sample_text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
sample_tokens = tokenizer.encode(sample_text)
print(f"  Sample tokens: {len(sample_tokens)}")
print(f"  Sample: {sample_text[:150]}...")

# --- Training config ---
print("\n[4] Configuring training...")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    max_length=2048,
    max_grad_norm=1.0,
    packing=False,
    report_to="none",
    gradient_checkpointing=True,
)

# --- Trainer ---
# Pass model OBJECT (not string) to ensure QLoRA is used
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    peft_config=lora_config,
    processing_class=tokenizer,
)

# --- Check GPU memory before training ---
for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1e9
    reserved = torch.cuda.memory_reserved(i) / 1e9
    print(f"  GPU {i}: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

# --- Train ---
print("\n[5] Starting training...")
trainer.train()

# --- Save ---
print("\n[6] Saving...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"  Adapter saved to {OUTPUT_DIR}")

# ============================================================
# Cell 3: Quick test (optional)
# ============================================================
# model.eval()
# messages = [
#     {"role": "system", "content": "You are Delta Filing, a financial analyst AI."},
#     {"role": "user", "content": "What are Apple's key risk factors?"},
# ]
# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=300)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ============================================================
# Cell 4: Download adapter
# ============================================================
# !zip -r /kaggle/working/sft_adapter.zip /kaggle/working/adapters/sft_pytorch/
# # Then click "Output" tab on right sidebar to download
