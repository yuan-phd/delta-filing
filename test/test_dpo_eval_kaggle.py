# ============================================================
# Delta Filing — DPO Evaluation (Kaggle)
# 1. Preference accuracy: does model rank chosen > rejected?
# 2. SFT vs DPO comparison
# Run after DPO training on Kaggle
# ============================================================

SFT_ADAPTER = "/kaggle/input/datasets/yuanmazax/delta-filing-sft-result"
DPO_ADAPTER = "/kaggle/working/adapters/dpo_pytorch"
DPO_DATA = "/kaggle/input/datasets/yuanmazax/delta-filing-dpo/dpo_pytorch.jsonl"
MODEL_ID = "Qwen/Qwen3.5-4B"

import os, json, random, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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
    "When you need data, call the appropriate tool.\n\n"
    "Available tools: search_filings, get_filing_section, diff_filing_sections, "
    "stock_price, company_metrics, company_news, insider_trades, analyst_ratings\n\n"
    'Respond with JSON: {"tool": "name", "arguments": {...}}'
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)


def load_model(adapter_path):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config,
        trust_remote_code=True, device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, base


def compute_log_prob(model, input_ids):
    x = input_ids.unsqueeze(0).to("cuda:0")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(x).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = x[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


def eval_preference(model, test_data):
    """Compute preference accuracy on test set."""
    correct = 0
    for i, ex in enumerate(test_data):
        category = ex.get("category", "analysis")
        sys_prompt = TOOL_SYSTEM_PROMPT if category == "tool_calling" else SYSTEM_PROMPT
        chosen_text = tokenizer.apply_chat_template([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["chosen"]},
        ], tokenize=False)
        rejected_text = tokenizer.apply_chat_template([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["rejected"]},
        ], tokenize=False)
        chosen_ids = torch.tensor(tokenizer.encode(chosen_text, max_length=1024, truncation=True))
        rejected_ids = torch.tensor(tokenizer.encode(rejected_text, max_length=1024, truncation=True))
        if compute_log_prob(model, chosen_ids) > compute_log_prob(model, rejected_ids):
            correct += 1
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(test_data)} done")
    return correct


# --- Load test data ---
with open(DPO_DATA) as f:
    dpo_data = [json.loads(line) for line in f if line.strip()]
random.seed(42)
random.shuffle(dpo_data)
test_data = dpo_data[int(len(dpo_data) * 0.85):]
print(f"Test pairs: {len(test_data)}")

# --- Evaluate DPO ---
print("\n[1] Evaluating DPO model...")
dpo_model, dpo_base = load_model(DPO_ADAPTER)
dpo_correct = eval_preference(dpo_model, test_data)
del dpo_model, dpo_base
torch.cuda.empty_cache()

# --- Evaluate SFT ---
print("\n[2] Evaluating SFT model...")
sft_model, sft_base = load_model(SFT_ADAPTER)
sft_correct = eval_preference(sft_model, test_data)
del sft_model, sft_base
torch.cuda.empty_cache()

# --- Results ---
total = len(test_data)
print(f"\n{'='*60}")
print(f"  PREFERENCE ACCURACY")
print(f"{'='*60}")
print(f"  SFT:  {sft_correct}/{total} ({sft_correct/total:.1%})")
print(f"  DPO:  {dpo_correct}/{total} ({dpo_correct/total:.1%})")
print(f"  Delta: {(dpo_correct-sft_correct)/total:+.1%}")
print(f"{'='*60}")

with open("/kaggle/working/dpo_eval_results.txt", "w") as f:
    f.write(f"Preference Accuracy\n{'='*40}\n")
    f.write(f"SFT:  {sft_correct}/{total} ({sft_correct/total:.1%})\n")
    f.write(f"DPO:  {dpo_correct}/{total} ({dpo_correct/total:.1%})\n")
    f.write(f"Delta: {(dpo_correct-sft_correct)/total:+.1%}\n")
