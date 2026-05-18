# ============================================================
# Cell 1: Install
# ============================================================
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q --upgrade peft accelerate bitsandbytes

# ============================================================
# Cell 2: GRPO Training + Testing
# ============================================================
import os, json, random, re, shutil, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================
MODEL_ID = "Qwen/Qwen3.5-4B"
SFT_ADAPTER = "/kaggle/input/datasets/yuanmazax/delta-filing-sft-result"
DPO_DATA = "/kaggle/input/datasets/yuanmazax/delta-filing-dpo/dpo_pytorch.jsonl"
OUTPUT_DIR = "/kaggle/working/adapters/grpo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BETA = 0.5            # KL penalty weight
LR = 1e-5             # Low LR for RL stability
EPOCHS = 1
GROUP_SIZE = 4         # Completions per prompt
MAX_GEN_LEN = 150      # Tokens per generation
MAX_SEQ_LEN = 512      # Max prompt+completion for log prob computation
NUM_PROMPTS = 80       # Number of prompts to use
LOG_EVERY = 5
EVAL_EVERY = 25
SAVE_EVERY = 50

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
ROUTER_SYSTEM_PROMPT = (
    "You are a query router for Delta Filing, a financial analysis system. "
    "Classify the user's query into exactly one category. Respond with ONLY the category name.\n\n"
    "Categories:\n"
    "- FILING_ANALYSIS: Questions about a specific company's filing content\n"
    "- FILING_DIFF: Questions comparing filings across time periods\n"
    "- COMPANY_DEEP_DIVE: Requests for comprehensive company analysis\n"
    "- SIMPLE_QUERY: Simple factual queries about filings\n"
    "- OUT_OF_SCOPE: Questions unrelated to SEC filings"
)
REVIEW_SYSTEM_PROMPT = (
    "You are a quality reviewer for financial analysis. Review the analysis below and respond with either:\n"
    "- COMPLETE if the analysis is thorough with specific numbers, filing section references, and analytical judgment\n"
    "- NEEDS_FOLLOW_UP: <reason> if the analysis lacks depth, specifics, or analytical judgment"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)

print(f"GPU: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# HAND-WRITTEN REWARD FUNCTION
# ============================================================

def compute_reward(question, completion, category):
    """Domain-specific reward function for SEC filing analysis.

    Hand-written — no learned reward model. Scores completions based on
    task-specific criteria that define "good financial analysis":

    Tool calling:  Valid JSON + correct structure + ticker present
    Router:        Valid category + concise response
    Review:        Starts with COMPLETE or NEEDS_FOLLOW_UP
    Analysis:      Filing references + specific numbers + analytical judgment

    Returns: float reward score (higher = better)
    """
    reward = 0.0

    if category == "tool_calling":
        # Tool calls must be valid JSON with correct structure
        first_line = completion.strip().split('\n')[0]
        try:
            parsed = json.loads(first_line)
            reward += 1.0  # Valid JSON
            if "tool" in parsed and "arguments" in parsed:
                reward += 2.0  # Correct structure
                if "ticker" in parsed.get("arguments", {}):
                    reward += 2.0  # Has ticker
            # Penalize extra text after JSON
            if len(completion.strip().split('\n')) == 1:
                reward += 1.0  # Clean, single-line response
        except json.JSONDecodeError:
            reward -= 2.0  # Not valid JSON at all

    elif category == "router":
        valid_routes = ["FILING_ANALYSIS", "FILING_DIFF", "COMPANY_DEEP_DIVE",
                        "SIMPLE_QUERY", "OUT_OF_SCOPE"]
        resp = completion.strip()
        if any(r in resp for r in valid_routes):
            reward += 3.0  # Correct format
            # Bonus for concise response (just the category name)
            if resp in valid_routes:
                reward += 2.0
        else:
            reward -= 1.0

    elif category == "review":
        resp = completion.strip()
        if resp.startswith("COMPLETE"):
            reward += 3.0
        elif "NEEDS_FOLLOW_UP" in resp:
            reward += 3.0
            if ":" in resp:
                reward += 1.0  # Includes reason
        else:
            reward -= 1.0

    else:
        # Analysis tasks: section_summary, red_flag, change_detection, synthesis
        # Criterion 1: Filing section references
        section_refs = ["Item 1A", "Item 1B", "Item 7", "Item 7A", "Item 8",
                        "Risk Factors", "MD&A", "10-K", "10-Q", "SEC filing"]
        section_count = sum(1 for s in section_refs if s.lower() in completion.lower())
        reward += min(section_count * 0.5, 2.0)

        # Criterion 2: Specific numbers (dollar amounts, percentages, etc.)
        numbers = re.findall(
            r'\$[\d,.]+\s*(?:billion|million|trillion)?'
            r'|\d+\.?\d*\s*%'
            r'|\d+\.?\d*\s*(?:billion|million|trillion)',
            completion, re.IGNORECASE
        )
        reward += min(len(numbers) * 0.4, 2.0)

        # Criterion 3: Analytical judgment words
        judgment = ["significant", "critical", "notable", "concern", "material",
                    "substantial", "moderate", "severe", "elevated", "heightened"]
        j_count = sum(1 for j in judgment if j in completion.lower())
        reward += min(j_count * 0.3, 1.5)

        # Criterion 4: Caveat/nuance (shows sophistication)
        caveats = ["however", "although", "caveat", "risk", "uncertainty",
                   "it should be noted", "while", "on the other hand"]
        c_count = sum(1 for c in caveats if c in completion.lower())
        reward += min(c_count * 0.3, 1.0)

        # Criterion 5: Length — reward substance, penalize extremes
        words = len(completion.split())
        if words < 20:
            reward -= 2.0   # Way too short
        elif words < 50:
            reward -= 0.5   # A bit thin
        elif words > 400:
            reward -= 0.5   # Verbose

    return reward


# ============================================================
# LOAD PROMPTS
# ============================================================

print("\n[1] Loading prompts...")
with open(DPO_DATA) as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

# Stratified sample across categories
from collections import defaultdict
by_cat = defaultdict(list)
for ex in raw_data:
    by_cat[ex.get("category", "analysis")].append(ex)

prompts = []
per_cat = max(5, NUM_PROMPTS // len(by_cat))
random.seed(42)
for cat, examples in by_cat.items():
    random.shuffle(examples)
    for ex in examples[:per_cat]:
        sys_prompt = TOOL_SYSTEM_PROMPT if cat == "tool_calling" else SYSTEM_PROMPT
        prompts.append({
            "question": ex["question"],
            "category": cat,
            "system_prompt": sys_prompt,
        })

random.shuffle(prompts)
prompts = prompts[:NUM_PROMPTS]

cat_counts = defaultdict(int)
for p in prompts:
    cat_counts[p["category"]] += 1
print(f"  {len(prompts)} prompts: {dict(cat_counts)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n[2] Loading SFT model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config,
    trust_remote_code=True, device_map={"": 0},
)
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER, is_trainable=True)
model.eval()  # Start in eval mode for generation

print(f"  Memory: {base_model.get_memory_footprint() / 1e9:.1f} GB")


# ============================================================
# PHASE 1: GENERATE COMPLETIONS + SCORE + COMPUTE REFERENCE
# ============================================================

def generate_completion(model, prompt_text):
    """Generate one completion with sampling."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_GEN_LEN,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def compute_log_probs(model, input_ids):
    """Per-token average log probability (length-normalized)."""
    x = input_ids.unsqueeze(0).to("cuda:0")
    seq_len = x.shape[1]
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(x)
        logits = out.logits
    sl = logits[:, :-1, :].float()
    labels = x[:, 1:]
    lp = F.log_softmax(sl, dim=-1)
    tlp = torch.gather(lp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    avg = tlp.sum() / (seq_len - 1)
    del logits, sl, lp, tlp, out
    return avg


print(f"\n[3] Generating {len(prompts) * GROUP_SIZE} completions ({GROUP_SIZE} per prompt)...")
print(f"  This will take ~30-40 minutes...")

training_data = []  # List of {prompt_ids, completion_ids, full_ids, reward, ref_log_prob, group_idx}

for pi, prompt in enumerate(prompts):
    # Build prompt text
    prompt_text = tokenizer.apply_chat_template([
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": prompt["question"]},
    ], tokenize=False, add_generation_prompt=True)

    prompt_ids = tokenizer.encode(prompt_text)

    group_completions = []

    for g in range(GROUP_SIZE):
        # Generate
        completion_text = generate_completion(model, prompt_text)

        # Score with reward function
        reward = compute_reward(prompt["question"], completion_text, prompt["category"])

        # Tokenize full sequence (prompt + completion)
        full_text = tokenizer.apply_chat_template([
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["question"]},
            {"role": "assistant", "content": completion_text},
        ], tokenize=False)
        full_ids = tokenizer.encode(full_text, max_length=MAX_SEQ_LEN, truncation=True)

        # Compute reference log prob (model is still SFT at this point)
        full_tensor = torch.tensor(full_ids, dtype=torch.long)
        with torch.no_grad():
            ref_lp = compute_log_probs(model, full_tensor).item()

        group_completions.append({
            "full_ids": full_tensor,
            "reward": reward,
            "ref_log_prob": ref_lp,
            "prompt_len": len(prompt_ids),
            "group_idx": pi,
        })

    # Compute group advantages: A_i = (r_i - mean) / (std + eps)
    rewards = [c["reward"] for c in group_completions]
    mean_r = sum(rewards) / len(rewards)
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5

    for c in group_completions:
        if std_r > 1e-6:
            c["advantage"] = (c["reward"] - mean_r) / (std_r + 1e-8)
        else:
            c["advantage"] = 0.0  # All same reward, no signal

    training_data.extend(group_completions)

    if (pi + 1) % 10 == 0:
        avg_reward = sum(c["reward"] for c in training_data[-GROUP_SIZE*10:]) / (GROUP_SIZE * 10)
        print(f"  {pi+1}/{len(prompts)} prompts | avg reward: {avg_reward:.2f}")
        torch.cuda.empty_cache()

# Summary
all_rewards = [c["reward"] for c in training_data]
all_advantages = [c["advantage"] for c in training_data]
print(f"\n  Generation complete!")
print(f"  Total samples: {len(training_data)}")
print(f"  Reward: mean={sum(all_rewards)/len(all_rewards):.2f}, "
      f"min={min(all_rewards):.2f}, max={max(all_rewards):.2f}")
print(f"  Advantage: mean={sum(all_advantages)/len(all_advantages):.2f}, "
      f"std={sum(a**2 for a in all_advantages)/len(all_advantages)**0.5:.2f}")

# Split train/val
random.shuffle(training_data)
val_size = max(20, len(training_data) // 10)
train_data = training_data[:-val_size]
val_data = training_data[-val_size:]
print(f"  Train: {len(train_data)}, Val: {len(val_data)}")


# ============================================================
# PHASE 2: GRPO TRAINING
# ============================================================

def grpo_loss(model, full_ids, ref_log_prob, advantage, beta):
    """GRPO loss for one completion.

    loss = -advantage × policy_log_prob + β × KL(policy || reference)

    When advantage > 0 (good completion):
      Pushes policy_log_prob UP (makes good completions more likely)
    When advantage < 0 (bad completion):
      Pushes policy_log_prob DOWN (makes bad completions less likely)

    KL term prevents policy from diverging too far from SFT reference.
    """
    # Policy log prob (with gradients)
    pi_log_prob = compute_log_probs(model, full_ids)

    # Policy gradient: maximize log prob of good completions
    pg_loss = -advantage * pi_log_prob

    # KL penalty: keep policy close to reference
    kl = pi_log_prob - ref_log_prob  # Per-token average KL
    kl_loss = beta * kl

    total = pg_loss + kl_loss

    return total, {
        "total": total.item(),
        "pg": pg_loss.item(),
        "kl": kl.item(),
        "pi_lp": pi_log_prob.item(),
        "advantage": advantage,
    }


print(f"\n[4] Starting GRPO training...")
print(f"  β={BETA}, LR={LR}, samples={len(train_data)}")

# Enable training
model.gradient_checkpointing_enable()
model.train()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.01,
)

step = 0
best_val = float("inf")
pg_losses, kl_values = [], []
indices = list(range(len(train_data)))

for epoch in range(1, EPOCHS + 1):
    random.shuffle(indices)

    for i, idx in enumerate(indices):
        sample = train_data[idx]

        # Skip zero-advantage samples (no learning signal)
        if abs(sample["advantage"]) < 1e-6:
            continue

        optimizer.zero_grad()
        loss, metrics = grpo_loss(
            model, sample["full_ids"],
            sample["ref_log_prob"], sample["advantage"],
            BETA,
        )

        if torch.isnan(loss):
            del loss
            torch.cuda.empty_cache()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        del loss
        torch.cuda.empty_cache()

        pg_losses.append(metrics["pg"])
        kl_values.append(metrics["kl"])
        step += 1

        if step % LOG_EVERY == 0:
            avg_pg = sum(pg_losses[-LOG_EVERY:]) / LOG_EVERY
            avg_kl = sum(kl_values[-LOG_EVERY:]) / LOG_EVERY
            print(f"  Step {step:4d} | pg={avg_pg:.4f} | kl={avg_kl:.4f}")

        if step % EVAL_EVERY == 0:
            model.eval()
            vl = []
            with torch.no_grad():
                for vi in range(min(20, len(val_data))):
                    vs = val_data[vi]
                    if abs(vs["advantage"]) < 1e-6:
                        continue
                    l, m = grpo_loss(
                        model, vs["full_ids"],
                        vs["ref_log_prob"], vs["advantage"],
                        BETA,
                    )
                    if not torch.isnan(l):
                        vl.append(m["total"])
            if vl:
                avg_vl = sum(vl) / len(vl)
                is_best = avg_vl < best_val
                if is_best:
                    best_val = avg_vl
                    model.save_pretrained(OUTPUT_DIR)
                    tokenizer.save_pretrained(OUTPUT_DIR)
                print(f"  Step {step:4d} | VAL loss={avg_vl:.4f}{'  ★ saved' if is_best else ''}")
            model.train()

        if step % SAVE_EVERY == 0:
            ckpt = os.path.join(OUTPUT_DIR, f"checkpoint-{step}")
            model.save_pretrained(ckpt)

# Final save
if best_val == float("inf"):
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n  GRPO done: {step} steps, best val={best_val:.4f}")


# ============================================================
# PHASE 3: TESTING (35 tests)
# ============================================================

print("\n\n" + "#" * 60)
print("  TESTING GRPO MODEL (35 tests)")
print("#" * 60)

model.eval()

section_kw = ["Item 1A", "Item 1", "Item 7", "Risk Factors", "MD&A", "Management's Discussion", "risk factors"]
judgment_kw = ["significant", "critical", "notable", "concern", "impact", "important", "material", "substantial"]
severity_kw = ["high", "medium", "low", "severe", "critical", "significant", "moderate"]
section_kw_ex = section_kw + ["10-K", "10-Q", "filing", "annual report", "disclosure", "SEC"]
change_kw = ["added", "removed", "modified", "changed", "new", "deleted", "updated", "introduced"]
time_kw = ["2024", "2025", "year-over-year", "prior year", "previous"]
refusal_kw = ["not", "cannot", "don't", "shouldn't", "unable", "beyond", "scope",
              "investment advice", "not provide", "caveat", "disclaimer", "caution",
              "not recommend", "not in a position", "consult"]


def gen(msgs, max_tok=500):
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


results = {}
def rec(cat, name, ok):
    if cat not in results: results[cat] = []
    results[cat].append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

# Router (7)
print("\n  Router")
for exp, q in [("FILING_ANALYSIS","What are Apple's risk factors in their latest 10-K?"),
                ("FILING_DIFF","How did Tesla's risks change from last year?"),
                ("COMPANY_DEEP_DIVE","Full analysis of NVIDIA - filings, market data, insider trades"),
                ("SIMPLE_QUERY","List Microsoft's recent filings"),
                ("OUT_OF_SCOPE","What's the weather in Zurich?"),
                ("OUT_OF_SCOPE","What does Bitcoin's 10-K say about mining risks?"),
                ("FILING_ANALYSIS","Tell me about the risks Apple mentioned in their annual report")]:
    r = gen([{"role":"system","content":ROUTER_SYSTEM_PROMPT},{"role":"user","content":q}], 20)
    rec("Router", f"{exp}: {q[:45]}", exp in r)

# Tool calling (10)
print("\n  Tool calling")
for q, tool, tick in [("What are Apple's risk factors?","get_filing_section","AAPL"),
                       ("How did Microsoft's MD&A change from last year?","diff_filing_sections","MSFT"),
                       ("List Tesla's recent 10-K filings","search_filings","TSLA"),
                       ("What are Tesla's risk factors?","get_filing_section","TSLA"),
                       ("What's the analyst consensus on NVIDIA?","analyst_ratings","NVDA"),
                       ("Show me recent insider trades at Meta","insider_trades","META"),
                       ("What's AMD's stock price over the past month?","stock_price","AMD"),
                       ("Show me recent news about Google","company_news","GOOGL"),
                       ("What are Palantir's risk factors?","get_filing_section","PLTR"),
                       ("yo check TSLA insider trades","insider_trades","TSLA")]:
    r = gen([{"role":"system","content":TOOL_SYSTEM_PROMPT},{"role":"user","content":q}], 100)
    fl = r.strip().split('\n')[0]
    try:
        p = json.loads(fl)
        ok = p.get("tool")==tool and p.get("arguments",{}).get("ticker")==tick
    except: ok = False
    rec("Tool calling", f"{tool}({tick})", ok)

# Tool parsing (3)
print("\n  Tool parsing")
for nm, inp, ck in [
    ("AAPL", json.dumps({"ticker":"AAPL","section":"Risk Factors","filing_date":"2025-10-31",
                          "content":"International sales ~60%. Supply chain risks."}),
     {"ticker":["AAPL","Apple"],"section":["Risk","Item 1A"],"date":["2025","October"]}),
    ("MSFT", json.dumps({"ticker":"MSFT","section":"MD&A","filing_date":"2025-07-30",
                          "content":"Revenue +15% to $245.1B. Azure +33%."}),
     {"ticker":["MSFT","Microsoft"],"section":["MD&A","Management"],"date":["2025","July"]}),
    ("NVDA", json.dumps({"ticker":"NVDA","section":"Financial Highlights","filing_date":"2025-01-26",
                          "content":"Revenue surged 122% to $130.5 billion. Data Center grew 142%."}),
     {"ticker":["NVDA","NVIDIA"],"section":["Financial","revenue","income"],"date":["2025","January"]})]:
    r = gen([{"role":"system","content":SYSTEM_PROMPT},
              {"role":"user","content":f"I retrieved this filing section. Please analyze it:\n{inp}"}], 500)
    ok = all(any(x in r for x in ck[k]) for k in ck)
    rec("Tool parsing", nm, ok)

# Review (3)
print("\n  Review")
for nm, inp, exp in [
    ("Good","Original question: Apple risk factors?\n\nAnalysis: Apple's 10-K (Item 1A) identifies critical risks. International sales 60% ($234.3B). Gross margin 45.96%. Supply chain China 85%. R&D $29.9B.","COMPLETE"),
    ("Bad","Original question: Apple risk factors?\n\nAnalysis: Apple faces some risks. Competitive industry.","NEEDS_FOLLOW_UP"),
    ("Medium","Original question: Apple risk factors?\n\nAnalysis: Apple's 10-K Item 1A mentions supply chain and competition.","NEEDS_FOLLOW_UP")]:
    r = gen([{"role":"system","content":REVIEW_SYSTEM_PROMPT},{"role":"user","content":inp}], 100)
    ok = r.startswith("COMPLETE") if exp=="COMPLETE" else "NEEDS_FOLLOW_UP" in r
    rec("Review", nm, ok)

# Synthesis (2)
print("\n  Synthesis")
sd1 = json.dumps({"ticker":"DIS","metrics":{"market_cap":188.51e9,"pe_ratio":15.66},
                   "news":[{"headline":"Disney+ 150M subs"},{"headline":"Theme park +12% YoY"}],
                   "insider_trades":[{"name":"Bob Iger","type":"Sale","shares":50000}]}, indent=2)
r = gen([{"role":"system","content":SYSTEM_PROMPT},
         {"role":"user","content":f"Company: DIS\nData:\n{sd1}\n\nProvide a comprehensive analysis."}], 500)
ok = ("DIS" in r or "Disney" in r) and any(n in r for n in ["188","15.66","150M","150 million","12%"])
topics1 = sum(1 for t in ["market","revenue","subscriber","theme park","insider","Iger"] if t.lower() in r.lower())
rec("Synthesis", "DIS", ok and topics1 >= 3)

sd2 = json.dumps({"ticker":"NVDA","metrics":{"market_cap":3.2e12,"pe_ratio":55.4,"revenue":130.5e9},
                   "news":[{"headline":"Blackwell GPUs unprecedented demand"},{"headline":"Data center +142% YoY"}],
                   "insider_trades":[{"name":"Jensen Huang","type":"Sale","shares":120000}]}, indent=2)
r = gen([{"role":"system","content":SYSTEM_PROMPT},
         {"role":"user","content":f"Company: NVDA\nData:\n{sd2}\n\nProvide a comprehensive analysis."}], 500)
ok = ("NVDA" in r or "NVIDIA" in r) and any(n in r for n in ["3.2","55.4","130.5","142%","Blackwell"])
topics2 = sum(1 for t in ["market","revenue","data center","GPU","insider","Huang"] if t.lower() in r.lower())
rec("Synthesis", "NVDA", ok and topics2 >= 3)

# Section summary (2)
print("\n  Section summary")
for q, t, n in [("What are the key risk factors in Apple's most recent 10-K?","AAPL","Apple"),
                 ("Summarize NVIDIA's MD&A from their latest 10-K.","NVDA","NVIDIA")]:
    r = gen([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}], 500)
    ok = (any(k.lower() in r.lower() for k in section_kw) and (t in r or n in r)
          and any(k in r.lower() for k in judgment_kw))
    rec("Section summary", t, ok)

# Red flag (2)
print("\n  Red flag")
for q, t in [("What risks or concerns in Tesla's 10-K filing?","TSLA"),
              ("Identify warning signs in Meta's annual filing.","META")]:
    r = gen([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}], 500)
    ok = len(r)>100 and any(k in r.lower() for k in severity_kw) and any(k.lower() in r.lower() for k in section_kw_ex)
    rec("Red flag", t, ok)

# Change detection (2)
print("\n  Change detection")
for q, t in [("Changes between Google's 2024 and 2025 10-K risk factors?","GOOGL"),
              ("How has Amazon's MD&A changed from 2024 to 2025 10-K?","AMZN")]:
    r = gen([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}], 500)
    ok = (any(k in r.lower() for k in change_kw) and any(k in r for k in time_kw)
          and any(k in r.lower() for k in judgment_kw))
    rec("Change detection", t, ok)

# Refusal (2)
print("\n  Refusal")
r = gen([{"role":"system","content":SYSTEM_PROMPT},
         {"role":"user","content":"Should I buy AAPL stock right now? Give me a clear yes or no."}], 300)
rec("Refusal", "Investment advice", any(k in r.lower() for k in refusal_kw) and len(r) > 30)

r = gen([{"role":"system","content":SYSTEM_PROMPT},
         {"role":"user","content":"Will Tesla stock go up or down next quarter? Predict the price."}], 300)
rec("Refusal", "Stock prediction", any(k in r.lower() for k in refusal_kw) and len(r) > 30)

# Robustness (2)
print("\n  Robustness")
r = gen([{"role":"system","content":TOOL_SYSTEM_PROMPT},
         {"role":"user","content":"apple risk factors 10k"}], 100)
fl = r.strip().split('\n')[0]
try:
    p = json.loads(fl)
    ok = p.get("tool")=="get_filing_section" and p.get("arguments",{}).get("ticker")=="AAPL"
except: ok = False
rec("Robustness", "Lowercase query", ok)

r = gen([{"role":"system","content":TOOL_SYSTEM_PROMPT},
         {"role":"user","content":"I'm researching semiconductor risks and want to understand what NVIDIA disclosed about competitive risks and supply chain vulnerabilities in their most recent 10-K annual filing with the SEC."}], 100)
fl = r.strip().split('\n')[0]
try:
    p = json.loads(fl)
    ok = p.get("tool")=="get_filing_section" and p.get("arguments",{}).get("ticker")=="NVDA"
except: ok = False
rec("Robustness", "Long verbose query", ok)


# ============================================================
# RESULTS
# ============================================================

CATEGORIES = ["Router","Tool calling","Tool parsing","Review","Synthesis",
              "Section summary","Red flag","Change detection","Refusal","Robustness"]

print("\n" + "#" * 60)
print("  GRPO TEST RESULTS (35 tests)")
print("#" * 60)
print(f"  {'Category':<20} {'Pass':<10} {'Rate':<6}")
print(f"  {'-'*20} {'-'*10} {'-'*6}")

total_p, total_c = 0, 0
for cat in CATEGORIES:
    if cat in results:
        p = sum(1 for _, ok in results[cat] if ok)
        c = len(results[cat])
        total_p += p; total_c += c
        print(f"  {cat:<20} {p}/{c:<8} {p*100//c:>4}%")

print(f"  {'-'*20} {'-'*10} {'-'*6}")
print(f"  {'TOTAL':<20} {total_p}/{total_c:<8} {total_p*100//total_c:>4}%")
print("#" * 60)

# Save
shutil.copytree(OUTPUT_DIR, "/kaggle/working/grpo_adapter", dirs_exist_ok=True)

with open("/kaggle/working/grpo_results.txt", "w") as f:
    f.write(f"GRPO Results\n{'='*40}\n")
    f.write(f"Config: β={BETA}, LR={LR}, group_size={GROUP_SIZE}, prompts={NUM_PROMPTS}\n\n")
    for cat in CATEGORIES:
        if cat in results:
            for name, ok in results[cat]:
                f.write(f"[{'PASS' if ok else 'FAIL'}] {cat}: {name}\n")
    f.write(f"\nTotal: {total_p}/{total_c}\n")

print(f"\n  Adapter: /kaggle/working/grpo_adapter/")
print(f"  Results: /kaggle/working/grpo_results.txt")
