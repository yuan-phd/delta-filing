"""
Delta Filing — Unified Test Suite (35 tests, 10 categories)
=============================================================
Tests any adapter(s) on 35 capability tests. Produces comparison table
when multiple adapters are tested.

Usage:
    Cell 1: !pip install ...
    Cell 2: Copy this entire file
    Cell 3: Edit ADAPTERS dict at the top to match your input paths

Categories (35 tests):
    Router           7    Classification + boundary cases
    Tool calling    10    All tools + unseen company + casual phrasing
    Tool parsing     3    AAPL, MSFT, NVDA
    Review           3    Good / Bad / Medium
    Synthesis        2    DIS, NVDA
    Section summary  2    AAPL, NVDA
    Red flag         2    TSLA, META
    Change detection 2    GOOGL, AMZN
    Refusal          2    Investment advice, stock prediction
    Robustness       2    Lowercase, verbose query
"""

import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# CONFIG — EDIT THESE PATHS
# ============================================================
MODEL_ID = "Qwen/Qwen3.5-4B"

# Add/remove adapters as needed. Key = label, Value = path.
ADAPTERS = {
    "SFT":  "/kaggle/input/datasets/yuanmazax/delta-filing-sft-result",
    # "DPO":  "/kaggle/input/datasets/yuanmazax/dpo-ablation/best_dpo_adapter",
    # "IPO":  "/kaggle/input/datasets/yuanmazax/dpo-ablation/adapters/dpo_ipo",
    # "cDPO": "/kaggle/input/datasets/yuanmazax/dpo-ablation/adapters/dpo_cdpo",
    # "GRPO": "/kaggle/working/adapters/grpo",
}

# ============================================================
# PROMPTS & KEYWORDS (shared across all tests)
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
    "1. search_filings(ticker: str, filing_type: str = \"10-K\", count: int = 5)\n"
    "2. get_filing_section(ticker: str, section_id: str, filing_type: str = \"10-K\")\n"
    "3. diff_filing_sections(ticker: str, section_id: str, filing_type: str = \"10-K\")\n"
    "4. stock_price(ticker: str, period: str = \"1mo\")\n"
    "5. company_metrics(ticker: str)\n"
    "6. company_news(ticker: str, days: int = 7)\n"
    "7. insider_trades(ticker: str)\n"
    "8. analyst_ratings(ticker: str)\n\n"
    'To use a tool, respond with a JSON object: '
    '{"tool": "tool_name", "arguments": {"param": "value"}}\n'
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

CATEGORIES = ["Router", "Tool calling", "Tool parsing", "Review", "Synthesis",
              "Section summary", "Red flag", "Change detection", "Refusal", "Robustness"]

# Keyword sets
SECTION_KW = ["Item 1A", "Item 1", "Item 7", "Risk Factors", "MD&A",
              "Management's Discussion", "risk factors"]
JUDGMENT_KW = ["significant", "critical", "notable", "concern", "impact",
               "important", "material", "substantial"]
SEVERITY_KW = ["high", "medium", "low", "severe", "critical", "significant", "moderate"]
SECTION_KW_EX = SECTION_KW + ["10-K", "10-Q", "filing", "annual report", "disclosure", "SEC"]
CHANGE_KW = ["added", "removed", "modified", "changed", "new", "deleted", "updated", "introduced"]
TIME_KW = ["2024", "2025", "year-over-year", "prior year", "previous"]
REFUSAL_KW = ["not", "cannot", "don't", "shouldn't", "unable", "beyond", "scope",
              "investment advice", "not provide", "caveat", "disclaimer", "caution",
              "not recommend", "not in a position", "consult"]


# ============================================================
# SETUP
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Adapters to test: {list(ADAPTERS.keys())}")

# Verify paths
for name, path in ADAPTERS.items():
    ok = os.path.exists(os.path.join(path, "adapter_config.json"))
    print(f"  {name}: {'OK' if ok else 'MISSING!'} — {path}")


# ============================================================
# TEST RUNNER
# ============================================================

def run_35_tests(model, label):
    """Run all 35 tests. Returns (scores_dict, total_pass, total_count)."""

    def gen(msgs, max_tok=500):
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    results = {}
    def rec(cat, name, ok):
        if cat not in results:
            results[cat] = []
        results[cat].append((name, ok))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    # ---- 1. Router (7) ----
    print(f"\n  [{label}] Router (7)")
    for exp, q in [
        ("FILING_ANALYSIS", "What are Apple's risk factors in their latest 10-K?"),
        ("FILING_DIFF", "How did Tesla's risks change from last year?"),
        ("COMPANY_DEEP_DIVE", "Full analysis of NVIDIA - filings, market data, insider trades"),
        ("SIMPLE_QUERY", "List Microsoft's recent filings"),
        ("OUT_OF_SCOPE", "What's the weather in Zurich?"),
        ("OUT_OF_SCOPE", "What does Bitcoin's 10-K say about mining risks?"),
        ("FILING_ANALYSIS", "Tell me about the risks Apple mentioned in their annual report"),
    ]:
        r = gen([{"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                 {"role": "user", "content": q}], 20)
        rec("Router", f"{exp}: {q[:45]}", exp in r)

    # ---- 2. Tool calling (10) ----
    print(f"\n  [{label}] Tool calling (10)")
    for q, tool, tick in [
        ("What are Apple's risk factors?", "get_filing_section", "AAPL"),
        ("How did Microsoft's MD&A change from last year?", "diff_filing_sections", "MSFT"),
        ("List Tesla's recent 10-K filings", "search_filings", "TSLA"),
        ("What are Tesla's risk factors?", "get_filing_section", "TSLA"),
        ("What's the analyst consensus on NVIDIA?", "analyst_ratings", "NVDA"),
        ("Show me recent insider trades at Meta", "insider_trades", "META"),
        ("What's AMD's stock price over the past month?", "stock_price", "AMD"),
        ("Show me recent news about Google", "company_news", "GOOGL"),
        ("What are Palantir's risk factors?", "get_filing_section", "PLTR"),
        ("yo check TSLA insider trades", "insider_trades", "TSLA"),
    ]:
        r = gen([{"role": "system", "content": TOOL_SYSTEM_PROMPT},
                 {"role": "user", "content": q}], 100)
        fl = r.strip().split('\n')[0]
        try:
            p = json.loads(fl)
            ok = p.get("tool") == tool and p.get("arguments", {}).get("ticker") == tick
        except:
            ok = False
        rec("Tool calling", f"{tool}({tick})", ok)

    # ---- 3. Tool parsing (3) ----
    print(f"\n  [{label}] Tool parsing (3)")
    for nm, inp, ck in [
        ("AAPL Risk Factors",
         json.dumps({"ticker": "AAPL", "section": "Risk Factors", "filing_date": "2025-10-31",
                     "content": "International sales account for approximately 60% of net sales. "
                     "Supply chain disruptions could materially affect operations."}),
         {"ticker": ["AAPL", "Apple"], "section": ["Risk Factors", "risk factors", "Item 1A"],
          "date": ["2025", "October"]}),
        ("MSFT MD&A",
         json.dumps({"ticker": "MSFT", "section": "Management's Discussion and Analysis",
                     "filing_date": "2025-07-30",
                     "content": "Revenue increased 15% to $245.1 billion driven by Intelligent Cloud growth of 23%. "
                     "Azure revenue grew 33%."}),
         {"ticker": ["MSFT", "Microsoft"], "section": ["MD&A", "Management", "Discussion"],
          "date": ["2025", "July"]}),
        ("NVDA Financials",
         json.dumps({"ticker": "NVDA", "section": "Financial Highlights", "filing_date": "2025-01-26",
                     "content": "Revenue surged 122% to $130.5 billion. Data Center segment grew 142% to $115.2 billion. "
                     "Gross margin expanded to 74.6%."}),
         {"ticker": ["NVDA", "NVIDIA"], "section": ["Financial", "revenue", "income"],
          "date": ["2025", "January"]}),
    ]:
        r = gen([{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": f"I retrieved this filing section. Please analyze it:\n{inp}"}], 500)
        ok = all(any(x in r for x in ck[k]) for k in ck)
        rec("Tool parsing", nm, ok)

    # ---- 4. Review (3) ----
    print(f"\n  [{label}] Review (3)")
    for nm, inp, exp in [
        ("Good → COMPLETE",
         "Original question: What are Apple's key risk factors?\n\n"
         "Analysis: Apple's 10-K filing (Item 1A) identifies several critical risks. "
         "International sales represent 60% of net revenue ($234.3B in FY2024). "
         "Gross margin of 45.96% faces pressure from component cost increases. "
         "Supply chain concentration in China (estimated 85% of manufacturing). "
         "R&D spending of $29.9B (8% of revenue).",
         "COMPLETE"),
        ("Bad → NEEDS_FOLLOW_UP",
         "Original question: What are Apple's key risk factors?\n\n"
         "Analysis: Apple faces some risks related to competition and market conditions. "
         "The company operates in a competitive industry.",
         "NEEDS_FOLLOW_UP"),
        ("Medium → NEEDS_FOLLOW_UP",
         "Original question: What are Apple's key risk factors?\n\n"
         "Analysis: Apple's 10-K identifies risks in Item 1A including supply chain issues "
         "and competition. Regulatory changes could impact the business.",
         "NEEDS_FOLLOW_UP"),
    ]:
        r = gen([{"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                 {"role": "user", "content": inp}], 100)
        ok = r.startswith("COMPLETE") if exp == "COMPLETE" else "NEEDS_FOLLOW_UP" in r
        rec("Review", nm, ok)

    # ---- 5. Synthesis (2) ----
    print(f"\n  [{label}] Synthesis (2)")
    sd1 = json.dumps({
        "ticker": "DIS", "company": "The Walt Disney Company",
        "metrics": {"market_cap": 188.51e9, "pe_ratio": 15.66, "revenue": 95.72e9, "current_price": 106.30},
        "news": [{"headline": "Disney+ reaches 150M subscribers"}, {"headline": "Theme park revenue up 12% YoY"}],
        "insider_trades": [{"name": "Bob Iger", "type": "Sale", "shares": 50000}],
    }, indent=2)
    r = gen([{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": f"Company: DIS\nData:\n{sd1}\n\nProvide a comprehensive analysis."}], 500)
    ok = ("DIS" in r or "Disney" in r) and any(n in r for n in ["188", "15.66", "95.7", "106", "150M", "150 million", "12%"])
    topics = sum(1 for t in ["market", "revenue", "subscriber", "theme park", "insider", "Iger"] if t.lower() in r.lower())
    rec("Synthesis", "DIS deep dive", ok and topics >= 3)

    sd2 = json.dumps({
        "ticker": "NVDA", "company": "NVIDIA Corporation",
        "metrics": {"market_cap": 3.2e12, "pe_ratio": 55.4, "revenue": 130.5e9, "current_price": 130.50},
        "news": [{"headline": "NVIDIA Blackwell GPUs see unprecedented demand"},
                 {"headline": "Data center revenue grows 142% YoY"}],
        "insider_trades": [{"name": "Jensen Huang", "type": "Sale", "shares": 120000}],
    }, indent=2)
    r = gen([{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": f"Company: NVDA\nData:\n{sd2}\n\nProvide a comprehensive analysis."}], 500)
    ok = ("NVDA" in r or "NVIDIA" in r) and any(n in r for n in ["3.2", "55.4", "130.5", "142%", "Blackwell"])
    topics = sum(1 for t in ["market", "revenue", "data center", "GPU", "insider", "Huang", "Jensen"] if t.lower() in r.lower())
    rec("Synthesis", "NVDA deep dive", ok and topics >= 3)

    # ---- 6. Section summary (2) ----
    print(f"\n  [{label}] Section summary (2)")
    for q, t, n in [
        ("What are the key risk factors disclosed in Apple's most recent 10-K filing?", "AAPL", "Apple"),
        ("Summarize NVIDIA's MD&A section from their latest 10-K.", "NVDA", "NVIDIA"),
    ]:
        r = gen([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}], 500)
        ok = (any(k.lower() in r.lower() for k in SECTION_KW) and (t in r or n in r)
              and any(k in r.lower() for k in JUDGMENT_KW))
        rec("Section summary", t, ok)

    # ---- 7. Red flag (2) ----
    print(f"\n  [{label}] Red flag (2)")
    for q, t in [
        ("What potential risks or concerns can be identified from Tesla's 10-K filing?", "TSLA"),
        ("Identify warning signs in Meta's annual filing disclosures.", "META"),
    ]:
        r = gen([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}], 500)
        ok = (len(r) > 100
              and any(k in r.lower() for k in SEVERITY_KW)
              and any(k.lower() in r.lower() for k in SECTION_KW_EX))
        rec("Red flag", t, ok)

    # ---- 8. Change detection (2) ----
    print(f"\n  [{label}] Change detection (2)")
    for q, t in [
        ("What are the significant changes between Google's 2024 and 2025 10-K risk factor disclosures?", "GOOGL"),
        ("How has Amazon's MD&A section changed from their 2024 to 2025 10-K filing?", "AMZN"),
    ]:
        r = gen([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}], 500)
        ok = (any(k in r.lower() for k in CHANGE_KW)
              and any(k in r for k in TIME_KW)
              and any(k in r.lower() for k in JUDGMENT_KW))
        rec("Change detection", t, ok)

    # ---- 9. Refusal (2) ----
    print(f"\n  [{label}] Refusal (2)")
    r = gen([{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": "Should I buy AAPL stock right now? Give me a clear yes or no."}], 300)
    rec("Refusal", "Investment advice (buy AAPL?)",
        any(k in r.lower() for k in REFUSAL_KW) and len(r) > 30)

    r = gen([{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": "Will Tesla stock go up or down next quarter? Predict the price."}], 300)
    rec("Refusal", "Stock prediction (TSLA price?)",
        any(k in r.lower() for k in REFUSAL_KW) and len(r) > 30)

    # ---- 10. Robustness (2) ----
    print(f"\n  [{label}] Robustness (2)")
    r = gen([{"role": "system", "content": TOOL_SYSTEM_PROMPT},
             {"role": "user", "content": "apple risk factors 10k"}], 100)
    fl = r.strip().split('\n')[0]
    try:
        p = json.loads(fl)
        ok = p.get("tool") == "get_filing_section" and p.get("arguments", {}).get("ticker") == "AAPL"
    except:
        ok = False
    rec("Robustness", "Lowercase/no punctuation → tool call", ok)

    r = gen([{"role": "system", "content": TOOL_SYSTEM_PROMPT},
             {"role": "user", "content":
              "I'm doing research on the semiconductor industry and I'm particularly interested in understanding "
              "what NVIDIA has disclosed about their competitive risks and supply chain vulnerabilities in their "
              "most recent annual filing with the SEC. Could you help me find and analyze the relevant risk "
              "factors section from their 10-K?"}], 100)
    fl = r.strip().split('\n')[0]
    try:
        p = json.loads(fl)
        ok = p.get("tool") == "get_filing_section" and p.get("arguments", {}).get("ticker") == "NVDA"
    except:
        ok = False
    rec("Robustness", "Long verbose query → correct tool call", ok)

    # ---- Compute scores ----
    scores = {}
    total_p, total_c = 0, 0
    for cat in CATEGORIES:
        if cat in results:
            p = sum(1 for _, ok in results[cat] if ok)
            c = len(results[cat])
            scores[cat] = (p, c)
            total_p += p
            total_c += c

    return scores, total_p, total_c


# ============================================================
# RUN ALL ADAPTERS
# ============================================================

all_scores = {}
all_totals = {}

for label, path in ADAPTERS.items():
    cfg = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg):
        print(f"\n  SKIPPING {label}: adapter_config.json not found at {path}")
        continue

    print(f"\n{'=' * 60}")
    print(f"  TESTING: {label}")
    print(f"{'=' * 60}")

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config,
        trust_remote_code=True, device_map={"": 0},
    )
    mdl = PeftModel.from_pretrained(base, path)
    mdl.eval()

    scores, tp, tc = run_35_tests(mdl, label)
    all_scores[label] = scores
    all_totals[label] = (tp, tc)

    # Print individual summary
    print(f"\n  {label}: {tp}/{tc} ({tp * 100 // tc}%)")

    del mdl, base
    torch.cuda.empty_cache()


# ============================================================
# COMPARISON TABLE
# ============================================================

labels = [l for l in ADAPTERS if l in all_scores]

if not labels:
    print("\n  ERROR: No adapters were tested. Check paths above.")
else:
    print("\n\n" + "#" * 70)
    print("  35-TEST COMPARISON")
    print("#" * 70)

    header = f"  {'Category':<20}" + "".join(f"{l:<10}" for l in labels)
    print(header)
    print(f"  {'-' * 20}" + "-" * 10 * len(labels))

    for cat in CATEGORIES:
        row = f"  {cat:<20}"
        for l in labels:
            if cat in all_scores.get(l, {}):
                p, c = all_scores[l][cat]
                row += f"{p}/{c:<8}"
            else:
                row += f"{'—':<10}"
        print(row)

    print(f"  {'-' * 20}" + "-" * 10 * len(labels))
    row = f"  {'TOTAL':<20}"
    for l in labels:
        tp, tc = all_totals[l]
        row += f"{tp}/{tc:<8}"
    print(row)
    print("#" * 70)

    # Best
    best = max(labels, key=lambda l: all_totals[l][0])
    print(f"\n  Best: {best} ({all_totals[best][0]}/{all_totals[best][1]})")

    # Save results
    with open("/kaggle/working/test_35_results.txt", "w") as f:
        f.write("35-Test Results\n" + "=" * 60 + "\n\n")
        for l in labels:
            tp, tc = all_totals[l]
            f.write(f"\n{l}: {tp}/{tc}\n")
            for cat in CATEGORIES:
                if cat in all_scores.get(l, {}):
                    p, c = all_scores[l][cat]
                    f.write(f"  {cat}: {p}/{c}\n")
                    if cat in all_scores[l]:
                        for name, ok in []:  # placeholder
                            pass
        f.write(f"\nBest: {best}\n")

    print("  Results saved to /kaggle/working/test_35_results.txt")
