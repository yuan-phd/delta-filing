# ============================================================
# Delta Filing — Model Test Script (Kaggle)
# All 8 categories, 23 test cases, all fixes included
# ============================================================

import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Config: CHANGE THIS to test different adapters ---
MODEL_ID = "Qwen/Qwen3.5-4B"
ADAPTER_PATH = "/kaggle/working/adapters/dpo_pytorch"  # DPO adapter
# ADAPTER_PATH = "/kaggle/input/datasets/yuanmazax/delta-filing-sft-result"  # SFT adapter

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

# --- Load model ---
print(f"Loading model with adapter: {ADAPTER_PATH}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config,
    trust_remote_code=True, device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model memory: {base_model.get_memory_footprint() / 1e9:.1f} GB")


def generate(messages, max_new_tokens=500):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ============================================================
# Test infrastructure
# ============================================================
results = {}


def record(category, name, passed):
    if category not in results:
        results[category] = []
    results[category].append((name, passed))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")


# ============================================================
# 1. Router (5 cases)
# ============================================================
print("\n" + "=" * 60)
print("  ROUTER TESTS")
print("=" * 60)

router_tests = [
    ("FILING_ANALYSIS", "What are Apple's risk factors in their latest 10-K?"),
    ("FILING_DIFF", "How did Tesla's risks change from last year?"),
    ("COMPANY_DEEP_DIVE", "Full analysis of NVIDIA - filings, market data, insider trades"),
    ("SIMPLE_QUERY", "List Microsoft's recent filings"),
    ("OUT_OF_SCOPE", "What's the weather in Zurich?"),
]
for expected, query in router_tests:
    resp = generate([
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ], max_new_tokens=20)
    passed = expected in resp
    record("Router", f"{expected}: {query[:40]}...", passed)
    if not passed:
        print(f"    Expected: {expected}, Got: {resp[:80]}")

# ============================================================
# 2. Tool calling (6 cases) — parse first line only
# ============================================================
print("\n" + "=" * 60)
print("  TOOL CALLING TESTS")
print("=" * 60)

tool_tests = [
    ("What are Apple's risk factors?", "get_filing_section", "AAPL"),
    ("How did Microsoft's MD&A change from last year?", "diff_filing_sections", "MSFT"),
    ("List Tesla's recent 10-K filings", "search_filings", "TSLA"),
    ("What are Tesla's risk factors?", "get_filing_section", "TSLA"),
    ("What's the analyst consensus on NVIDIA?", "analyst_ratings", "NVDA"),
    ("Show me recent insider trades at Meta", "insider_trades", "META"),
]
for query, expected_tool, expected_ticker in tool_tests:
    resp = generate([
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ], max_new_tokens=100)
    # Fix: parse first line only (model may generate trailing text)
    first_line = resp.strip().split('\n')[0]
    try:
        parsed = json.loads(first_line)
        tool_ok = parsed.get("tool") == expected_tool
        ticker_ok = parsed.get("arguments", {}).get("ticker") == expected_ticker
        passed = tool_ok and ticker_ok
    except json.JSONDecodeError:
        passed = False
    record("Tool calling", f"{expected_tool}({expected_ticker})", passed)
    if not passed:
        print(f"    First line: {first_line[:120]}")

# ============================================================
# 3. Tool result parsing (2 cases)
# ============================================================
print("\n" + "=" * 60)
print("  TOOL RESULT PARSING TESTS")
print("=" * 60)

tool_parse_tests = [
    {
        "name": "AAPL Risk Factors",
        "input": json.dumps({
            "tool": "get_filing_section", "ticker": "AAPL",
            "section": "Risk Factors", "filing_date": "2025-10-31",
            "content": "The Company's business can be impacted by global conditions... "
            "International sales account for approximately 60% of net sales. "
            "Supply chain disruptions could materially affect operations."
        }),
        "checks": {
            "ticker": ["AAPL", "Apple"],
            "section": ["Risk Factors", "risk factors", "Item 1A"],
            "date": ["2025", "October"],
        },
    },
    {
        "name": "MSFT MD&A",
        "input": json.dumps({
            "tool": "get_filing_section", "ticker": "MSFT",
            "section": "Management's Discussion and Analysis", "filing_date": "2025-07-30",
            "content": "Revenue increased 15% to $245.1 billion driven by Intelligent Cloud growth of 23%. "
            "Azure revenue grew 33%. Operating income increased 24% to $109.4 billion."
        }),
        "checks": {
            "ticker": ["MSFT", "Microsoft"],
            "section": ["MD&A", "Management", "Discussion"],
            "date": ["2025", "July"],
        },
    },
]
for test in tool_parse_tests:
    resp = generate([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"I retrieved this filing section using get_filing_section. Please analyze it:\n{test['input']}"},
    ], max_new_tokens=500)
    ticker_ok = any(t in resp for t in test["checks"]["ticker"])
    section_ok = any(s in resp for s in test["checks"]["section"])
    date_ok = any(d in resp for d in test["checks"]["date"])
    passed = ticker_ok and section_ok and date_ok
    record("Tool parsing", test["name"], passed)
    if not passed:
        print(f"    ticker={ticker_ok}, section={section_ok}, date={date_ok}")
        print(f"    Response: {resp[:150]}")

# ============================================================
# 4. Review (3 cases)
# ============================================================
print("\n" + "=" * 60)
print("  REVIEW TESTS")
print("=" * 60)

review_tests = [
    {
        "name": "Good analysis → COMPLETE",
        "input": (
            "Original question: What are Apple's key risk factors?\n\n"
            "Analysis: Apple's 10-K filing (Item 1A) identifies several critical risks. "
            "International sales represent 60% of net revenue ($234.3B in FY2024), exposing the company to "
            "currency fluctuations and geopolitical tensions. The company's gross margin of 45.96% faces pressure "
            "from component cost increases. Supply chain concentration in China (estimated 85% of manufacturing) "
            "creates significant operational risk. Regulatory risks include the EU Digital Markets Act and "
            "ongoing App Store antitrust litigation. R&D spending of $29.9B (8% of revenue) must maintain "
            "competitive positioning against AI-focused competitors."
        ),
        "expected": "COMPLETE",
    },
    {
        "name": "Bad analysis → NEEDS_FOLLOW_UP",
        "input": (
            "Original question: What are Apple's key risk factors?\n\n"
            "Analysis: Apple faces some risks related to competition and market conditions. "
            "The company operates in a competitive industry and needs to continue innovating. "
            "There are also some regulatory concerns."
        ),
        "expected": "NEEDS_FOLLOW_UP",
    },
    {
        "name": "Medium analysis → NEEDS_FOLLOW_UP",
        "input": (
            "Original question: What are Apple's key risk factors?\n\n"
            "Analysis: Apple's 10-K identifies risks in Item 1A including supply chain issues "
            "and competition. The company has significant international operations. "
            "Regulatory changes could impact the business."
        ),
        "expected": "NEEDS_FOLLOW_UP",
    },
]
for test in review_tests:
    resp = generate([
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": test["input"]},
    ], max_new_tokens=100)
    if test["expected"] == "COMPLETE":
        passed = resp.startswith("COMPLETE")
    else:
        passed = "NEEDS_FOLLOW_UP" in resp
    record("Review", test["name"], passed)
    if not passed:
        print(f"    Expected: {test['expected']}, Got: {resp[:80]}")

# ============================================================
# 5. Synthesis (1 case)
# ============================================================
print("\n" + "=" * 60)
print("  SYNTHESIS TEST")
print("=" * 60)

synthesis_data = json.dumps({
    "ticker": "DIS", "company": "The Walt Disney Company",
    "metrics": {"market_cap": 188.51e9, "pe_ratio": 15.66, "revenue": 95.72e9,
                "profit_margin": 0.128, "current_price": 106.30},
    "news": [{"headline": "Disney+ reaches 150M subscribers", "date": "2025-03-15"},
             {"headline": "Theme park revenue up 12% YoY", "date": "2025-03-10"}],
    "insider_trades": [{"name": "Bob Iger", "type": "Sale", "shares": 50000, "date": "2025-03-01"}],
}, indent=2)

resp = generate([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Company: DIS\nData:\n{synthesis_data}\n\nProvide a comprehensive analysis."},
], max_new_tokens=500)

ticker_ok = "DIS" in resp or "Disney" in resp
numbers_ok = any(n in resp for n in ["188", "15.66", "95.7", "106", "150M", "150 million", "12%"])
topics = sum(1 for t in ["market", "revenue", "subscriber", "theme park", "insider", "Iger"]
             if t.lower() in resp.lower())
passed = ticker_ok and numbers_ok and topics >= 3
record("Synthesis", "DIS deep dive", passed)
if not passed:
    print(f"    ticker={ticker_ok}, numbers={numbers_ok}, topics={topics}")

# ============================================================
# 6. Section summary (2 cases)
# ============================================================
print("\n" + "=" * 60)
print("  SECTION SUMMARY TESTS")
print("=" * 60)

summary_tests = [
    ("What are the key risk factors disclosed in Apple's most recent 10-K filing?", "AAPL", "Apple"),
    ("Summarize NVIDIA's MD&A section from their latest 10-K.", "NVDA", "NVIDIA"),
]
section_keywords = ["Item 1A", "Item 1", "Item 7", "Risk Factors", "MD&A",
                    "Management's Discussion", "risk factors"]
judgment_keywords = ["significant", "critical", "notable", "concern", "impact",
                     "important", "material", "substantial"]

for query, ticker, name in summary_tests:
    resp = generate([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ], max_new_tokens=500)
    section_ok = any(k.lower() in resp.lower() for k in section_keywords)
    company_ok = ticker in resp or name in resp
    judgment_ok = any(k in resp.lower() for k in judgment_keywords)
    passed = section_ok and company_ok and judgment_ok
    record("Section summary", f"{ticker}", passed)
    if not passed:
        print(f"    section={section_ok}, company={company_ok}, judgment={judgment_ok}")

# ============================================================
# 7. Red flag (2 cases) — expanded section keywords
# ============================================================
print("\n" + "=" * 60)
print("  RED FLAG TESTS")
print("=" * 60)

red_flag_tests = [
    ("What potential risks or concerns can be identified from Tesla's 10-K filing?", "TSLA"),
    ("Identify warning signs in Meta's annual filing disclosures.", "META"),
]
severity_keywords = ["high", "medium", "low", "severe", "critical", "significant", "moderate"]
section_keywords_expanded = ["Item 1A", "Item 1", "Item 7", "Risk Factors", "risk factors",
                              "MD&A", "Management", "10-K", "10-Q", "filing", "annual report",
                              "disclosure", "SEC"]

for query, ticker in red_flag_tests:
    resp = generate([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ], max_new_tokens=500)
    risks_ok = len(resp) > 100
    severity_ok = any(k in resp.lower() for k in severity_keywords)
    section_ok = any(k.lower() in resp.lower() for k in section_keywords_expanded)
    passed = risks_ok and severity_ok and section_ok
    record("Red flag", ticker, passed)
    if not passed:
        print(f"    risks={risks_ok}, severity={severity_ok}, section={section_ok}")

# ============================================================
# 8. Change detection (2 cases)
# ============================================================
print("\n" + "=" * 60)
print("  CHANGE DETECTION TESTS")
print("=" * 60)

change_tests = [
    ("What are the significant changes between Google's 2024 and 2025 10-K risk factor disclosures?", "GOOGL"),
    ("How has Amazon's MD&A section changed from their 2024 to 2025 10-K filing?", "AMZN"),
]
change_keywords = ["added", "removed", "modified", "changed", "new", "deleted", "updated", "introduced"]
time_keywords = ["2024", "2025", "year-over-year", "prior year", "previous"]

for query, ticker in change_tests:
    resp = generate([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ], max_new_tokens=500)
    change_ok = any(k in resp.lower() for k in change_keywords)
    time_ok = any(k in resp for k in time_keywords)
    judgment_ok = any(k in resp.lower() for k in judgment_keywords)
    passed = change_ok and time_ok and judgment_ok
    record("Change detection", ticker, passed)
    if not passed:
        print(f"    change={change_ok}, time={time_ok}, judgment={judgment_ok}")

# ============================================================
# Summary
# ============================================================
print("\n" + "#" * 60)
print(f"  RESULTS — {ADAPTER_PATH.split('/')[-1]}")
print("#" * 60)
print(f"  {'Category':<20} {'Pass':<10} {'Rate':<6}")
print(f"  {'-'*20} {'-'*10} {'-'*6}")

total_pass = 0
total_count = 0
for cat in ["Router", "Tool calling", "Tool parsing", "Review",
            "Synthesis", "Section summary", "Red flag", "Change detection"]:
    if cat in results:
        passed = sum(1 for _, p in results[cat] if p)
        count = len(results[cat])
        total_pass += passed
        total_count += count
        pct = passed * 100 // count if count > 0 else 0
        print(f"  {cat:<20} {passed}/{count:<8} {pct:>4}%")

print(f"  {'-'*20} {'-'*10} {'-'*6}")
print(f"  {'TOTAL':<20} {total_pass}/{total_count:<8} {total_pass*100//total_count:>4}%")
print("#" * 60)
