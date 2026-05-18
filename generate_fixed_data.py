"""
Delta Filing — Fixed Supplementary Data Generator
===================================================
Fixes all P0 and P1 data quality issues from the previous version:

P0-1: Tool calling — full coverage, every ticker × every tool × every template
P0-2: Tool result parsing — real filing text, not placeholders
P1:   Review — varied feedback per sample, not one fixed string

Also regenerates router and synthesis data (minor fixes).

Usage:
    # Test mode (3 tickers, verify format before full run)
    python generate_fixed_data.py --test

    # Full run
    python generate_fixed_data.py
"""

import json
import os
import random
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

OUTPUT_DIR = Path("training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(42)


# ============================================================
#  Constants
# ============================================================

ALL_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "JPM",
    "V", "JNJ", "WMT", "PFE", "KO", "DIS", "NKE", "HD", "COST",
    "NFLX", "CRM", "ADBE", "ORCL", "AMD", "INTC", "BA", "CAT",
]

TEST_TICKERS = ["AAPL", "MSFT", "GOOGL"]

# Ticker to company name mapping (so model learns both)
TICKER_TO_NAME = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google/Alphabet",
    "NVDA": "NVIDIA", "META": "Meta", "AMZN": "Amazon", "TSLA": "Tesla",
    "JPM": "JPMorgan", "V": "Visa", "JNJ": "Johnson & Johnson",
    "WMT": "Walmart", "PFE": "Pfizer", "KO": "Coca-Cola", "DIS": "Disney",
    "NKE": "Nike", "HD": "Home Depot", "COST": "Costco", "NFLX": "Netflix",
    "CRM": "Salesforce", "ADBE": "Adobe", "ORCL": "Oracle", "AMD": "AMD",
    "INTC": "Intel", "BA": "Boeing", "CAT": "Caterpillar",
}

SYSTEM_PROMPT = (
    "You are Delta Filing, a financial analyst AI specializing in SEC filing analysis. "
    "You have access to tools for retrieving SEC filings, financial data, news, and insider trades. "
    "When you need data, call the appropriate tool. When you have data, analyze it thoroughly."
)

TOOLS_SCHEMA = """Available tools:

1. search_filings(ticker: str, filing_type: str = "10-K", count: int = 5)
   List recent SEC filings for a company. Returns filing dates and accession numbers.

2. get_filing_section(ticker: str, section_id: str, filing_type: str = "10-K", filing_index: int = 0)
   Get text of a specific section from a filing. Sections: 1 (Business), 1A (Risk Factors), 7 (MD&A), 7A (Market Risk), 8 (Financial Statements).

3. diff_filing_sections(ticker: str, section_id: str, filing_type: str = "10-K")
   Compare same section across two consecutive filings. Returns added, removed, modified paragraphs.

4. stock_price(ticker: str, period: str = "1mo")
   Get recent stock price, change %, period high/low.

5. company_metrics(ticker: str)
   Get PE ratio, market cap, revenue, margins, ROE, debt-to-equity.

6. company_news(ticker: str, days: int = 7)
   Get recent news headlines and summaries.

7. insider_trades(ticker: str)
   Get recent insider buy/sell transactions.

8. analyst_ratings(ticker: str)
   Get analyst consensus buy/hold/sell and price targets.

To use a tool, respond with a JSON object:
{"tool": "tool_name", "arguments": {"param": "value"}}
Respond with ONLY the JSON, nothing else."""


# ============================================================
#  1. Tool Calling Data — FULL COVERAGE
# ============================================================

def generate_tool_calling_data(tickers: list[str]) -> list[dict]:
    """Generate tool calling examples with full ticker coverage.

    Every ticker appears with every tool at least once.
    Uses both ticker and company name in queries so model
    learns to handle both "AAPL" and "Apple".
    """
    examples = []

    templates = [
        # search_filings
        {
            "queries_with_ticker": [
                "List {ticker}'s recent 10-K filings",
                "Show me SEC filings for {ticker}",
            ],
            "queries_with_name": [
                "What 10-K filings has {name} submitted recently?",
                "Find {name}'s annual reports",
            ],
            "tool": "search_filings",
            "args_fn": lambda t: {"ticker": t, "filing_type": "10-K", "count": 5},
        },
        # get_filing_section — Risk Factors
        {
            "queries_with_ticker": [
                "What are {ticker}'s risk factors?",
                "Show me Item 1A from {ticker}'s 10-K",
            ],
            "queries_with_name": [
                "Analyze the risk factors in {name}'s latest 10-K",
                "What risks does {name} disclose?",
            ],
            "tool": "get_filing_section",
            "args_fn": lambda t: {"ticker": t, "section_id": "1A", "filing_type": "10-K"},
        },
        # get_filing_section — MD&A
        {
            "queries_with_ticker": [
                "Get {ticker}'s MD&A section",
                "Show me Item 7 from {ticker}'s filing",
            ],
            "queries_with_name": [
                "What does {name}'s management discussion say?",
                "Analyze {name}'s MD&A",
            ],
            "tool": "get_filing_section",
            "args_fn": lambda t: {"ticker": t, "section_id": "7", "filing_type": "10-K"},
        },
        # get_filing_section — Business
        {
            "queries_with_ticker": [
                "Show me {ticker}'s business description",
                "Get Item 1 from {ticker}'s 10-K",
            ],
            "queries_with_name": [
                "Describe {name}'s business from their filing",
            ],
            "tool": "get_filing_section",
            "args_fn": lambda t: {"ticker": t, "section_id": "1", "filing_type": "10-K"},
        },
        # diff_filing_sections — Risk Factors
        {
            "queries_with_ticker": [
                "How did {ticker}'s risk factors change from last year?",
                "Compare {ticker}'s latest and previous risk disclosures",
            ],
            "queries_with_name": [
                "What changed in {name}'s risk factors year over year?",
                "Show differences in {name}'s Item 1A between filings",
            ],
            "tool": "diff_filing_sections",
            "args_fn": lambda t: {"ticker": t, "section_id": "1A", "filing_type": "10-K"},
        },
        # diff_filing_sections — MD&A
        {
            "queries_with_ticker": [
                "Compare {ticker}'s MD&A across filings",
            ],
            "queries_with_name": [
                "How did {name}'s management discussion change?",
            ],
            "tool": "diff_filing_sections",
            "args_fn": lambda t: {"ticker": t, "section_id": "7", "filing_type": "10-K"},
        },
        # stock_price
        {
            "queries_with_ticker": [
                "What's {ticker}'s current stock price?",
                "How has {ticker} performed this month?",
            ],
            "queries_with_name": [
                "Show me {name}'s recent price movement",
                "What's {name}'s stock doing?",
            ],
            "tool": "stock_price",
            "args_fn": lambda t: {"ticker": t, "period": "1mo"},
        },
        # company_metrics
        {
            "queries_with_ticker": [
                "What's {ticker}'s PE ratio and market cap?",
                "Show me key metrics for {ticker}",
            ],
            "queries_with_name": [
                "What are {name}'s financial metrics?",
                "Get {name}'s valuation ratios",
            ],
            "tool": "company_metrics",
            "args_fn": lambda t: {"ticker": t},
        },
        # company_news
        {
            "queries_with_ticker": [
                "What's the latest news about {ticker}?",
                "Any recent headlines for {ticker}?",
            ],
            "queries_with_name": [
                "Show me recent news about {name}",
                "What's been happening with {name}?",
            ],
            "tool": "company_news",
            "args_fn": lambda t: {"ticker": t, "days": 7},
        },
        # insider_trades
        {
            "queries_with_ticker": [
                "Have {ticker} insiders been buying or selling?",
                "Show insider trades for {ticker}",
            ],
            "queries_with_name": [
                "What are {name} executives doing with their shares?",
                "Any insider trading at {name}?",
            ],
            "tool": "insider_trades",
            "args_fn": lambda t: {"ticker": t},
        },
        # analyst_ratings
        {
            "queries_with_ticker": [
                "What do analysts say about {ticker}?",
                "Show me analyst ratings for {ticker}",
            ],
            "queries_with_name": [
                "What's the analyst consensus on {name}?",
                "Get {name}'s buy/sell ratings",
            ],
            "tool": "analyst_ratings",
            "args_fn": lambda t: {"ticker": t},
        },
    ]

    system_content = SYSTEM_PROMPT + "\n\n" + TOOLS_SCHEMA

    for ticker in tickers:
        name = TICKER_TO_NAME.get(ticker, ticker)

        for template in templates:
            # Generate one example with ticker in query
            for q in template["queries_with_ticker"]:
                query = q.format(ticker=ticker)
                tool_call = {"tool": template["tool"], "arguments": template["args_fn"](ticker)}
                examples.append({
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": json.dumps(tool_call)},
                    ],
                    "category": "tool_calling",
                })

            # Generate one example with company name in query
            for q in template.get("queries_with_name", []):
                query = q.format(name=name)
                tool_call = {"tool": template["tool"], "arguments": template["args_fn"](ticker)}
                examples.append({
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": json.dumps(tool_call)},
                    ],
                    "category": "tool_calling",
                })

    random.shuffle(examples)
    print(f"    Tool calling: {len(examples)} examples "
          f"({len(tickers)} tickers × {len(templates)} tools)")
    return examples


# ============================================================
#  2. Tool Result Parsing — REAL FILING TEXT
# ============================================================

def generate_tool_result_data(tickers: list[str]) -> list[dict]:
    """Generate tool result parsing examples using REAL filing text.

    Fetches actual filing sections from EDGAR, wraps in MCP JSON,
    then uses GPT to generate analysis that references the JSON content.
    """
    from edgar import get_section

    examples = []
    sections_to_try = ["1A", "7", "1", "7A"]
    section_names = {"1A": "Risk Factors", "7": "MD&A", "1": "Business", "7A": "Market Risk"}

    for ticker in tickers:
        for section_id in sections_to_try:
            # Fetch real filing text
            try:
                section = get_section(ticker, "10-K", section_id, filing_index=0)
            except Exception as e:
                print(f"    Skipping {ticker} {section_id}: {e}")
                continue

            # Truncate to ~800 chars (enough for meaningful analysis)
            real_text = section["text"][:800]
            if len(section["text"]) > 800:
                real_text += "..."

            # Build MCP-style JSON result
            tool_result = json.dumps({
                "company": ticker,
                "filing_date": section["filing_date"],
                "section_name": section_names.get(section_id, section_id),
                "text_length": len(section["text"]),
                "text": real_text,
            }, indent=2)

            # Use GPT to generate analysis that explicitly references
            # the company name, section name, and content from the JSON
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": (
                            "You are a financial analyst. You received tool output in JSON format. "
                            "Analyze the filing section. You MUST:\n"
                            "1. Reference the company by its ticker from the JSON\n"
                            "2. Reference the specific section name (e.g., 'Item 1A Risk Factors')\n"
                            "3. Reference the filing date from the JSON\n"
                            "4. Quote or paraphrase specific content from the text\n"
                            "5. Provide analytical judgment\n"
                            "Be concise but thorough. 150-250 words."
                        )},
                        {"role": "user", "content": (
                            f"I retrieved this filing section using get_filing_section. "
                            f"Please analyze it:\n{tool_result}"
                        )},
                    ],
                    temperature=0.7,
                    max_tokens=600,
                )
                analysis = response.choices[0].message.content.strip()

                examples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"I retrieved this filing section using get_filing_section. "
                            f"Please analyze it:\n{tool_result}"
                        )},
                        {"role": "assistant", "content": analysis},
                    ],
                    "category": "tool_result_parsing",
                })
                print(f"    Tool result: {ticker} {section_id} done")

            except Exception as e:
                print(f"    Tool result GPT failed for {ticker} {section_id}: {e}")

            time.sleep(0.3)

    random.shuffle(examples)
    return examples


# ============================================================
#  3. Review Data — VARIED FEEDBACK
# ============================================================

def generate_review_data() -> list[dict]:
    """Generate review judgment examples with varied, specific feedback.

    Uses existing DPO data. For each pair:
    - Chosen (good analysis) → COMPLETE
    - Rejected (bad analysis) → NEEDS_FOLLOW_UP with SPECIFIC reason

    The reason is determined by checking what the rejected answer is missing.
    """
    review_system = (
        "You are a quality reviewer for SEC filing analysis.\n"
        "Review the analysis and determine if it is thorough enough.\n"
        "A good analysis MUST:\n"
        "1. Reference specific filing sections (Item 1A, Item 7, etc.)\n"
        "2. Cite specific numbers, dates, or quotes\n"
        "3. Provide analytical judgment, not just summary\n"
        "4. Address the user's original question directly\n\n"
        "If missing any of these, respond with:\n"
        "NEEDS_FOLLOW_UP: [specific description of what is missing]\n\n"
        "If thorough enough, respond with:\n"
        "COMPLETE\n\n"
        "Respond with ONLY one of these formats."
    )

    dpo_path = OUTPUT_DIR / "dpo_data_combined.jsonl"
    if not dpo_path.exists():
        print("    Warning: dpo_data_combined.jsonl not found, skipping review data")
        return []

    with open(dpo_path) as f:
        dpo_data = [json.loads(line) for line in f]

    examples = []

    for ex in dpo_data[:60]:  # Use 60 pairs → 120 review examples
        # --- COMPLETE example (good analysis) ---
        examples.append({
            "messages": [
                {"role": "system", "content": review_system},
                {"role": "user", "content": (
                    f"Original question: {ex['question']}\n\n"
                    f"Analysis to review:\n{ex['chosen']}"
                )},
                {"role": "assistant", "content": "COMPLETE"},
            ],
            "category": "review",
        })

        # --- NEEDS_FOLLOW_UP example (bad analysis) ---
        # Diagnose specific problems in the rejected answer
        rejected = ex["rejected"]
        problems = []

        # Check for missing section references
        has_item_ref = any(kw in rejected for kw in ["Item 1", "Item 7", "10-K", "10-Q", "MD&A"])
        if not has_item_ref:
            problems.append("does not reference specific filing sections (e.g., Item 1A, Item 7)")

        # Check for missing numbers
        import re
        has_numbers = bool(re.search(r'\d+\.?\d*[%$]|\$[\d,]+|\d{4}', rejected))
        if not has_numbers:
            problems.append("lacks specific numbers, percentages, or dates")

        # Check for analytical depth
        judgment_words = ["suggests", "indicates", "concerning", "notable", "significant",
                         "however", "despite", "although", "risk", "impact", "severity"]
        has_judgment = sum(1 for w in judgment_words if w in rejected.lower()) >= 2
        if not has_judgment:
            problems.append("provides only surface-level summary without analytical judgment")

        # Check length
        if len(rejected) < 400:
            problems.append("is too brief to adequately address the question")

        # Build specific feedback
        if not problems:
            problems.append("lacks depth and specificity compared to what a thorough analysis requires")

        if len(problems) > 1:
            problems = random.sample(problems, random.randint(1, min(2, len(problems))))

        feedback = "NEEDS_FOLLOW_UP: The analysis " + "; ".join(problems) + "."

        examples.append({
            "messages": [
                {"role": "system", "content": review_system},
                {"role": "user", "content": (
                    f"Original question: {ex['question']}\n\n"
                    f"Analysis to review:\n{rejected}"
                )},
                {"role": "assistant", "content": feedback},
            ],
            "category": "review",
        })

    random.shuffle(examples)
    return examples


# ============================================================
#  4. Router Data — same as before, template-based
# ============================================================

def generate_router_data(tickers: list[str]) -> list[dict]:
    """Generate router classification examples. Template-based, no API."""
    router_system = (
        "You are a query classifier for a SEC filing analysis system.\n"
        "Classify into exactly ONE category:\n"
        "- FILING_ANALYSIS: analyze a specific filing section\n"
        "- FILING_DIFF: compare filings across time periods\n"
        "- COMPANY_DEEP_DIVE: comprehensive company overview\n"
        "- SIMPLE_QUERY: quick data lookup\n"
        "- OUT_OF_SCOPE: not related to SEC filings\n"
        "Reply with ONLY the category name, nothing else."
    )

    templates = {
        "FILING_ANALYSIS": [
            "What are {ticker}'s risk factors?",
            "Analyze {name}'s MD&A section",
            "Summarize {ticker}'s Item 1A",
            "What does {name}'s latest 10-K say about competition?",
            "Show me {ticker}'s business description from their filing",
            "What risks does {name} disclose in their annual report?",
        ],
        "FILING_DIFF": [
            "How did {ticker}'s risk factors change from last year?",
            "What's different in {name}'s latest 10-K compared to the previous one?",
            "Compare {ticker}'s risk disclosures year over year",
            "What new risks did {name} add in their latest filing?",
            "Did {ticker} remove any risk factors?",
        ],
        "COMPANY_DEEP_DIVE": [
            "Full analysis of {ticker}",
            "Give me a comprehensive overview of {name}",
            "What's going on with {ticker}?",
            "Deep dive into {name}",
            "Tell me everything about {ticker}'s current situation",
        ],
        "SIMPLE_QUERY": [
            "What's {ticker}'s stock price?",
            "List {name}'s recent filings",
            "What's {ticker}'s PE ratio?",
            "Show me {name}'s market cap",
            "Any recent news about {ticker}?",
        ],
        "OUT_OF_SCOPE": [
            "What's the weather in Zurich?",
            "Write me a poem about finance",
            "How do I cook pasta?",
            "What's the capital of France?",
            "Tell me a joke",
            "Help me with my homework",
            "Translate this to French",
            "What time is it in Tokyo?",
        ],
    }

    examples = []
    for category, query_templates in templates.items():
        for template in query_templates:
            if "{ticker}" in template or "{name}" in template:
                for ticker in random.sample(tickers, min(3, len(tickers))):
                    name = TICKER_TO_NAME.get(ticker, ticker)
                    query = template.format(ticker=ticker, name=name)
                    examples.append({
                        "messages": [
                            {"role": "system", "content": router_system},
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": category},
                        ],
                        "category": "router",
                    })
            else:
                examples.append({
                    "messages": [
                        {"role": "system", "content": router_system},
                        {"role": "user", "content": template},
                        {"role": "assistant", "content": category},
                    ],
                    "category": "router",
                })

    random.shuffle(examples)
    return examples


# ============================================================
#  5. Synthesis Data — REAL market data
# ============================================================

def generate_synthesis_data(tickers: list[str]) -> list[dict]:
    """Generate synthesis examples using real market data where possible."""
    from market import get_key_metrics, get_stock_price, get_company_news, get_insider_trades

    synthesis_system = (
        "You are a financial analyst synthesizing a company deep dive report.\n"
        "Combine financial data, recent news, and insider trading into "
        "a concise overview. Structure as:\n"
        "- Company snapshot (key metrics)\n"
        "- Recent developments (news)\n"
        "- Insider activity\n"
        "- Overall assessment\n"
        "Be specific with numbers. Note concerning patterns."
    )

    examples = []
    sample = random.sample(tickers, min(10, len(tickers)))

    for ticker in sample:
        try:
            # Fetch real data
            metrics = get_key_metrics(ticker)
            price = get_stock_price(ticker, "1mo")
            try:
                news = get_company_news(ticker, days=7)[:3]
            except Exception:
                news = [{"headline": "No recent news available", "date": "N/A"}]
            try:
                trades = get_insider_trades(ticker)[:3]
            except Exception:
                trades = [{"name": "N/A", "transaction_type": "N/A", "change": 0}]

            financial_json = json.dumps({"metrics": metrics, "price": price}, indent=2)
            news_json = json.dumps(news, indent=2)
            trades_json = json.dumps(trades, indent=2)

            combined = (
                f"FINANCIAL DATA:\n{financial_json}\n\n"
                f"RECENT NEWS:\n{news_json}\n\n"
                f"INSIDER TRADES:\n{trades_json}"
            )

            # GPT generates synthesis report from real data
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": synthesis_system},
                    {"role": "user", "content": f"Company: {ticker}\n\nData:\n{combined}"},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            report = response.choices[0].message.content.strip()

            examples.append({
                "messages": [
                    {"role": "system", "content": synthesis_system},
                    {"role": "user", "content": f"Company: {ticker}\n\nData:\n{combined}"},
                    {"role": "assistant", "content": report},
                ],
                "category": "synthesis",
            })
            print(f"    Synthesis: {ticker} done")

        except Exception as e:
            print(f"    Synthesis: {ticker} failed: {e}")

        time.sleep(0.5)

    return examples


# ============================================================
#  6. Tool Calling DPO
# ============================================================

def generate_tool_calling_dpo(tool_calling_data: list[dict],
                               max_pairs: int = 200) -> list[dict]:
    """Generate DPO pairs for tool calling: correct tool vs wrong tool."""
    all_tools = [
        "search_filings", "get_filing_section", "diff_filing_sections",
        "stock_price", "company_metrics", "company_news",
        "insider_trades", "analyst_ratings",
    ]

    dpo = []
    sampled = random.sample(tool_calling_data, min(max_pairs, len(tool_calling_data)))

    for ex in sampled:
        chosen = ex["messages"][2]["content"]
        try:
            correct = json.loads(chosen)
        except json.JSONDecodeError:
            continue

        # Wrong tool
        wrong_tools = [t for t in all_tools if t != correct["tool"]]
        wrong_tool = random.choice(wrong_tools)
        rejected = json.dumps({"tool": wrong_tool, "arguments": correct["arguments"]})

        dpo.append({
            "question": ex["messages"][1]["content"],
            "chosen": chosen,
            "rejected": rejected,
            "category": "tool_calling",
        })

    return dpo


# ============================================================
#  7. Tool Result Parsing DPO
# ============================================================

TOOL_RESULT_DEGRADE_PROMPT = """Take this detailed financial analysis based on a SEC filing tool result and create a WORSE version.
The worse version should:
- Remove references to the company ticker
- Remove references to the specific section name (e.g., Item 1A, Item 7)
- Remove the filing date
- Replace specific quotes from the filing with vague generalizations
- Be shorter (50-100 words)
- Sound generic, like it could apply to any company's filing

Original analysis:
{good_answer}

Respond with ONLY the degraded text, nothing else."""


def generate_tool_result_dpo(tool_result_data: list[dict]) -> list[dict]:
    """Generate DPO pairs for tool result parsing: specific analysis vs generic."""
    dpo = []

    for ex in tool_result_data:
        question = ex["messages"][1]["content"]
        chosen = ex["messages"][2]["content"]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": TOOL_RESULT_DEGRADE_PROMPT.format(
                    good_answer=chosen
                )}],
                temperature=0.9,
                max_tokens=300,
            )
            rejected = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Tool result DPO degrade failed: {e}")
            continue

        dpo.append({
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "category": "tool_result_parsing",
        })
        time.sleep(0.3)

    return dpo


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 3 tickers only, verify format")
    args = parser.parse_args()

    tickers = TEST_TICKERS if args.test else ALL_TICKERS
    mode = "TEST" if args.test else "FULL"

    print("=" * 60)
    print(f"  Delta Filing — Fixed Data Generator ({mode} mode)")
    print(f"  Tickers: {len(tickers)}")
    print("=" * 60)

    # --- Generate all data types ---
    print("\n[1/5] Tool calling (template, no API)...")
    tool_calling = generate_tool_calling_data(tickers)

    print("\n[2/5] Tool result parsing (real filing text + GPT)...")
    tool_result = generate_tool_result_data(tickers)

    print("\n[3/5] Review judgments (from existing DPO, varied feedback)...")
    review = generate_review_data()

    print("\n[4/5] Router classification (template, no API)...")
    router = generate_router_data(tickers)

    print("\n[5/5] Synthesis (real market data + GPT)...")
    synthesis = generate_synthesis_data(tickers)

    # --- Combine ---
    all_supplementary = tool_calling + tool_result + review + router + synthesis
    random.shuffle(all_supplementary)

    # --- Category breakdown ---
    cats = {}
    for ex in all_supplementary:
        cat = ex.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Category breakdown ({len(all_supplementary)} total):")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        pct = count * 100 // len(all_supplementary)
        print(f"    {cat}: {count} ({pct}%)")

    # --- Quality checks ---
    print(f"\n  Quality checks:")

    # Check tool calling covers all tickers
    tc_tickers = set()
    for ex in tool_calling:
        try:
            tc = json.loads(ex["messages"][2]["content"])
            tc_tickers.add(tc["arguments"]["ticker"])
        except (json.JSONDecodeError, KeyError):
            pass
    missing = set(tickers) - tc_tickers
    print(f"    Tool calling ticker coverage: {len(tc_tickers)}/{len(tickers)} "
          f"{'✅' if not missing else f'❌ missing: {missing}'}")

    # Check tool result parsing uses real text (not placeholder)
    placeholder_count = 0
    for ex in tool_result:
        if "[Filing section content" in ex["messages"][1]["content"]:
            placeholder_count += 1
    print(f"    Tool result placeholders: {placeholder_count} "
          f"{'✅ none' if placeholder_count == 0 else '❌ has placeholders'}")

    # Check review has varied feedback
    review_feedbacks = set()
    for ex in review:
        assistant = ex["messages"][2]["content"]
        if assistant.startswith("NEEDS_FOLLOW_UP"):
            review_feedbacks.add(assistant)
    print(f"    Review unique feedbacks: {len(review_feedbacks)} "
          f"{'✅' if len(review_feedbacks) > 5 else '❌ too few variations'}")

    # --- Save ---
    suffix = "_test" if args.test else ""

    supp_path = OUTPUT_DIR / f"sft_supplementary_fixed{suffix}.jsonl"
    with open(supp_path, "w") as f:
        for ex in all_supplementary:
            f.write(json.dumps(ex) + "\n")
    print(f"\n  Saved: {supp_path}")

    # Tool calling DPO
    tc_dpo = generate_tool_calling_dpo(tool_calling)
    tr_dpo = generate_tool_result_dpo(tool_result)
    all_dpo = tc_dpo + tr_dpo
    dpo_path = OUTPUT_DIR / f"dpo_tool_calling_fixed{suffix}.jsonl"
    with open(dpo_path, "w") as f:
        for ex in all_dpo:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved: {dpo_path} ({len(all_dpo)} DPO pairs: "
          f"{len(tc_dpo)} tool_calling, {len(tr_dpo)} tool_result_parsing)")

    # --- Verify one sample per category ---
    if args.test:
        print(f"\n  Sample verification:")
        seen = set()
        for ex in all_supplementary:
            cat = ex.get("category")
            if cat not in seen:
                seen.add(cat)
                print(f"\n    [{cat}]")
                print(f"    User: {ex['messages'][1]['content'][:100]}...")
                print(f"    Asst: {ex['messages'][2]['content'][:100]}...")

    print(f"\n{'='*60}")
    if args.test:
        print(f"  TEST complete. Review samples above.")
        print(f"  If OK, run without --test for full generation.")
    else:
        print(f"  FULL generation complete.")
        print(f"  Next: run merge_training_data.py to combine with analysis data.")
    print(f"{'='*60}")
