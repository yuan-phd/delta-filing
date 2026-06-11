#!/usr/bin/env python
"""§3a Track-1 eval generation — GPT-4o-mini labeler (same method as existing data).

Grows the two pivotal Track-1 eval tasks to ~80 balanced items each:
  - binary (Review): output space exactly {COMPLETE, NEEDS_FOLLOW_UP}, balanced,
    gold per item (we control quality of the generated analysis -> known label).
  - json (tool-calling): gold = {tool, arguments:{ticker}}, balanced across the 8
    tools; GPT-4o-mini phrases a natural query per (tool, ticker) target.

Quality gates (test_data_quality.py style): no <think>/placeholders, balanced
classes, gold present, no overlap with training questions.

The validated 5-item smoke slice lives separately in
data_eval/track1_{binary,json}_smoke.jsonl and is NOT touched here.

  python data_prep/gen_track1_eval.py --n-binary 80 --n-json 80          # full
  python data_prep/gen_track1_eval.py --n-binary 5 --n-json 5 --tag smoke  # slice
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data_eval")
TRAIN_PREF = os.path.join(ROOT, "training_data", "dpo_pytorch.jsonl")

PLACEHOLDER_TOKENS = ["TODO", "PLACEHOLDER", "lorem ipsum", "XXXX", "<think>", "</think>",
                      "[insert", "{{", "}}", "your analysis here"]

# 40 real large/mid-cap companies (ticker, name) -> 2 binary items each = 80.
COMPANIES = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"), ("GOOGL", "Alphabet"),
    ("AMZN", "Amazon"), ("META", "Meta"), ("TSLA", "Tesla"), ("NFLX", "Netflix"),
    ("DIS", "Disney"), ("BA", "Boeing"), ("JPM", "JPMorgan Chase"), ("BAC", "Bank of America"),
    ("WMT", "Walmart"), ("COST", "Costco"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"),
    ("MCD", "McDonald's"), ("NKE", "Nike"), ("INTC", "Intel"), ("AMD", "AMD"),
    ("CRM", "Salesforce"), ("ORCL", "Oracle"), ("ADBE", "Adobe"), ("CSCO", "Cisco"),
    ("IBM", "IBM"), ("QCOM", "Qualcomm"), ("TXN", "Texas Instruments"), ("PFE", "Pfizer"),
    ("MRK", "Merck"), ("JNJ", "Johnson & Johnson"), ("UNH", "UnitedHealth"), ("XOM", "ExxonMobil"),
    ("CVX", "Chevron"), ("CAT", "Caterpillar"), ("GE", "GE Aerospace"), ("F", "Ford"),
    ("GM", "General Motors"), ("T", "AT&T"), ("VZ", "Verizon"), ("PYPL", "PayPal"),
]
TOOLS = ["search_filings", "get_filing_section", "diff_filing_sections", "stock_price",
         "company_metrics", "company_news", "insider_trades", "analyst_ratings"]

REVIEW_GEN_SYS = (
    "You write realistic financial-analysis snippets for a benchmark. Given a company "
    "and a target quality, output ONLY a short analysis paragraph (3-6 sentences) of that "
    "company's latest 10-K risk factors. No preamble, no labels, no <think> tags."
)
QUERY_GEN_SYS = (
    "You write realistic end-user queries for a SEC-filing analysis assistant. Output ONLY "
    "the user's one-sentence query, nothing else. No preamble, no quotes, no <think> tags."
)
TOOL_INTENT = {
    "search_filings": "list/find a company's recent filings",
    "get_filing_section": "read a specific section (e.g. risk factors, MD&A) of a filing",
    "diff_filing_sections": "compare how a filing section changed versus the prior year",
    "stock_price": "the company's stock price over a period",
    "company_metrics": "key financial metrics/ratios for the company",
    "company_news": "recent news about the company",
    "insider_trades": "recent insider trading activity at the company",
    "analyst_ratings": "the analyst consensus/ratings on the company",
}


def has_placeholder(text):
    low = text.lower()
    return any(tok.lower() in low for tok in PLACEHOLDER_TOKENS)


def gen_review_analysis(company, ticker, quality):
    if quality == "COMPLETE":
        instruct = (f"Target quality = THOROUGH: cite specific 10-K section references "
                    f"(e.g. Item 1A), concrete numbers/percentages/dollar figures, and dates, "
                    f"with clear analytical judgment for {company} ({ticker}).")
    else:
        instruct = (f"Target quality = SHALLOW/VAGUE: generic statements about {company} "
                    f"({ticker}) risks with NO specific numbers, NO section references, and "
                    f"NO analytical depth (the kind that should need follow-up).")
    r = client.chat.completions.create(
        model=MODEL, temperature=0.7,
        messages=[{"role": "system", "content": REVIEW_GEN_SYS},
                  {"role": "user", "content": instruct}])
    return r.choices[0].message.content.strip()


def gen_tool_query(tool, ticker, company):
    instruct = (f"Write a natural user query that asks for: {TOOL_INTENT[tool]}. "
                f"It must clearly be about {company} ({ticker}). Vary the phrasing.")
    r = client.chat.completions.create(
        model=MODEL, temperature=0.8,
        messages=[{"role": "system", "content": QUERY_GEN_SYS},
                  {"role": "user", "content": instruct}])
    return r.choices[0].message.content.strip().strip('"')


def build_binary(n):
    items, i = [], 0
    # interleave COMPLETE / NEEDS_FOLLOW_UP per company -> balanced
    plan = []
    for ticker, company in COMPANIES:
        plan.append((ticker, company, "COMPLETE"))
        plan.append((ticker, company, "NEEDS_FOLLOW_UP"))
    for ticker, company, gold in plan[:n]:
        analysis = gen_review_analysis(company, ticker, gold)
        prompt = (f"Original question: What are {company}'s key risk factors?\n\n"
                  f"Analysis: {analysis}")
        assert not has_placeholder(prompt), f"placeholder in binary item {i}"
        assert len(analysis) > 40, f"binary analysis too short, item {i}"
        items.append({"item_id": f"t1_binary_{i:03d}", "prompt": prompt,
                      "gold": gold, "category": "binary"})
        i += 1
    return items


def build_json(n):
    items, i = [], 0
    per_tool = max(1, n // len(TOOLS))
    plan = []
    for t_idx, tool in enumerate(TOOLS):
        for j in range(per_tool):
            ticker, company = COMPANIES[(t_idx * per_tool + j) % len(COMPANIES)]
            plan.append((tool, ticker, company))
    for tool, ticker, company in plan[:n]:
        query = gen_tool_query(tool, ticker, company)
        assert not has_placeholder(query), f"placeholder in json item {i}"
        assert ticker in query or company.split()[0] in query, \
            f"json query {i} doesn't name the company/ticker: {query!r}"
        items.append({"item_id": f"t1_json_{i:03d}", "prompt": query,
                      "gold": {"tool": tool, "arguments": {"ticker": ticker}},
                      "category": "function_call"})
        i += 1
    return items


def check_no_overlap(items, key):
    if not os.path.exists(TRAIN_PREF):
        return
    train_qs = {json.loads(l)["question"].strip() for l in open(TRAIN_PREF) if l.strip()}
    for it in items:
        assert it[key].strip() not in train_qs, f"overlap with training: {it['item_id']}"


def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-binary", type=int, default=80)
    ap.add_argument("--n-json", type=int, default=80)
    ap.add_argument("--tag", default="", help="filename suffix, e.g. 'smoke' (default: full set)")
    args = ap.parse_args()

    binary = build_binary(args.n_binary)
    jsons = build_json(args.n_json)

    # quality gates
    from collections import Counter
    bal = Counter(it["gold"] for it in binary)
    assert set(bal) == {"COMPLETE", "NEEDS_FOLLOW_UP"}, f"binary not balanced: {bal}"
    assert abs(bal["COMPLETE"] - bal["NEEDS_FOLLOW_UP"]) <= 1, f"binary imbalance: {bal}"
    tool_bal = Counter(it["gold"]["tool"] for it in jsons)
    check_no_overlap(binary, "prompt")
    check_no_overlap(jsons, "prompt")

    suffix = f"_{args.tag}" if args.tag else ""
    bpath = os.path.join(OUT_DIR, f"track1_binary{suffix}.jsonl")
    jpath = os.path.join(OUT_DIR, f"track1_json{suffix}.jsonl")
    write_jsonl(bpath, binary)
    write_jsonl(jpath, jsons)

    print(f"binary: {len(binary)} items  balance={dict(bal)}")
    print(f"json:   {len(jsons)} items  tool_balance={dict(tool_bal)}")
    print(f"-> {bpath}\n-> {jpath}")


if __name__ == "__main__":
    sys.exit(main())
