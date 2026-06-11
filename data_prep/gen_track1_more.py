#!/usr/bin/env python
"""§3a — generate the two NEW candidate Track-1 eval tasks (pre-registered):

  router     : closed-set 5-class enum (FILING_ANALYSIS / FILING_DIFF /
               COMPANY_DEEP_DIVE / SIMPLE_QUERY / OUT_OF_SCOPE). gold = category.
  extraction : flat JSON schema {ticker, section, filing_date} from a filing
               snippet. gold = the three fields (date canonical YYYY-MM-DD).

Both are rigid exact-format, format-fragile, and degeneracy-visible (router floor
1/5, extraction ~1/n). Inclusion in the final suite is decided by the BASELINE
competence gate (>=0.80 free-generation), NOT by whether they collapse.

Quality gates: balanced classes, n~80, no overlap with training, no placeholders,
gold-as-output -> PASS under the scorer.

  python data_prep/gen_track1_more.py --n-router 80 --n-extraction 80
"""
import argparse
import json
import os
import sys
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scorer as SC  # noqa: E402

load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data_eval")
TRAIN_PREF = os.path.join(ROOT, "training_data", "dpo_pytorch.jsonl")
PLACEHOLDERS = ["TODO", "PLACEHOLDER", "lorem ipsum", "XXXX", "<think>", "</think>", "[insert", "{{", "}}"]

COMPANIES = [("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"), ("AMZN", "Amazon"),
             ("TSLA", "Tesla"), ("GOOGL", "Alphabet"), ("META", "Meta"), ("JPM", "JPMorgan"),
             ("XOM", "ExxonMobil"), ("WMT", "Walmart"), ("KO", "Coca-Cola"), ("BA", "Boeing"),
             ("NKE", "Nike"), ("DIS", "Disney"), ("CAT", "Caterpillar"), ("PFE", "Pfizer")]
SECTIONS = ["Item 1A", "Item 7", "Item 8", "Item 1", "Item 7A", "Item 3"]

ROUTER_DEFS = {
    "FILING_ANALYSIS": "a question about the content of ONE specific company's filing (a single period)",
    "FILING_DIFF": "a question COMPARING a company's filings across two time periods / years",
    "COMPANY_DEEP_DIVE": "a request for a COMPREHENSIVE multi-source analysis of a company",
    "SIMPLE_QUERY": "a simple factual lookup about filings (e.g. list filings, a filing date)",
    "OUT_OF_SCOPE": "a question UNRELATED to SEC filings or finance (weather, sports, cooking, trivia)",
}
ROUTER_SYS = ("You write realistic user queries for a benchmark. Output ONLY the one-sentence "
              "query, no quotes, no preamble, no <think> tags.")


def placeholder(t):
    low = t.lower()
    return any(p.lower() in low for p in PLACEHOLDERS)


def gen_router_query(category, company):
    nudge = (f" about {company}" if category != "OUT_OF_SCOPE" else "")
    r = client.chat.completions.create(
        model=MODEL, temperature=0.85,
        messages=[{"role": "system", "content": ROUTER_SYS},
                  {"role": "user", "content":
                   f"Write a query that is unambiguously {ROUTER_DEFS[category]}{nudge}."}])
    return r.choices[0].message.content.strip().strip('"')


def gen_extraction_snippet(ticker, company, section, date_iso):
    import datetime
    human = datetime.datetime.strptime(date_iso, "%Y-%m-%d").strftime("%B %d, %Y")
    sysmsg = ("You write a realistic 2-3 sentence excerpt from a SEC filing for a benchmark. "
              "It MUST clearly state the ticker, the section, and the filing date. No <think> tags.")
    user = (f"Company {company} (ticker {ticker}); section {section}; filing date {human}. "
            f"Write the excerpt; mention all three explicitly.")
    r = client.chat.completions.create(
        model=MODEL, temperature=0.7,
        messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": user}])
    return r.choices[0].message.content.strip()


def build_router(n):
    per = max(1, n // len(ROUTER_DEFS))
    items, i = [], 0
    for cat in ROUTER_DEFS:
        for j in range(per):
            company = COMPANIES[(i) % len(COMPANIES)][1]
            q = gen_router_query(cat, company)
            assert not placeholder(q), f"placeholder router {i}"
            items.append({"item_id": f"t1_router_{i:03d}", "prompt": q,
                          "gold": cat, "category": "router"})
            i += 1
    return items[:n]


def build_extraction(n):
    items, i = [], 0
    while i < n:
        ticker, company = COMPANIES[i % len(COMPANIES)]
        section = SECTIONS[i % len(SECTIONS)]
        date_iso = f"2025-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}"
        snippet = gen_extraction_snippet(ticker, company, section, date_iso)
        assert not placeholder(snippet), f"placeholder extraction {i}"
        prompt = f"Extract the ticker, section, and filing date from this filing snippet:\n{snippet}"
        items.append({"item_id": f"t1_extraction_{i:03d}", "prompt": prompt,
                      "gold": {"ticker": ticker, "section": section, "filing_date": date_iso},
                      "category": "extraction"})
        i += 1
    return items


def no_overlap(items):
    if not os.path.exists(TRAIN_PREF):
        return
    tq = {json.loads(l)["question"].strip() for l in open(TRAIN_PREF) if l.strip()}
    for it in items:
        assert it["prompt"].strip() not in tq, f"overlap {it['item_id']}"


def write(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-router", type=int, default=80)
    ap.add_argument("--n-extraction", type=int, default=80)
    args = ap.parse_args()

    router = build_router(args.n_router)
    extraction = build_extraction(args.n_extraction)

    rbal = Counter(it["gold"] for it in router)
    assert max(rbal.values()) - min(rbal.values()) <= 1, f"router imbalance {rbal}"
    no_overlap(router); no_overlap(extraction)
    # gold-as-output self-consistency
    assert all(SC.score("router", it["gold"], it["gold"], track="track1")[0] == "PASS" for it in router)
    assert all(SC.score("extraction", json.dumps(it["gold"]), it["gold"], track="track1")[0] == "PASS"
               for it in extraction)

    write(os.path.join(OUT_DIR, "track1_router.jsonl"), router)
    write(os.path.join(OUT_DIR, "track1_extraction.jsonl"), extraction)
    print(f"router:     {len(router)} items balance={dict(rbal)}")
    print(f"extraction: {len(extraction)} items sections={dict(Counter(it['gold']['section'] for it in extraction))}")
    print(f"-> {OUT_DIR}/track1_router.jsonl, track1_extraction.jsonl")


if __name__ == "__main__":
    sys.exit(main())
