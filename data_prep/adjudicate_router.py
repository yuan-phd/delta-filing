#!/usr/bin/env python
"""Second-LLM adjudication of router gold labels (gpt-4o adjudicates gpt-4o-mini's
generated labels).

Conservative, publishable criterion: KEEP an item only if the adjudicator agrees
with the gold category AND does not flag it ambiguous. Drop contested/ambiguous.
Then top up to 16/class with new double-agreed items. Reports how many agreed /
were relabeled-away (contested) / ambiguous, and the final balance.

  python data_prep/adjudicate_router.py
"""
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorer as SC          # noqa: E402
import gen_track1_more as G  # noqa: E402

load_dotenv()
client = OpenAI()
ADJ_MODEL = "gpt-4o"         # stronger model as independent adjudicator
ROUTER_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data_eval", "track1_router.jsonl")
TARGET_PER_CLASS = 32

DEFS = ("FILING_ANALYSIS = about the content of ONE specific company's filing (single period); "
        "FILING_DIFF = comparing a company's filings across two time periods/years; "
        "COMPANY_DEEP_DIVE = a comprehensive multi-source analysis of a company; "
        "SIMPLE_QUERY = a simple factual lookup about filings (list filings, a filing date); "
        "OUT_OF_SCOPE = unrelated to SEC filings or finance.")
ADJ_SYS = ("You are a strict labeling adjudicator. Classify the user query into exactly one of "
           "five categories and say whether it is genuinely ambiguous (could reasonably be two+). "
           f"Categories: {DEFS} Respond ONLY as JSON: "
           '{"category": "<ONE>", "ambiguous": true|false}.')


def adjudicate(query):
    r = client.chat.completions.create(
        model=ADJ_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": ADJ_SYS},
                  {"role": "user", "content": query}])
    d = json.loads(r.choices[0].message.content)
    return d.get("category", "").strip().upper(), bool(d.get("ambiguous", False))


def main():
    items = [json.loads(l) for l in open(ROUTER_FILE)]
    agree, contested, ambiguous = [], [], []
    for it in items:
        cat, amb = adjudicate(it["prompt"])
        if amb:
            ambiguous.append((it, cat))
        elif cat == it["gold"]:
            agree.append(it)
        else:
            contested.append((it, cat))

    n = len(items)
    print(f"adjudicated {n} router items with {ADJ_MODEL}")
    print(f"  agree (kept)      : {len(agree)}")
    print(f"  contested (drop)  : {len(contested)}  e.g. " +
          "; ".join(f"{it['gold']}->{c}" for it, c in contested[:5]))
    print(f"  ambiguous (drop)  : {len(ambiguous)}  e.g. " +
          "; ".join(f"{it['gold']}?" for it, _ in ambiguous[:5]))
    contested_frac = (len(contested) + len(ambiguous)) / n
    print(f"  contested+ambiguous fraction: {contested_frac:.1%}")
    if contested_frac > 0.25:
        print("  WARNING: >25% contested/ambiguous -> router labels are noisy; "
              "recommend RECONSIDER router before using it.")

    # top up to 16/class with new double-agreed items
    from collections import Counter
    kept = list(agree)
    by_cat = Counter(it["gold"] for it in kept)
    seen = {it["prompt"].strip() for it in kept}
    cats = ["FILING_ANALYSIS", "FILING_DIFF", "COMPANY_DEEP_DIVE", "SIMPLE_QUERY", "OUT_OF_SCOPE"]
    added = Counter()
    for cat in cats:
        attempts = 0
        while by_cat[cat] < TARGET_PER_CLASS and attempts < 6 * TARGET_PER_CLASS:
            attempts += 1
            comp = G.COMPANIES[attempts % len(G.COMPANIES)][1]
            q = G.gen_router_query(cat, comp)
            if q.strip() in seen:
                continue
            adj_cat, amb = adjudicate(q)
            if (not amb) and adj_cat == cat:          # double-agreed
                kept.append({"item_id": f"t1_router_x{cat[:3]}{by_cat[cat]:02d}",
                             "prompt": q, "gold": cat, "category": "router"})
                seen.add(q.strip()); by_cat[cat] += 1; added[cat] += 1

    # renumber + write
    kept.sort(key=lambda it: it["gold"])
    for i, it in enumerate(kept):
        it["item_id"] = f"t1_router_{i:03d}"
    with open(ROUTER_FILE, "w") as f:
        for it in kept:
            f.write(json.dumps(it) + "\n")

    bal = Counter(it["gold"] for it in kept)
    uniq = len({it["prompt"].strip() for it in kept})
    ok = all(SC.score("router", it["gold"], it["gold"], track="track1")[0] == "PASS" for it in kept)
    print(f"\ntop-up added per class: {dict(added)}")
    print(f"final router set: n={len(kept)} unique={uniq} balance={dict(bal)} gold-self-consistency={ok}")
    print(f"-> {ROUTER_FILE}")


if __name__ == "__main__":
    main()
