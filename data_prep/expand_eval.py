#!/usr/bin/env python
"""Grow the locked eval suite (json, router, extraction) from ~80 to ~160 items
each, by APPENDING new unique items to the existing gated sets (same GPT-4o-mini
method). Keeps the already-gated 80 as a subset; balanced; deduped; gold-self-
consistent. Router is re-adjudicated separately (adjudicate_router.py).

  python data_prep/expand_eval.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorer as SC                 # noqa: E402
import gen_track1_eval as GE        # json: COMPANIES, TOOLS, gen_tool_query   # noqa: E402
import gen_track1_more as GM        # router/extraction generators            # noqa: E402

DE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_eval")
TARGET = 160
TRAIN_Q = {json.loads(l)["question"].strip()
           for l in open(os.path.join(os.path.dirname(DE), "training_data", "dpo_pytorch.jsonl"))}


def load(name):
    return [json.loads(l) for l in open(os.path.join(DE, name)) if l.strip()]


def write(name, items, idprefix):
    for i, it in enumerate(items):
        it["item_id"] = f"{idprefix}_{i:03d}"
    with open(os.path.join(DE, name), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def grow_json():
    items = load("track1_json.jsonl")
    seen = {it["prompt"].strip() for it in items}
    by = Counter(it["gold"]["tool"] for it in items)
    per = TARGET // len(GE.TOOLS)
    for ti, tool in enumerate(GE.TOOLS):
        a = 0
        while by[tool] < per and a < 8 * per:
            a += 1
            ticker, company = GE.COMPANIES[(ti * per + by[tool] + a) % len(GE.COMPANIES)]
            q = GE.gen_tool_query(tool, ticker, company)
            if q.strip() in seen or GE.has_placeholder(q):
                continue
            items.append({"item_id": "tmp", "prompt": q, "category": "function_call",
                          "gold": {"tool": tool, "arguments": {"ticker": ticker}}})
            seen.add(q.strip()); by[tool] += 1
    write("track1_json.jsonl", items, "t1_json")
    return Counter(it["gold"]["tool"] for it in items), len(items)


def grow_router():
    items = load("track1_router.jsonl")
    seen = {it["prompt"].strip() for it in items}
    by = Counter(it["gold"] for it in items)
    cats = list(GM.ROUTER_DEFS)
    per = TARGET // len(cats)
    for ci, cat in enumerate(cats):
        a = 0
        while by[cat] < per and a < 8 * per:
            a += 1
            company = GM.COMPANIES[(ci * per + by[cat] + a) % len(GM.COMPANIES)][1]
            q = GM.gen_router_query(cat, company)
            if q.strip() in seen or GM.placeholder(q):
                continue
            items.append({"item_id": "tmp", "prompt": q, "gold": cat, "category": "router"})
            seen.add(q.strip()); by[cat] += 1
    items.sort(key=lambda it: it["gold"])
    write("track1_router.jsonl", items, "t1_router")
    return Counter(it["gold"] for it in items), len(items)


def grow_extraction():
    items = load("track1_extraction.jsonl")
    seen = {it["prompt"].strip() for it in items}
    i = len(items)
    while len(items) < TARGET:
        ticker, company = GM.COMPANIES[i % len(GM.COMPANIES)]
        section = GM.SECTIONS[i % len(GM.SECTIONS)]
        date_iso = f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}"   # 2024 set (distinct from existing 2025)
        snippet = GM.gen_extraction_snippet(ticker, company, section, date_iso)
        i += 1
        if GM.placeholder(snippet):
            continue
        prompt = f"Extract the ticker, section, and filing date from this filing snippet:\n{snippet}"
        if prompt.strip() in seen:
            continue
        items.append({"item_id": "tmp", "prompt": prompt, "category": "extraction",
                      "gold": {"ticker": ticker, "section": section, "filing_date": date_iso}})
        seen.add(prompt.strip())
    write("track1_extraction.jsonl", items, "t1_extraction")
    return Counter(it["gold"]["section"] for it in items), len(items)


def verify():
    J = load("track1_json.jsonl"); R = load("track1_router.jsonl"); X = load("track1_extraction.jsonl")
    for name, data, st, key in [("json", J, "function_call", lambda d: json.dumps(d["gold"])),
                                ("router", R, "router", lambda d: d["gold"]),
                                ("extraction", X, "extraction", lambda d: json.dumps(d["gold"]))]:
        leak = sum(d["prompt"].strip() in TRAIN_Q for d in data)
        uniq = len({d["prompt"].strip() for d in data})
        sc = sum(SC.score(st, key(d), d["gold"], track="track1")[0] == "PASS" for d in data)
        assert leak == 0, f"{name} leakage {leak}"
        assert sc == len(data), f"{name} gold-self-consistency {sc}/{len(data)}"
        print(f"  {name}: n={len(data)} unique={uniq} leakage={leak} gold-self-consistency={sc}/{len(data)}")


if __name__ == "__main__":
    jb, jn = grow_json();  print(f"json -> {jn} tool_balance={dict(jb)}")
    rb, rn = grow_router(); print(f"router -> {rn} class_balance={dict(rb)}")
    xb, xn = grow_extraction(); print(f"extraction -> {xn} section_balance={dict(xb)}")
    print("verify:")
    verify()
    print("DONE. Next: re-adjudicate router (adjudicate_router.py), then GPU baseline gate.")
