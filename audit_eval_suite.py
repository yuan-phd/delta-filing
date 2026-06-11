#!/usr/bin/env python
"""Eval-suite hygiene audit — re-derive every hygiene number from the data on disk
(no memory, no GPU). Output -> results/hygiene_audit/.

Components:
 (a) Degeneracy floor = best accuracy a DEGENERATE strategy can reach, by definition:
     max over {constant = most-common gold/class, empty output} of accuracy on the
     real gold set (scored with the real scorer). Reported per task.
 (b) Item hygiene: prompt uniqueness, gold-as-output self-consistency, train/eval
     leakage (eval prompts vs the 647 preference pairs AND the SFT data).
 (c) Class/coverage balance: router class counts, extraction field-value diversity,
     json tool-name coverage.

  python audit_eval_suite.py
"""
import json
import os
import re
from collections import Counter

import scorer as SC

ROOT = os.path.dirname(os.path.abspath(__file__))
DE = os.path.join(ROOT, "data_eval")
OUT = os.path.join(ROOT, "results", "hygiene_audit")
TASKS = {"json": ("track1_json.jsonl", "function_call"),
         "router": ("track1_router.jsonl", "router"),
         "extraction": ("track1_extraction.jsonl", "extraction")}
# Canonical degeneracy floors (re-derived). extraction CORRECTED 0.0 -> 0.0063
# (every gold unique -> best constant strategy scores 1/160 = 0.00625).
QUOTED_FLOOR = {"json": 0.013, "router": 0.20, "extraction": 0.0063}

# --- best-effort source-provenance recovery for the descriptive overlap readout ---
NAME_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "alphabet": "GOOGL",
    "google": "GOOGL", "amazon": "AMZN", "meta": "META", "tesla": "TSLA", "netflix": "NFLX",
    "disney": "DIS", "boeing": "BA", "jpmorgan": "JPM", "bank of america": "BAC",
    "walmart": "WMT", "costco": "COST", "coca-cola": "KO", "pepsico": "PEP", "mcdonald": "MCD",
    "nike": "NKE", "intel": "INTC", "salesforce": "CRM", "oracle": "ORCL", "adobe": "ADBE",
    "cisco": "CSCO", "ibm": "IBM", "qualcomm": "QCOM", "texas instruments": "TXN",
    "pfizer": "PFE", "merck": "MRK", "johnson & johnson": "JNJ", "unitedhealth": "UNH",
    "exxon": "XOM", "chevron": "CVX", "caterpillar": "CAT", "ford": "F", "general motors": "GM",
    "verizon": "VZ", "paypal": "PYPL", "visa": "V", "broadcom": "AVGO", "home depot": "HD",
}
SECTION_KW = [("item 1a", "1A"), ("risk factor", "1A"), ("item 7a", "7A"),
              ("quantitative and qualitative", "7A"), ("item 7", "7"), ("md&a", "7"),
              ("management's discussion", "7"), ("management’s discussion", "7"),
              ("item 8", "8"), ("financial statement", "8"), ("item 1b", "1B"),
              ("item 1", "1"), ("business", "1"), ("item 3", "3"), ("legal proceeding", "3")]
TICKER_UNIVERSE = set(NAME_TO_TICKER.values()) | {
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "XOM", "WMT", "KO",
    "BA", "NKE", "DIS", "CAT", "PFE", "AVGO", "CRM", "V", "HD", "AMD", "ORCL", "INTC"}


def rec_ticker(text):
    m = re.search(r'"company"\s*:\s*"([A-Z]{1,5})"', text)        # tool_result_parsing JSON
    if m:
        return m.group(1)
    for tok in re.findall(r"\b[A-Z]{1,5}\b", text):              # explicit ticker token
        if tok in TICKER_UNIVERSE:
            return tok
    low = text.lower()
    for name, tk in NAME_TO_TICKER.items():                      # company name
        if name in low:
            return tk
    return None


def rec_section(text):
    m = re.search(r'"section_name"\s*:\s*"([^"]+)"', text)
    s = (m.group(1).lower() if m else text.lower())
    for kw, sid in SECTION_KW:
        if kw in s:
            return sid
    return None


def load(name):
    return [json.loads(l) for l in open(os.path.join(DE, name)) if l.strip()]


def gold_key(task, gold):
    return gold if isinstance(gold, str) else json.dumps(gold, sort_keys=True)


def render_constant(task, gold):
    """Render a gold value as an output string the way a degenerate model would emit."""
    return gold if isinstance(gold, str) else json.dumps(gold)


def degeneracy_floor(task, st, data):
    n = len(data)
    golds = [d["gold"] for d in data]
    # most-common gold (exact) -> a constant strategy outputs it
    freq = Counter(gold_key(task, g) for g in golds)
    common_key, common_count = freq.most_common(1)[0]
    common_gold = next(g for g in golds if gold_key(task, g) == common_key)
    const_out = render_constant(task, common_gold)
    const_acc = sum(SC.score(st, const_out, d["gold"], track="track1")[0] == "PASS" for d in data) / n
    empty_acc = sum(SC.score(st, "", d["gold"], track="track1")[0] == "PASS" for d in data) / n
    floor = max(const_acc, empty_acc)
    return {"floor": floor, "constant_strategy_acc": const_acc, "empty_output_acc": empty_acc,
            "most_common_gold_count": common_count, "most_common_gold_freq": common_count / n,
            "best_strategy": "constant" if const_acc >= empty_acc else "empty"}


def leakage(prompts, ref_strings):
    """EXACT string match (after .strip()) — verbatim overlap only."""
    return sum(p.strip() in ref_strings for p in prompts)


def eval_ticker_section(task, item):
    g = item["gold"]
    if task == "extraction":
        return g["ticker"], rec_section(g["section"]) or g["section"]
    if task == "json":
        return g["arguments"].get("ticker"), None
    return rec_ticker(item["prompt"]), None       # router: from NL prompt


def provenance_readout():
    """DESCRIPTIVE (not a gate): per task, how many eval items share a ticker (and
    ticker x section) with a preference pair's SOURCE FILING. Domain overlap is by
    design (both derive from EDGAR fetches); this just quantifies it."""
    pref = [json.loads(l) for l in open(os.path.join(ROOT, "training_data", "dpo_pytorch.jsonl")) if l.strip()]
    pref_tk, pref_tk_sec = set(), set()
    n_tk = n_tk_sec = 0
    for r in pref:
        t = rec_ticker(r["question"]); s = rec_section(r["question"])
        if t:
            pref_tk.add(t); n_tk += 1
        if t and s:
            pref_tk_sec.add((t, s)); n_tk_sec += 1
    out = {"matching_criterion": "ticker (and ticker x section); section id via keyword map",
           "pref_provenance_coverage": {
               "ticker_recoverable": f"{n_tk}/{len(pref)}",
               "ticker_and_section_recoverable": f"{n_tk_sec}/{len(pref)}",
               "distinct_pref_tickers": len(pref_tk),
               "distinct_pref_ticker_section": len(pref_tk_sec)},
           "tasks": {}}
    for task, (fname, _) in TASKS.items():
        data = load(fname)
        share_tk = share_tk_sec = have_tk = have_sec = 0
        for it in data:
            t, s = eval_ticker_section(task, it)
            if t:
                have_tk += 1
                if t in pref_tk:
                    share_tk += 1
            if t and s:
                have_sec += 1
                if (t, s) in pref_tk_sec:
                    share_tk_sec += 1
        out["tasks"][task] = {
            "n": len(data), "eval_ticker_recoverable": have_tk,
            "share_ticker_with_pref": share_tk,
            "eval_ticker_section_recoverable": have_sec,
            "share_ticker_section_with_pref": (share_tk_sec if have_sec else None)}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    # reference text for leakage: 647 preference questions + SFT questions
    pref_q = {json.loads(l)["question"].strip()
              for l in open(os.path.join(ROOT, "training_data", "dpo_pytorch.jsonl")) if l.strip()}
    sft_q = set()
    sftc = os.path.join(ROOT, "training_data", "sft_data_combined.jsonl")
    if os.path.exists(sftc):
        sft_q |= {json.loads(l).get("question", "").strip() for l in open(sftc) if l.strip()}
    sfttrain = os.path.join(ROOT, "training_data", "sft_pytorch", "train.jsonl")
    if os.path.exists(sfttrain):
        for l in open(sfttrain):
            if not l.strip():
                continue
            msgs = json.loads(l).get("messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    sft_q.add(m["content"].strip())

    report = {"reference_sizes": {"pref_pairs": len(pref_q), "sft_questions": len(sft_q)}, "tasks": {}}
    for task, (fname, st) in TASKS.items():
        data = load(fname)
        n = len(data)
        prompts = [d["prompt"] for d in data]
        # (a)
        floor = degeneracy_floor(task, st, data)
        # (b)
        uniq = len(set(prompts))
        gsc = sum(SC.score(st, render_constant(task, d["gold"]), d["gold"], track="track1")[0] == "PASS"
                  for d in data)
        leak_pref = leakage(prompts, pref_q)
        leak_sft = leakage(prompts, sft_q)
        # (c)
        if task == "router":
            balance = dict(Counter(d["gold"] for d in data))
        elif task == "json":
            balance = {"tool_counts": dict(Counter(d["gold"]["tool"] for d in data)),
                       "tools_covered": len(set(d["gold"]["tool"] for d in data))}
        else:  # extraction
            balance = {"unique_tickers": len(set(d["gold"]["ticker"] for d in data)),
                       "unique_sections": len(set(d["gold"]["section"] for d in data)),
                       "unique_dates": len(set(d["gold"]["filing_date"] for d in data)),
                       "unique_full_gold": len(set(json.dumps(d["gold"], sort_keys=True) for d in data))}
        report["tasks"][task] = {
            "n": n, "degeneracy_floor": floor,
            "quoted_floor": QUOTED_FLOOR[task],
            "floor_matches_quoted": abs(floor["floor"] - QUOTED_FLOOR[task]) < 1e-3,
            "unique_prompts": uniq, "duplicate_prompts": n - uniq,
            "gold_self_consistency": f"{gsc}/{n}",
            "leakage_vs_pref_pairs": leak_pref, "leakage_vs_sft": leak_sft,
            "balance": balance,
        }

    report["leakage_matching_criterion"] = "EXACT string match after .strip() (verbatim overlap only)"
    report["near_duplicate_source_overlap"] = provenance_readout()
    json.dump(report, open(os.path.join(OUT, "suite_hygiene.json"), "w"), indent=2)
    # human-readable
    lines = ["EVAL-SUITE HYGIENE AUDIT (re-derived from data_eval/)", "=" * 60,
             f"leakage reference: {len(pref_q)} pref-pair questions + {len(sft_q)} SFT questions\n"]
    for task, r in report["tasks"].items():
        f = r["degeneracy_floor"]
        lines.append(f"### {task}  (n={r['n']})")
        lines.append(f"  (a) degeneracy floor = {f['floor']:.4f}  [{f['best_strategy']}: const={f['constant_strategy_acc']:.4f}, "
                     f"empty={f['empty_output_acc']:.4f}]  | quoted={r['quoted_floor']}  "
                     f"-> {'MATCH' if r['floor_matches_quoted'] else 'DIFFERS'}")
        lines.append(f"      most-common gold appears {f['most_common_gold_count']}x ({f['most_common_gold_freq']:.4f})")
        lines.append(f"  (b) unique_prompts={r['unique_prompts']}/{r['n']} (dups={r['duplicate_prompts']}); "
                     f"gold-self-consistency={r['gold_self_consistency']}; "
                     f"leakage pref={r['leakage_vs_pref_pairs']} sft={r['leakage_vs_sft']}")
        lines.append(f"  (c) balance: {r['balance']}\n")
    nd = report["near_duplicate_source_overlap"]
    lines.append("LEAKAGE CRITERION: " + report["leakage_matching_criterion"])
    lines.append("DESCRIPTIVE SOURCE-DOMAIN OVERLAP (not a gate; domain overlap is by design):")
    lines.append(f"  pref provenance coverage: ticker {nd['pref_provenance_coverage']['ticker_recoverable']}, "
                 f"ticker+section {nd['pref_provenance_coverage']['ticker_and_section_recoverable']} "
                 f"({nd['pref_provenance_coverage']['distinct_pref_tickers']} distinct tickers)")
    for task, t in nd["tasks"].items():
        lines.append(f"  {task}: {t['share_ticker_with_pref']}/{t['eval_ticker_recoverable']} items share a ticker; "
                     f"ticker+section shared = {t['share_ticker_section_with_pref']} "
                     f"(of {t['eval_ticker_section_recoverable']} with section)")
    open(os.path.join(OUT, "suite_hygiene.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {OUT}/suite_hygiene.json + suite_hygiene.txt")


if __name__ == "__main__":
    main()
