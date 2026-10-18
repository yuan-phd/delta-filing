#!/usr/bin/env python
"""Evaluation-suite audit (scorer-side, model-free) for the format-dissociation paper.

Runs the GPU-free parts of the Part-A audit for every task that has an eval file:
  #2 metric-can-capture-failure (synthetic bad outputs -> verdict distribution; degeneracy)
  #3 format-fragility (does realistic verbosity drift break the gate?)
  #5 class balance / n / leakage / gold self-consistency

Baseline accuracy (#1) and headroom (#4) need a GPU baseline eval -> run_baseline_audit.py.
"""
import json
import os
from collections import Counter

import scorer as SC

TRAIN_Q = {json.loads(l)["question"].strip()
           for l in open("training_data/dpo_pytorch.jsonl") if l.strip()}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else None


# per-task audit config: (file, scorer-task, gold-key for balance, degenerate constant raw,
#                         format-broken probes [raw], content-wrong probe raw, a real gold for probes)
def cfg_for(task):
    if task == "binary":
        return dict(file="data_eval/track1_binary.jsonl", st="binary",
                    balkey=lambda d: d["gold"],
                    degen="COMPLETE",
                    fb=["Based on the analysis, it is complete.", "The analysis is thorough."],
                    wrong="NEEDS_FOLLOW_UP", probe_gold="COMPLETE",
                    drift_ok="COMPLETE. Furthermore, the filing details...",
                    drift_lead="After careful review, COMPLETE")
    if task == "router":
        return dict(file="data_eval/track1_router.jsonl", st="router",
                    balkey=lambda d: d["gold"],
                    degen="FILING_ANALYSIS",
                    fb=["This is about analyzing a filing.", "I think it concerns a filing diff."],
                    wrong="OUT_OF_SCOPE", probe_gold="FILING_ANALYSIS",
                    drift_ok="FILING_ANALYSIS: the user asks about a specific 10-K",
                    drift_lead="Probably FILING_ANALYSIS")
    if task in ("json", "function_call"):
        g = {"tool": "get_filing_section", "arguments": {"ticker": "AAPL"}}
        return dict(file="data_eval/track1_json.jsonl", st="function_call",
                    balkey=lambda d: d["gold"]["tool"],
                    degen=json.dumps({"tool": "search_filings", "arguments": {"ticker": "AAPL"}}),
                    fb=["I will call get_filing_section for AAPL.", f"Sure: {json.dumps(g)}"],
                    wrong='{"tool":"stock_price","arguments":{"ticker":"AAPL"}}', probe_gold=g,
                    drift_ok=f"{json.dumps(g)} Here is why I chose it.",
                    drift_lead=f"Let me help: {json.dumps(g)}")
    if task == "extraction":
        g = {"ticker": "AAPL", "section": "Item 1A", "filing_date": "2025-10-28"}
        return dict(file="data_eval/track1_extraction.jsonl", st="extraction",
                    balkey=lambda d: d["gold"]["section"],
                    degen=json.dumps(g),
                    fb=["ticker AAPL, section Item 1A, date 2025-10-28",
                        f"Here you go: {json.dumps(g)}"],
                    wrong=json.dumps({**g, "ticker": "MSFT"}), probe_gold=g,
                    drift_ok=f"{json.dumps(g)} (extracted from the snippet)",
                    drift_lead=f"Sure: {json.dumps(g)}")
    raise ValueError(task)


def audit(task):
    c = cfg_for(task)
    data = load(c["file"])
    if data is None:
        print(f"\n### {task}: NO FILE ({c['file']}) — skipped")
        return
    st = c["st"]
    print("\n" + "=" * 80 + f"\n### TASK: {task}  (n={len(data)})\n" + "=" * 80)

    # ---- #5 ----
    bal = Counter(c["balkey"](d) for d in data)
    leak = sum(d["prompt"].strip() in TRAIN_Q for d in data)
    sc = sum(SC.score(st, d["gold"] if st in ("binary", "router") else json.dumps(d["gold"]),
                      d["gold"], track="track1")[0] == "PASS" for d in data)
    print(f"#5 n={len(data)} unique_prompts={len({d['prompt'] for d in data})} "
          f"balance={dict(bal)}")
    print(f"   leakage={leak}  gold-as-output PASS={sc}/{len(data)}")

    # ---- #2 ----
    for raw in c["fb"]:
        v = SC.score(st, raw, c["probe_gold"], track="track1")[0]
        print(f"#2 format-broken {raw[:42]!r:<46} -> {v}")
    v = SC.score(st, c["wrong"], c["probe_gold"], track="track1")[0]
    print(f"#2 content-wrong  {c['wrong'][:42]!r:<46} -> {v}")
    deg = Counter(SC.score(st, c["degen"], d["gold"], track="track1")[0] for d in data)
    floor = {"binary": 0.50, "router": 0.20}.get(task, 1.0 / len(data))
    print(f"#2 DEGENERATE (always one answer): acc={deg['PASS']/len(data):.3f} "
          f"format_break={deg['FORMAT_BREAK']/len(data):.3f}  random_floor~{floor:.2f}  {dict(deg)}")

    # ---- #3 ----
    vo, fo = SC.score(st, c["drift_ok"], c["probe_gold"], track="track1")
    vl, fl = SC.score(st, c["drift_lead"], c["probe_gold"], track="track1")
    print(f"#3 trailing-elaboration -> {vo} (format_pass={fo['format_pass']})")
    print(f"#3 leading-prose        -> {vl} (format_pass={fl['format_pass']})")


if __name__ == "__main__":
    for t in ["json", "router", "extraction"]:
        audit(t)
    print("\n(#1 baseline accuracy + #4 headroom: run_baseline_audit.py on GPU.)")
