#!/usr/bin/env python
"""Dry-run the two-stage scorer (§6) on the Track-1 eval slice, printing the
intermediate values requested (§0 rule 5): format_pass, normalization_applied,
content_correct, verdict.

IMPORTANT: the `raw_output` strings here are HAND-CONSTRUCTED scorer probes, NOT
model generations. The real generations come from the Kaggle smoke test. The
probes are chosen to exercise every verdict path (PASS / FORMAT_BREAK /
REASONING_ERROR) and the normalize / prose-wrapped-JSON branches, against the
REAL gold from data_eval/. Outputs are written under *_dryrun/ so they are never
confused with real run results.

  python dryrun_scorer.py
"""
import json
import os

import scorer as SC
from eval_pref import score_and_persist

ROOT = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(ROOT, "data_eval", "track1_binary.jsonl")
JSN = os.path.join(ROOT, "data_eval", "track1_json.jsonl")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def binary_probe(item, k):
    """k-th probe variant for a binary item with real gold (symmetric gate)."""
    g = item["gold"]                       # COMPLETE or NEEDS_FOLLOW_UP
    other = "NEEDS_FOLLOW_UP" if g == "COMPLETE" else "COMPLETE"
    return [
        (g, "exact gold label -> PASS"),
        (f"{g}: lacks specific figures", "label + trailing reason -> PASS (trailing ok)"),
        (f"Based on the filing, {g}", "LEADING prose before label -> FORMAT_BREAK"),
        (f"  {g.lower()}\n", "lowercase + whitespace -> PASS (norm: case+ws)"),
        (other, "wrong-but-valid label -> REASONING_ERROR"),
    ][k]


def json_probe(item, k):
    g = item["gold"]
    tool, tick = g["tool"], g["arguments"]["ticker"]
    good = json.dumps({"tool": tool, "arguments": {"ticker": tick}})
    wrong = json.dumps({"tool": tool, "arguments": {"ticker": "WRONG"}})
    fenced = f"```json\n{good}\n```"
    return [
        (good, "clean valid JSON -> PASS"),
        (f"{good} note: retrieved.", "JSON first, trailing note -> PASS (trailing ok)"),
        (fenced, "fenced ```json block -> PASS (norm: fence)"),
        (f"Sure: {good}", "LEADING prose before JSON -> FORMAT_BREAK"),
        (wrong, "valid JSON, wrong ticker -> REASONING_ERROR"),
    ][k]


def run(task, items, probe_fn, track="track1"):
    print("\n" + "=" * 92)
    print(f"TASK = {task}   (n_items={len(items)})")
    print("=" * 92)
    hdr = (f"{'item_id':<14}{'probe':<46}{'strict':<7}{'lenient':<8}"
           f"{'lead_prose':<11}{'norm':<6}{'content':<9}{'verdict'}")
    print(hdr)
    print("-" * 110)
    probe_items = []
    for i, it in enumerate(items):
        raw, desc = probe_fn(it, i % 5)
        verdict, f = SC.score(task, raw, it["gold"], prompt=it.get("prompt"), track=track)
        cc = "None" if f["content_correct"] is None else str(f["content_correct"])
        print(f"{it['item_id']:<14}{desc:<46}{str(f['format_pass']):<7}"
              f"{str(f['lenient_format_pass']):<8}{str(f['leading_prose']):<11}"
              f"{str(f['normalization_applied']):<6}{cc:<9}{verdict}")
        probe_items.append({"item_id": it["item_id"], "prompt": it["prompt"],
                            "gold": it["gold"], "raw_output": raw})
    # also persist through the harness to verify generations+results land on disk
    gen_path = f"generations_dryrun/track1_dryrun_{task}.jsonl"
    res_path = f"results_dryrun/track1_dryrun_{task}.json"
    results, _ = score_and_persist("track1", "dryrun", 0, task, probe_items,
                                   gen_path, res_path, tokenizer=None)
    print(f"\n  -> wrote {gen_path}  ({len(probe_items)} rows) and {res_path}")
    print(f"     aggregate: accuracy={results['accuracy']:.2f}  "
          f"format_break_rate(strict)={results['format_break_rate']:.2f}  "
          f"format_break_rate(lenient)={results['format_break_rate_lenient']:.2f}  "
          f"leading_prose_rate={results['leading_prose_rate']:.2f}  "
          f"reasoning_error_rate={results['reasoning_error_rate']:.2f}  "
          f"pct_normalized={results['pct_normalized']:.2f}")
    return results


def gate_edge_cases():
    """Symmetry + delimiter edge cases proving the two gates are now matched."""
    print("\n" + "=" * 92)
    print("GATE EDGE CASES (symmetry + delimiter)")
    print("=" * 92)
    cases = [
        ("binary", "COMPLETED", "COMPLETE", "delimiter: COMPLETED != COMPLETE"),
        ("binary", "COMPLETE extra words", "COMPLETE", "label first + trailing -> PASS"),
        ("binary", "well, COMPLETE", "COMPLETE", "leading prose -> BREAK"),
        ("function_call", '{"tool":"x","arguments":{}} trailing',
         {"tool": "x", "arguments": {}}, "JSON first + trailing -> PASS"),
        ("function_call", 'Sure: {"tool":"x","arguments":{}}',
         {"tool": "x", "arguments": {}}, "leading prose before JSON -> BREAK"),
    ]
    for task, raw, gold, desc in cases:
        verdict, f = SC.score(task, raw, gold, track="track1")
        print(f"  [{task:<13}] {desc:<38} strict={str(f['format_pass']):<5} "
              f"lenient={str(f['lenient_format_pass']):<5} "
              f"lead_prose={str(f['leading_prose']):<5} -> {verdict}  raw={raw!r}")


def detail_examples(task, items, probe_fn, track="track1", k=3):
    """Print the full fields dict for the first k probes (deep intermediate view)."""
    print(f"\n--- {task}: full per-item fields for first {k} probes ---")
    for i, it in enumerate(items[:k]):
        raw, desc = probe_fn(it, i % 5)
        verdict, f = SC.score(task, raw, it["gold"], prompt=it.get("prompt"), track=track)
        print(f"\n[{it['item_id']}] probe = {desc}")
        print(f"  raw_output = {raw!r}")
        print(f"  gold       = {it['gold']}")
        print(f"  fields     = {json.dumps(f, default=str)}")
        print(f"  verdict    = {verdict}")


if __name__ == "__main__":
    binary = load(BIN)
    jsons = load(JSN)
    run("binary", binary, binary_probe)
    run("function_call", jsons, json_probe)
    gate_edge_cases()
    detail_examples("binary", binary, binary_probe, k=3)
    detail_examples("function_call", jsons, json_probe, k=3)
    print("\nDRY-RUN COMPLETE (scorer).")
