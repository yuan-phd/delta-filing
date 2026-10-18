"""Scorer v2 — three-tier verdict (FROZEN-READY, unit-tested; not yet preregistered).

Tiers: FORMAT_BREAK / SURFACE_FORM / SEMANTIC_ERROR (+ PASS).
  - PASS          : format ok AND strict content correct.
  - SURFACE_FORM  : format ok, strict content fails, but passes the canonicalized
                    comparison (a register mismatch we can normalize away).
  - SEMANTIC_ERROR: format ok, fails even canonicalized (genuine wrong content).
  - FORMAT_BREAK  : fails the format gate.

Primary metric  = SEMANTIC accuracy = (PASS + SURFACE_FORM) / n.
Secondary metric = STRICT accuracy   = PASS / n.   (Both reported in all outputs.)

FROZEN canonicalization rules (extraction section field only):
  section identifier = first match of \\d+[A-Z]? , case-folded; appended titles
  ignored (e.g. "Item 8" == "8" == "Item 8, Risk Factors"). If no identifier is
  present (e.g. "Risk Factors"), it stays a SEMANTIC_ERROR.

This module REUSES scorer.py (format gates, parsing, value/date normalization);
it does not modify it. Other tasks (router/json) have no surface-form rule, so
their content errors map directly to SEMANTIC_ERROR.
"""
import re

import scorer as SC

TIER_PASS = "PASS"
TIER_SURFACE = "SURFACE_FORM"
TIER_SEMANTIC = "SEMANTIC_ERROR"
TIER_FORMAT_BREAK = "FORMAT_BREAK"

CANON_RULES = {
    "extraction.section": r"first \d+[A-Z]? , case-folded, appended titles ignored",
}


def section_id(s):
    """FROZEN: first \\d+[A-Z]? identifier, upper-cased; None if absent."""
    if s is None:
        return None
    m = re.search(r"\d+[A-Z]?", str(s).upper())
    return m.group(0) if m else None


def _canon_extraction_correct(raw, gold):
    obj = SC.try_json_parse(SC.first_json_span(raw))
    if not isinstance(obj, dict):
        return False
    t = (SC._norm_value(obj.get("ticker")) == SC._norm_value(gold["ticker"])
         if obj.get("ticker") is not None else False)
    d = (SC._norm_date(obj.get("filing_date")) == SC._norm_date(gold["filing_date"])
         if obj.get("filing_date") is not None else False)
    sp, sg = section_id(obj.get("section")), section_id(gold["section"])
    s = sp is not None and sp == sg
    return t and d and s


def score_v2(task, raw, gold, prompt=None, track="track1"):
    """Return (tier, fields). Reuses scorer.score for the strict verdict, then
    refines REASONING_ERROR into SURFACE_FORM vs SEMANTIC_ERROR."""
    verdict, fields = SC.score(task, raw, gold, prompt=prompt, track=track)
    if verdict == SC.VERDICT_FORMAT_BREAK:
        tier = TIER_FORMAT_BREAK
    elif verdict == SC.VERDICT_PASS:
        tier = TIER_PASS
    else:  # REASONING_ERROR -> surface vs semantic
        if task == "extraction" and _canon_extraction_correct(raw, gold):
            tier = TIER_SURFACE
        else:
            tier = TIER_SEMANTIC
    fields = dict(fields)
    fields["tier"] = tier
    fields["strict_pass"] = (tier == TIER_PASS)
    fields["semantic_pass"] = tier in (TIER_PASS, TIER_SURFACE)
    return tier, fields


def aggregate(rows):
    """rows: list of fields dicts (each with 'tier'). Returns the metric summary."""
    from collections import Counter
    c = Counter(r["tier"] for r in rows)
    n = len(rows) or 1
    return {
        "n": len(rows),
        "semantic_accuracy": (c[TIER_PASS] + c[TIER_SURFACE]) / n,   # PRIMARY
        "strict_accuracy": c[TIER_PASS] / n,                          # secondary
        "pass": c[TIER_PASS], "surface_form": c[TIER_SURFACE],
        "semantic_error": c[TIER_SEMANTIC], "format_break": c[TIER_FORMAT_BREAK],
    }
