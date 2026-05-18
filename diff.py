"""
Delta Filing — Change Detection
================================
Compares the same section from two filings (e.g., this year's Risk Factors
vs last year's) and identifies what was added, removed, or modified.

This is the core "delta" functionality — the reason the project is named
what it is.

Usage:
    python diff.py
"""

from difflib import SequenceMatcher
from edgar import get_section


def split_into_paragraphs(text: str) -> list[str]:
    """Split section text into paragraphs.

    SEC filings vary in formatting. Some use double newlines between
    paragraphs, others use single newlines. This function handles both.

    Short lines (< 80 chars) that look like sub-headers (e.g.,
    "Macroeconomic and Industry Risks") are merged with the paragraph
    that follows them, so they don't become separate tiny fragments.
    """
    # First try double newline split
    raw_paragraphs = text.split("\n\n")

    # If that only gives us 1-2 chunks, the filing uses single newlines
    if len(raw_paragraphs) <= 2:
        raw_paragraphs = text.split("\n")

    # Merge sub-headers with the following paragraph
    # Sub-headers are short lines that don't end with a period
    merged = []
    pending_header = ""

    for p in raw_paragraphs:
        cleaned = p.strip()
        if not cleaned:
            continue

        # Looks like a sub-header: short, doesn't end with period
        if len(cleaned) < 80 and not cleaned.endswith("."):
            pending_header = cleaned + "\n"
        else:
            merged.append(pending_header + cleaned)
            pending_header = ""

    # Don't forget a trailing header without a following paragraph
    if pending_header:
        merged.append(pending_header.strip())

    # Filter out very short fragments
    paragraphs = [p for p in merged if len(p) >= 50]

    return paragraphs


def classify_paragraph(paragraph: str) -> str:
    """Classify a paragraph type for better reporting.

    Returns one of: 'risk', 'financial', 'legal', 'operational', 'general'
    This is a simple keyword-based classifier — the LLM will do deeper analysis later.
    """
    text_lower = paragraph.lower()

    if any(w in text_lower for w in ["litigation", "lawsuit", "legal proceedings",
                                      "regulatory", "compliance", "SEC", "court"]):
        return "legal"
    if any(w in text_lower for w in ["revenue", "income", "earnings", "cash flow",
                                      "margin", "operating", "profit", "loss"]):
        return "financial"
    if any(w in text_lower for w in ["risk", "uncertainty", "adverse", "volatile",
                                      "threat", "vulnerability"]):
        return "risk"
    if any(w in text_lower for w in ["supply chain", "manufacturing", "operations",
                                      "employees", "workforce", "facilities"]):
        return "operational"
    return "general"


def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute similarity ratio between two text strings (0.0 to 1.0)."""
    return SequenceMatcher(None, text_a, text_b).ratio()


def diff_sections(section_a: dict, section_b: dict,
                  similarity_threshold: float = 0.6) -> dict:
    """Compare two sections and identify changes.

    Args:
        section_a: Earlier filing section (from get_section)
        section_b: Later filing section (from get_section)
        similarity_threshold: Below this, paragraphs are considered different.
                             Above this, they're considered modified versions
                             of each other.

    Returns:
        Dict with:
            - summary: quick stats
            - added: paragraphs in B but not in A (new content)
            - removed: paragraphs in A but not in B (deleted content)
            - modified: paragraphs that exist in both but with changes
            - unchanged_count: number of paragraphs that didn't change
    """
    paragraphs_a = split_into_paragraphs(section_a["text"])
    paragraphs_b = split_into_paragraphs(section_b["text"])

    # Track which paragraphs have been matched
    matched_a = [False] * len(paragraphs_a)
    matched_b = [False] * len(paragraphs_b)

    modified = []
    unchanged_count = 0

    # Find matching/modified paragraphs
    # For each paragraph in A, find the best match in B
    for i, para_a in enumerate(paragraphs_a):
        best_match_idx = -1
        best_match_score = 0.0

        for j, para_b in enumerate(paragraphs_b):
            if matched_b[j]:
                continue

            score = compute_similarity(para_a, para_b)
            if score > best_match_score:
                best_match_score = score
                best_match_idx = j

        if best_match_score > 0.95:
            # Nearly identical — unchanged
            matched_a[i] = True
            matched_b[best_match_idx] = True
            unchanged_count += 1

        elif best_match_score > similarity_threshold:
            # Similar enough to be the same paragraph, but modified
            matched_a[i] = True
            matched_b[best_match_idx] = True
            modified.append({
                "type": classify_paragraph(paragraphs_b[best_match_idx]),
                "similarity": round(best_match_score, 2),
                "before": para_a[:300] + ("..." if len(para_a) > 300 else ""),
                "after": paragraphs_b[best_match_idx][:300] + (
                    "..." if len(paragraphs_b[best_match_idx]) > 300 else ""
                ),
            })

    # Unmatched paragraphs in A = removed
    removed = []
    for i, para in enumerate(paragraphs_a):
        if not matched_a[i]:
            removed.append({
                "type": classify_paragraph(para),
                "text": para[:300] + ("..." if len(para) > 300 else ""),
            })

    # Unmatched paragraphs in B = added
    added = []
    for j, para in enumerate(paragraphs_b):
        if not matched_b[j]:
            added.append({
                "type": classify_paragraph(para),
                "text": para[:300] + ("..." if len(para) > 300 else ""),
            })

    return {
        "company": section_b["company"],
        "section": section_b["section_name"],
        "filing_a_date": section_a["filing_date"],
        "filing_b_date": section_b["filing_date"],
        "summary": {
            "total_paragraphs_before": len(paragraphs_a),
            "total_paragraphs_after": len(paragraphs_b),
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
            "unchanged": unchanged_count,
        },
        "added": added,
        "removed": removed,
        "modified": modified,
    }


def print_diff_report(diff: dict):
    """Print a human-readable diff report."""
    s = diff["summary"]

    print(f"\n{'='*60}")
    print(f"  DELTA REPORT: {diff['company']} — {diff['section']}")
    print(f"  Comparing: {diff['filing_a_date']} → {diff['filing_b_date']}")
    print(f"{'='*60}")

    print(f"\n  Paragraphs: {s['total_paragraphs_before']} → {s['total_paragraphs_after']}")
    print(f"  Added:     {s['added']}")
    print(f"  Removed:   {s['removed']}")
    print(f"  Modified:  {s['modified']}")
    print(f"  Unchanged: {s['unchanged']}")

    if diff["added"]:
        print(f"\n{'─'*60}")
        print("  NEW CONTENT (added in later filing)")
        print(f"{'─'*60}")
        for i, item in enumerate(diff["added"], 1):
            print(f"\n  [{i}] Type: {item['type']}")
            print(f"      {item['text'][:200]}...")

    if diff["removed"]:
        print(f"\n{'─'*60}")
        print("  REMOVED CONTENT (was in earlier filing, now gone)")
        print(f"{'─'*60}")
        for i, item in enumerate(diff["removed"], 1):
            print(f"\n  [{i}] Type: {item['type']}")
            print(f"      {item['text'][:200]}...")

    if diff["modified"]:
        print(f"\n{'─'*60}")
        print("  MODIFIED CONTENT (same topic, wording changed)")
        print(f"{'─'*60}")
        for i, item in enumerate(diff["modified"], 1):
            print(f"\n  [{i}] Type: {item['type']} | Similarity: {item['similarity']}")
            print(f"      BEFORE: {item['before'][:150]}...")
            print(f"      AFTER:  {item['after'][:150]}...")

    print(f"\n{'='*60}\n")


# --- Test ---

if __name__ == "__main__":
    print("Delta Filing — Change Detection Test")
    print("Comparing Apple's two most recent 10-K Risk Factors...\n")
    print("This will download two filings from SEC. Takes about 30 seconds.\n")

    # Fetch Risk Factors from the two most recent 10-Ks
    print("[1] Fetching latest 10-K (Item 1A)...")
    section_new = get_section("AAPL", "10-K", "1A", filing_index=0)
    print(f"    Got: {section_new['filing_date']} ({len(section_new['text'])} chars)")

    print("[2] Fetching previous 10-K (Item 1A)...")
    section_old = get_section("AAPL", "10-K", "1A", filing_index=1)
    print(f"    Got: {section_old['filing_date']} ({len(section_old['text'])} chars)")

    print("[3] Computing diff...")
    diff = diff_sections(section_old, section_new)

    print_diff_report(diff)
