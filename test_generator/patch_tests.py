"""
Delta Filing — Patch extended_tests.json
==========================================
Manually fix 5 quality issues identified during human review:

1. Review#2 (Boeing): label COMPLETE → NEEDS_FOLLOW_UP
   Reason: "projection" + "approximately" = soft/imprecise data, fails COMPLETE bar
2. Tool parsing#2 (NFLX "Change Detection"): replace section name
   Reason: "Change Detection" is not a real SEC filing section
3. Tool parsing#3 (CSCO Item 9A): replace empty number list with text keywords
   Reason: any([]) returns False → test would always fail
4. Replace Review#0 (IBM, NEEDS_FOLLOW_UP) with hallucinated-precision failure mode
   Reason: original Review#0 is generic "lacks depth" — we already have that pattern.
           Replacing with hallucinated-precision case fills a gap.
5. Add 1 high-quality COMPLETE Review case to restore 2C+3NEEDS balance
   Reason: after fix #1, distribution became 1C+4NEEDS. We need to test
           "model can correctly identify GOOD analysis" — handwritten high-quality case.

Usage:
    python patch_tests.py

Reads:  extended_tests.json
Writes: extended_tests.json (in-place, backup at extended_tests.json.bak)
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
JSON_PATH = HERE / "extended_tests.json"
BACKUP_PATH = HERE / "extended_tests.json.bak"


def main():
    if not JSON_PATH.exists():
        print(f"ERROR: {JSON_PATH} not found")
        return

    # Backup
    shutil.copy(JSON_PATH, BACKUP_PATH)
    print(f"Backup: {BACKUP_PATH}")

    with open(JSON_PATH) as f:
        data = json.load(f)

    changes = []

    # ---- Fix 1: Review#2 label ----
    r2 = data["Review"]["main"][2]
    if r2["name"] == "Change Detection for BA":
        old = r2["expected_label"]
        r2["expected_label"] = "NEEDS_FOLLOW_UP"
        r2["name"] = "Change Detection for BA → NEEDS_FOLLOW_UP (soft data)"
        changes.append(f"Review#2 label: {old} → NEEDS_FOLLOW_UP")
    else:
        print(f"WARN: Review#2 name unexpected: {r2['name']}")

    # ---- Fix 2: Tool parsing#2 section ----
    tp2 = data["Tool parsing"]["main"][2]
    if tp2["ticker"] == "NFLX" and tp2["section"] == "Change Detection":
        tp2["name"] = "NFLX MD&A"
        tp2["section"] = "Management's Discussion and Analysis"
        # Update check_keywords.section accordingly
        tp2["check_keywords"]["section"] = ["MD&A", "Management", "Discussion"]
        changes.append("Tool parsing#2: section 'Change Detection' → 'Management's Discussion and Analysis'")
    else:
        print(f"WARN: Tool parsing#2 unexpected: ticker={tp2.get('ticker')}, section={tp2.get('section')}")

    # ---- Fix 3: Tool parsing#3 empty number list ----
    tp3 = data["Tool parsing"]["main"][3]
    if tp3["ticker"] == "CSCO" and tp3["check_keywords"].get("number") == []:
        # Replace number key with text keywords from the content
        # Content says: "controls over financial reporting were deemed effective,
        # however, no specific operating margin was disclosed due to ongoing audits"
        tp3["check_keywords"]["number"] = ["effective", "audits", "controls"]
        # Rename the key to be honest — it's text, not numbers
        tp3["check_keywords"]["keyword"] = tp3["check_keywords"].pop("number")
        changes.append("Tool parsing#3: empty number list → text keywords ['effective','audits','controls']")
    else:
        print(f"WARN: Tool parsing#3 unexpected: ticker={tp3.get('ticker')}, number={tp3['check_keywords'].get('number')}")

    # ---- Fix 4: Replace Review#0 with hallucinated-precision case ----
    # Original Review#0 (IBM) is generic "lacks depth" — same failure mode as
    # Review#4 (Pfizer). Replace with hallucinated-precision to add variety.
    r0 = data["Review"]["main"][0]
    if r0["name"] == "Risk Factors Analysis for IBM":
        data["Review"]["main"][0] = {
            "name": "Hallucinated precision for IBM → NEEDS_FOLLOW_UP",
            "input_analysis": (
                "Original question: What are IBM's key risk factors?\n\n"
                "Analysis: IBM's 10-K (Item 1A) discloses risks with exact financial impact. "
                "Cloud revenue declined precisely 4.7382% in Q3 2024, contributing to a total "
                "revenue contraction of $1.847 billion. Operating margin compressed to 13.9421% "
                "from 14.0033% prior year. Management cited geopolitical exposure costing "
                "exactly $283,452,000 in restructuring charges across Q5 of fiscal 2024."
            ),
            "expected_label": "NEEDS_FOLLOW_UP"
        }
        changes.append("Review#0: replaced generic IBM analysis with hallucinated-precision failure mode")
    else:
        print(f"WARN: Review#0 unexpected: {r0['name']}")

    # ---- Fix 5: Add 1 high-quality COMPLETE Review to restore 2C+3NEEDS balance ----
    # Check current state: if we have exactly 1 COMPLETE, append a new one.
    labels = [t["expected_label"] for t in data["Review"]["main"]]
    if labels.count("COMPLETE") == 1:
        # Handwritten high-quality COMPLETE case — JPM regulatory risk analysis
        # Hits all criteria: section ref, multiple numbers, judgment, caveats, 4+ sentences
        new_complete = {
            "name": "Regulatory risk for JPM → COMPLETE",
            "input_analysis": (
                "Original question: What are JPMorgan's key regulatory risks?\n\n"
                "Analysis: JPMorgan's 10-K (Item 1A) identifies regulatory compliance as a "
                "material risk, with the bank holding $3.4 trillion in assets subject to "
                "Federal Reserve oversight. The CCAR stress test in 2024 required an SCB of "
                "3.3%, up from 2.9% prior year, restricting capital return flexibility. "
                "Operational risk reserves grew to $5.8 billion (Item 7A), reflecting "
                "anticipated penalties from ongoing OCC investigations into trading practices. "
                "However, the bank's CET1 ratio of 15.0% provides significant buffer above "
                "the 12.5% requirement, suggesting these risks are well-capitalized despite "
                "the elevated regulatory pressure."
            ),
            "expected_label": "COMPLETE"
        }
        data["Review"]["main"].append(new_complete)
        changes.append(f"Review: added handwritten COMPLETE case (JPM regulatory) — now 2C+3NEEDS")
    elif labels.count("COMPLETE") == 2:
        print("INFO: Review already has 2 COMPLETE — skipping fix 5")
    else:
        print(f"WARN: Review COMPLETE count unexpected: {labels.count('COMPLETE')}")

    # ---- Save ----
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print()
    print(f"Applied {len(changes)} changes:")
    for c in changes:
        print(f"  - {c}")

    # ---- Verify final distribution ----
    print()
    print("Final distribution:")
    total_main = 0
    total_stretch = 0
    for cat, content in data.items():
        m, s = len(content["main"]), len(content["stretch"])
        print(f"  {cat:<20} main={m}, stretch={s}")
        total_main += m
        total_stretch += s
    print(f"  {'-' * 40}")
    print(f"  Main:    {total_main}")
    print(f"  Stretch: {total_stretch}")
    print(f"  Grand:   {total_main + total_stretch}")
    print()
    print("Review label distribution:")
    labels = [t["expected_label"] for t in data["Review"]["main"]]
    print(f"  COMPLETE: {labels.count('COMPLETE')}")
    print(f"  NEEDS_FOLLOW_UP: {labels.count('NEEDS_FOLLOW_UP')}")


if __name__ == "__main__":
    main()
