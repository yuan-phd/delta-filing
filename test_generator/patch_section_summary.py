"""
Delta Filing — Patch test_85_unified.py Section summary assertion
==================================================================
Fixes a bug where company names like "Pfizer Inc.", "Boeing Company",
"Netflix, Inc." are used in literal `in` matching against model output.
Models output "Pfizer" not "Pfizer Inc." → false negatives.

Fix: Match on first word of company name (re.split on whitespace/comma).

Usage:
    python patch_section_summary.py

Reads:  test_85_unified.py
Writes: test_85_unified.py (in-place, backup at test_85_unified.py.bak)

Verification: includes a dry-test simulating model outputs for all 8 tests.
"""

import re
import shutil
from pathlib import Path

HERE = Path(__file__).parent
TARGET = HERE / "test_85_unified.py"
BACKUP = HERE / "test_85_unified.py.bak"


OLD_BLOCK = '''        r = gen([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}], 500)
        ok = (any(k.lower() in r.lower() for k in SECTION_KW) and (t in r or n in r)
              and any(k in r.lower() for k in JUDGMENT_KW))
        rec("Section summary", t, ok)'''

NEW_BLOCK = '''        r = gen([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}], 500)
        # Match on first word of company name to handle "Pfizer Inc.", "Boeing Company", etc.
        company_first = re.split(r"[\\s,]", n)[0]
        ok = (any(k.lower() in r.lower() for k in SECTION_KW)
              and (t in r or company_first in r or n.lower() in r.lower())
              and any(k in r.lower() for k in JUDGMENT_KW))
        rec("Section summary", t, ok)'''


def dry_test():
    """Simulate plausible model outputs and verify assertion behavior."""
    SECTION_KW = ["Item 1A", "Item 1", "Item 7", "Item 7A", "Risk Factors", "MD&A",
                  "Management's Discussion", "risk factors", "Business"]
    JUDGMENT_KW = ["significant", "critical", "notable", "concern", "impact",
                   "important", "material", "substantial"]

    tests = [
        ("AAPL", "Apple", "Apple's 10-K Item 1A discloses significant supply chain risks..."),
        ("NVDA", "NVIDIA", "NVIDIA's MD&A discusses notable revenue from data center..."),
        ("WMT", "Walmart", "Walmart's Business section (Item 1) outlines critical retail segments..."),
        ("PFE", "Pfizer Inc.", "Pfizer faces material risks per Item 1A including patent expirations..."),
        ("BA", "Boeing Company", "Boeing's Item 1A risk factors and MD&A reveal significant production concerns..."),
        ("SBUX", "Starbucks Corporation", "Starbucks generates revenue from retail stores as noted in Item 1 Business; significant..."),
        ("TGT", "Target Corporation", "Target's MD&A section provides notable commentary on inventory levels..."),
        ("F", "Ford Motor Company", "Ford's 10-K does not contain Item 12; the question may refer to Item 7. Significant..."),
        ("NFLX", "Netflix, Inc.", "Netflix's Business section (Item 1) describes critical streaming operations..."),
    ]

    print("=" * 70)
    print("Dry test: simulating model outputs against NEW assertion")
    print("=" * 70)

    all_pass = True
    for t, n, r in tests:
        company_first = re.split(r"[\s,]", n)[0]

        # Apply NEW assertion logic
        section_ok = any(k.lower() in r.lower() for k in SECTION_KW)
        ticker_ok = t in r or company_first in r or n.lower() in r.lower()
        judgment_ok = any(k in r.lower() for k in JUDGMENT_KW)
        new_ok = section_ok and ticker_ok and judgment_ok

        # Apply OLD assertion logic (for comparison)
        old_ticker_ok = t in r or n in r
        old_ok = section_ok and old_ticker_ok and judgment_ok

        marker = "✓" if new_ok else "✗"
        change = ""
        if new_ok and not old_ok:
            change = "  ← FIXED (was failing under old assertion)"
        elif not new_ok:
            change = "  ← STILL FAILS — investigate"
            all_pass = False

        print(f"  {marker} {t:<6} company_first={company_first!r:<20} new={new_ok} old={old_ok}{change}")
        if not new_ok:
            print(f"      section={section_ok} ticker={ticker_ok} judgment={judgment_ok}")
            print(f"      response: {r}")

    print()
    if all_pass:
        print("  ALL TESTS PASS under new assertion.")
    else:
        print("  ⚠️  SOME TESTS STILL FAIL — review above.")
    print()
    return all_pass


def main():
    print("Running dry test first...")
    if not dry_test():
        print("Aborting patch — dry test indicates assertion is still wrong.")
        return

    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        return

    # Backup
    shutil.copy(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    with open(TARGET) as f:
        src = f.read()

    # Add `import re` if not present
    if "import re" not in src:
        src = src.replace(
            "import os, json, torch",
            "import os, json, re, torch",
        )
        print("Added 'import re' to top-level imports.")

    if OLD_BLOCK in src:
        new_src = src.replace(OLD_BLOCK, NEW_BLOCK)
        with open(TARGET, "w") as f:
            f.write(new_src)
        print(f"Patched: {TARGET}")
        print("  Section summary assertion now uses first-word matching.")
    elif NEW_BLOCK in src:
        print("Already patched (NEW_BLOCK found in source). No changes.")
    else:
        print("ERROR: Expected OLD_BLOCK not found in source. Manual inspection required.")
        print("Looking for fragment...")
        if "any(k.lower() in r.lower() for k in SECTION_KW) and (t in r or n in r)" in src:
            print("  Old single-line form found — patch script assumes multi-line form.")
        return

    # Verify the patched file still has valid Python syntax
    import ast
    try:
        with open(TARGET) as f:
            ast.parse(f.read())
        print("Syntax check: PASSED")
    except SyntaxError as e:
        print(f"Syntax check: FAILED — {e}")


if __name__ == "__main__":
    main()
