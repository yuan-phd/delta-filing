"""
Tests for edgar.py — SEC EDGAR parsing logic
Tests parsing functions only — no network calls.
Run: python tests/test_edgar.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from edgar import parse_filing_sections


SAMPLE_10K_HTML = """
<html><body>
<h2>Item 1A. Risk Factors</h2>
<p>The Company is subject to various risks including market competition,
regulatory changes, and macroeconomic conditions. International operations
account for approximately 60% of revenue.</p>
<p>Supply chain concentration in certain regions creates operational risk.</p>

<h2>Item 7. Management's Discussion and Analysis</h2>
<p>Revenue increased 15% to $245.1 billion. The growth was primarily driven
by cloud services which grew 23% year over year.</p>
<p>Operating income grew 24% to $109.4 billion reflecting improved margins.</p>

<h2>Item 8. Financial Statements</h2>
<p>See consolidated financial statements beginning on page F-1.</p>
</body></html>
"""

SAMPLE_MINIMAL_HTML = """
<html><body>
<p>Item 1A. Risk Factors</p>
<p>The company faces risks related to competition.</p>
<p>Item 7. Management Discussion and Analysis</p>
<p>Revenue grew year over year.</p>
</body></html>
"""


def test_parse_extracts_sections():
    """parse_filing_sections should extract named sections."""
    sections = parse_filing_sections(SAMPLE_10K_HTML)
    assert isinstance(sections, dict)
    assert len(sections) > 0

def test_parse_finds_risk_factors():
    """Should find Item 1A / Risk Factors."""
    sections = parse_filing_sections(SAMPLE_10K_HTML)
    # Check if any key contains "1A" or "Risk"
    found = any("1A" in k or "Risk" in k.lower() for k in sections)
    assert found, f"Risk Factors not found in: {list(sections.keys())}"

def test_parse_finds_mda():
    """Should find Item 7 / MD&A."""
    sections = parse_filing_sections(SAMPLE_10K_HTML)
    found = any("7" in k or "Management" in k or "MD&A" in k for k in sections)
    assert found, f"MD&A not found in: {list(sections.keys())}"

def test_parse_content_not_empty():
    """Extracted sections should have non-empty content."""
    sections = parse_filing_sections(SAMPLE_10K_HTML)
    for name, content in sections.items():
        assert len(content.strip()) > 0, f"Section '{name}' is empty"

def test_parse_empty_html():
    """Empty HTML should return empty dict or not crash."""
    result = parse_filing_sections("<html><body></body></html>")
    assert isinstance(result, dict)

def test_parse_risk_content_has_keywords():
    """Risk Factors section should contain risk-related content."""
    sections = parse_filing_sections(SAMPLE_10K_HTML)
    risk_content = ""
    for k, v in sections.items():
        if "1A" in k or "Risk" in k.lower():
            risk_content = v
            break
    assert "risk" in risk_content.lower() or "competition" in risk_content.lower()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"  [PASS] {t.__name__}"); passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}"); failed += 1
    print(f"\n  edgar.py: {passed}/{passed+failed} passed")
