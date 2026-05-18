"""
Tests for diff.py — Change Detection
Run: python tests/test_diff.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from diff import split_into_paragraphs, classify_paragraph, compute_similarity, diff_sections


def test_split_double_newline():
    text = "First paragraph about revenue growth.\n\nSecond paragraph about risk factors and their implications on the business."
    result = split_into_paragraphs(text)
    assert len(result) == 2
    assert "revenue" in result[0]
    assert "risk" in result[1]

def test_split_single_newline_fallback():
    text = "Revenue grew 15% year over year to reach new highs in the domestic market.\nOperating expenses increased due to supply chain disruptions affecting margins significantly."
    result = split_into_paragraphs(text)
    assert len(result) == 2

def test_split_merges_subheaders():
    text = "Macroeconomic Risks\nThe company faces significant exposure to global economic conditions that could adversely affect demand.\n\nSupply Chain\nManufacturing is concentrated in regions subject to geopolitical instability and trade restrictions."
    result = split_into_paragraphs(text)
    assert any("Macroeconomic" in p and "global economic" in p for p in result)

def test_split_filters_short_fragments():
    text = "Short.\n\nThis is a longer paragraph that should be kept because it has real content about business operations and strategy."
    result = split_into_paragraphs(text)
    assert not any(p == "Short." for p in result)

def test_split_empty_input():
    assert split_into_paragraphs("") == []
    assert split_into_paragraphs("   \n\n   ") == []

def test_classify_legal():
    assert classify_paragraph("The company is involved in ongoing litigation regarding patent infringement.") == "legal"

def test_classify_financial():
    assert classify_paragraph("Revenue increased 15% to $245.1 billion driven by cloud growth and margin expansion.") == "financial"

def test_classify_risk():
    assert classify_paragraph("There is significant uncertainty regarding the impact of adverse market conditions.") == "risk"

def test_classify_operational():
    assert classify_paragraph("Supply chain disruptions in manufacturing facilities continue to affect production.") == "operational"

def test_classify_general():
    assert classify_paragraph("The company was founded in 1976 and is headquartered in Cupertino, California.") == "general"

def test_similarity_identical():
    text = "Revenue grew 15% year over year."
    assert compute_similarity(text, text) == 1.0

def test_similarity_different():
    score = compute_similarity(
        "Revenue grew 15% driven by cloud services expansion.",
        "The company faces litigation in multiple jurisdictions.",
    )
    assert score < 0.3

def test_similarity_modified():
    score = compute_similarity(
        "Revenue grew 15% to $245.1 billion driven by cloud growth.",
        "Revenue grew 18% to $289.3 billion driven by cloud and AI growth.",
    )
    assert 0.5 < score < 0.95

def test_diff_identical():
    text = "Revenue grew significantly in all segments.\n\nOperating margins improved year over year across the board with strong performance."
    section = {"text": text, "company": "TEST", "section_name": "MD&A", "filing_date": "2024-01-01"}
    result = diff_sections(section, section)
    assert result["summary"]["added"] == 0
    assert result["summary"]["removed"] == 0
    assert result["summary"]["modified"] == 0
    assert result["summary"]["unchanged"] > 0

def test_diff_added():
    old = "Revenue grew 15% driven by cloud services expansion and enterprise adoption."
    new = old + "\n\nThe company also expanded into AI services, generating $5 billion in new revenue streams from generative AI products."
    a = {"text": old, "company": "TEST", "section_name": "MD&A", "filing_date": "2024-01-01"}
    b = {"text": new, "company": "TEST", "section_name": "MD&A", "filing_date": "2025-01-01"}
    result = diff_sections(a, b)
    assert result["summary"]["added"] >= 1

def test_diff_removed():
    old = "Revenue grew 15% driven by cloud services and expansion.\n\nThe company maintained its legacy hardware division with stable but declining revenues over the period."
    new = "Revenue grew 15% driven by cloud services and expansion."
    a = {"text": old, "company": "TEST", "section_name": "MD&A", "filing_date": "2024-01-01"}
    b = {"text": new, "company": "TEST", "section_name": "MD&A", "filing_date": "2025-01-01"}
    result = diff_sections(a, b)
    assert result["summary"]["removed"] >= 1

def test_diff_output_structure():
    text = "Revenue grew significantly in all segments across domestic and international markets."
    section = {"text": text, "company": "AAPL", "section_name": "MD&A", "filing_date": "2024-01-01"}
    result = diff_sections(section, section)
    for key in ["company", "section", "filing_a_date", "filing_b_date", "summary", "added", "removed", "modified"]:
        assert key in result
    for key in ["total_paragraphs_before", "total_paragraphs_after", "added", "removed", "modified", "unchanged"]:
        assert key in result["summary"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"  [PASS] {t.__name__}"); passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}"); failed += 1
    print(f"\n  diff.py: {passed}/{passed+failed} passed")
