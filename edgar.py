"""
Delta Filing - SEC EDGAR Data Fetcher
=====================================
Fetches and parses SEC filings (10-K, 10-Q, 8-K) from EDGAR.

SEC EDGAR is free, no API key needed, but requires a User-Agent header
with your name and email (SEC policy to identify who's making requests).

Usage:
    python edgar.py
"""

import re
import os
import json
import hashlib
import time
import requests
from bs4 import BeautifulSoup


# SEC requires a User-Agent header identifying who you are.
# Replace with your actual name and email before running.
HEADERS = {
    "User-Agent": "Delta-Filing Research your-email@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Rate limit: SEC asks for max 10 requests/second
REQUEST_DELAY = 0.15  # seconds between requests

# Cache directory — stores downloaded filings locally
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(url: str) -> str:
    """Generate a cache file path from a URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, url_hash)


def _get(url: str) -> requests.Response:
    """Make a rate-limited GET request to SEC EDGAR."""
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


def _get_cached(url: str) -> str:
    """GET with file-based caching. Returns response text.

    First check checks if we already downloaded this URL.
    If yes, read from disk. If no, download and save.
    """
    path = _cache_path(url)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    resp = _get(url)
    text = resp.text

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return text


# --- Company Lookup ---

def get_cik(ticker: str) -> str:
    """Convert a stock ticker to SEC's CIK number.

    Example:
        get_cik("AAPL") → "0000320193"
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    text = _get_cached(url)
    data = json.loads(text)

    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker_upper:
            # CIK needs to be zero-padded to 10 digits
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in SEC database")


# --- Filing Index ---

def get_filings(ticker: str, filing_type: str = "10-K", count: int = 5) -> list[dict]:
    """Get recent filings for a company.

    Args:
        ticker: Stock ticker (e.g., "AAPL")
        filing_type: "10-K", "10-Q", or "8-K"
        count: Number of recent filings to return

    Returns:
        List of dicts with: filing_date, accession_number, filing_url

    Example:
        get_filings("AAPL", "10-K", count=3)
    """
    cik = get_cik(ticker)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    text = _get_cached(url)
    data = json.loads(text)

    recent = data["filings"]["recent"]
    filings = []

    for i in range(len(recent["form"])):
        if recent["form"][i] == filing_type:
            accession = recent["accessionNumber"][i].replace("-", "")
            accession_display = recent["accessionNumber"][i]
            filing_date = recent["filingDate"][i]
            primary_doc = recent["primaryDocument"][i]

            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik}/{accession}/{primary_doc}"
            )

            filings.append({
                "filing_date": filing_date,
                "accession_number": accession_display,
                "filing_url": filing_url,
                "form_type": filing_type,
            })

            if len(filings) >= count:
                break

    return filings


# --- Filing Content ---

def fetch_filing_html(filing_url: str) -> str:
    """Download the raw HTML of a filing (cached)."""
    return _get_cached(filing_url)


# --- Section Parsing ---

# 10-K section patterns (Item numbers and their common names)
SECTION_PATTERNS_10K = {
    "1": r"(?:Item\s*1[\.\s]*[\-—–]?\s*Business)",
    "1A": r"(?:Item\s*1A[\.\s]*[\-—–]?\s*Risk\s+Factors)",
    "1B": r"(?:Item\s*1B[\.\s]*[\-—–]?\s*Unresolved\s+Staff\s+Comments)",
    "2": r"(?:Item\s*2[\.\s]*[\-—–]?\s*Properties)",
    "3": r"(?:Item\s*3[\.\s]*[\-—–]?\s*Legal\s+Proceedings)",
    "5": r"(?:Item\s*5[\.\s]*[\-—–]?\s*Market)",
    "6": r"(?:Item\s*6[\.\s]*[\-—–]?\s*(?:\[Reserved\]|Selected))",
    "7": r"(?:Item\s*7[\.\s]*[\-—–]?\s*Management)",
    "7A": r"(?:Item\s*7A[\.\s]*[\-—–]?\s*Quantitative)",
    "8": r"(?:Item\s*8[\.\s]*[\-—–]?\s*Financial\s+Statements)",
    "9": r"(?:Item\s*9[\.\s]*[\-—–]?\s*Changes)",
}

SECTION_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data / Reserved",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
}


def parse_filing_sections(html: str) -> dict[str, str]:
    """Parse a 10-K filing HTML into named sections.

    Returns:
        Dict mapping section ID to clean text.
        Example: {"1A": "Risk Factors text...", "7": "MD&A text..."}

    Note:
        SEC filings have inconsistent formatting across companies and years.
        This parser handles common patterns but may not capture every filing
        perfectly. For a production system, you'd want more robust parsing
        (e.g., using SEC's XBRL structured data).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, and hidden elements
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    full_text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = [line.strip() for line in full_text.split("\n")]
    clean_text = "\n".join(line for line in lines if line)

    sections = {}

    # Find each section by its Item header pattern
    section_positions = []
    for section_id, pattern in SECTION_PATTERNS_10K.items():
        matches = list(re.finditer(pattern, clean_text, re.IGNORECASE))
        if matches:
            # Use the LAST match (earlier matches might be in table of contents)
            match = matches[-1]
            section_positions.append((match.start(), section_id))

    # Sort by position in document
    section_positions.sort(key=lambda x: x[0])

    # Extract text between consecutive section headers
    for i, (start_pos, section_id) in enumerate(section_positions):
        if i + 1 < len(section_positions):
            end_pos = section_positions[i + 1][0]
        else:
            # Last section: take next 50000 chars (arbitrary but safe)
            end_pos = start_pos + 50000

        section_text = clean_text[start_pos:end_pos].strip()

        # Basic cleanup: remove excessive whitespace
        section_text = re.sub(r"\n{3,}", "\n\n", section_text)

        if len(section_text) > 100:  # Skip empty/trivial sections
            sections[section_id] = section_text

    return sections


def get_section(ticker: str, filing_type: str, section_id: str,
                filing_index: int = 0) -> dict:
    """Convenience function: get a specific section from a company's filing.

    Args:
        ticker: Stock ticker (e.g., "AAPL")
        filing_type: "10-K" or "10-Q"
        section_id: e.g., "1A" for Risk Factors, "7" for MD&A
        filing_index: 0 = most recent, 1 = second most recent, etc.

    Returns:
        Dict with: company, filing_date, section_id, section_name, text
    """
    filings = get_filings(ticker, filing_type, count=filing_index + 1)
    if not filings:
        raise ValueError(f"No {filing_type} filings found for {ticker}")
    if filing_index >= len(filings):
        raise ValueError(f"Only {len(filings)} filings available, requested index {filing_index}")

    filing = filings[filing_index]
    html = fetch_filing_html(filing["filing_url"])
    sections = parse_filing_sections(html)

    if section_id not in sections:
        available = list(sections.keys())
        raise ValueError(
            f"Section '{section_id}' not found. Available sections: {available}"
        )

    return {
        "company": ticker.upper(),
        "filing_date": filing["filing_date"],
        "form_type": filing_type,
        "section_id": section_id,
        "section_name": SECTION_NAMES.get(section_id, section_id),
        "text": sections[section_id],
        "filing_url": filing["filing_url"],
    }


# --- Test it ---

if __name__ == "__main__":
    print("=" * 60)
    print("  Delta Filing — SEC EDGAR Test")
    print("=" * 60)

    # Test 1: Look up Apple's CIK
    print("\n[1] Looking up Apple's CIK...")
    cik = get_cik("AAPL")
    print(f"    AAPL CIK: {cik}")

    # Test 2: List recent 10-K filings
    print("\n[2] Fetching Apple's recent 10-K filings...")
    filings = get_filings("AAPL", "10-K", count=3)
    for f in filings:
        print(f"    {f['filing_date']} | {f['accession_number']}")

    # Test 3: Get Risk Factors from the most recent 10-K
    print("\n[3] Fetching Risk Factors (Item 1A) from latest 10-K...")
    result = get_section("AAPL", "10-K", "1A")
    text_preview = result["text"][:500]
    print(f"    Filing date: {result['filing_date']}")
    print(f"    Section: {result['section_name']}")
    print(f"    Text length: {len(result['text'])} characters")
    print(f"    Preview:\n    {text_preview}...")

    # Test 4: Also get MD&A
    print("\n[4] Fetching MD&A (Item 7) from latest 10-K...")
    result_mda = get_section("AAPL", "10-K", "7")
    print(f"    Text length: {len(result_mda['text'])} characters")
    print(f"    Preview:\n    {result_mda['text'][:500]}...")

    print("\n" + "=" * 60)
    print("  All tests passed. EDGAR data pipeline is working.")
    print("=" * 60)
