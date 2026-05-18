"""
Delta Filing — EDGAR MCP Server
================================
Wraps SEC EDGAR functionality as MCP tools using FastMCP.

FastMCP is the standard way to build MCP servers — it auto-generates
the tool schemas from your Python function signatures and docstrings.
No manual inputSchema definition needed.

Run as standalone server:
    python mcp_edgar.py

The server communicates via stdin/stdout (stdio transport).
An MCP client connects to it by launching this script as a subprocess.
"""

import json
from mcp.server.fastmcp import FastMCP

from edgar import get_filings, get_section
from diff import diff_sections


# Create the MCP server
mcp = FastMCP("delta-edgar")


# --- Tools ---
# With FastMCP, each tool is just a decorated Python function.
# The schema (name, description, parameters) is auto-generated from
# the function name, docstring, and type hints.

@mcp.tool()
def search_filings(ticker: str, filing_type: str = "10-K", count: int = 5) -> str:
    """List recent SEC filings for a company.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, TSLA)
        filing_type: Type of filing: 10-K, 10-Q, or 8-K
        count: Number of recent filings to return
    """
    filings = get_filings(ticker, filing_type, count)
    return json.dumps(filings, indent=2)


@mcp.tool()
def get_filing_section(ticker: str, section_id: str,
                       filing_type: str = "10-K", filing_index: int = 0) -> str:
    """Get the text of a specific section from a company's SEC filing.

    Common sections: 1 (Business), 1A (Risk Factors),
    7 (MD&A), 8 (Financial Statements).

    Args:
        ticker: Stock ticker symbol
        section_id: Section ID: 1, 1A, 1B, 2, 3, 5, 6, 7, 7A, 8, 9
        filing_type: 10-K or 10-Q
        filing_index: 0 = most recent, 1 = previous, 2 = two filings ago
    """
    section = get_section(ticker, filing_type, section_id, filing_index)

    text = section["text"]
    if len(text) > 8000:
        text = text[:8000] + f"\n\n[...truncated, {len(section['text'])} total chars]"

    result = {
        "company": section["company"],
        "filing_date": section["filing_date"],
        "section_name": section["section_name"],
        "text_length": len(section["text"]),
        "text": text,
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def diff_filing_sections(ticker: str, section_id: str,
                         filing_type: str = "10-K") -> str:
    """Compare the same section from two consecutive filings to detect changes.

    Returns added, removed, and modified paragraphs.
    This is the core 'delta' detection function.

    Args:
        ticker: Stock ticker symbol
        section_id: Section to compare (e.g., 1A for Risk Factors)
        filing_type: 10-K or 10-Q
    """
    section_new = get_section(ticker, filing_type, section_id, filing_index=0)
    section_old = get_section(ticker, filing_type, section_id, filing_index=1)
    diff = diff_sections(section_old, section_new)
    return json.dumps(diff, indent=2)


# --- Run ---
if __name__ == "__main__":
    mcp.run(transport="stdio")
