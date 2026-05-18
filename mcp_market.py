"""
Delta Filing — Market Data MCP Server
=======================================
Wraps yfinance and Finnhub functions as MCP tools using FastMCP.

Same pattern as mcp_edgar.py — just a different data source.

Usage (as standalone, for testing):
    python mcp_market.py
"""

import json
from mcp.server.fastmcp import FastMCP
from market import (
    get_stock_price,
    get_key_metrics,
    compare_stocks,
    get_company_news,
    get_insider_trades,
    get_analyst_ratings,
)

mcp = FastMCP("delta-market")


@mcp.tool()
def stock_price(ticker: str, period: str = "1mo") -> str:
    """Get recent stock price data for a company.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, TSLA)
        period: Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y
    """
    data = get_stock_price(ticker, period)
    return json.dumps(data, indent=2)


@mcp.tool()
def company_metrics(ticker: str) -> str:
    """Get key financial metrics: PE ratio, market cap, revenue, margins, etc.

    Args:
        ticker: Stock ticker symbol
    """
    data = get_key_metrics(ticker)
    return json.dumps(data, indent=2)


@mcp.tool()
def compare_stock_performance(tickers: str, period: str = "3mo") -> str:
    """Compare price performance of multiple stocks over a time period.

    Args:
        tickers: Comma-separated ticker symbols (e.g., "AAPL,MSFT,GOOGL")
        period: Time period: 1mo, 3mo, 6mo, 1y
    """
    ticker_list = [t.strip() for t in tickers.split(",")]
    data = compare_stocks(ticker_list, period)
    return json.dumps(data, indent=2)


@mcp.tool()
def company_news(ticker: str, days: int = 7) -> str:
    """Get recent news articles about a company.

    Args:
        ticker: Stock ticker symbol
        days: Number of days to look back (default 7)
    """
    data = get_company_news(ticker, days)
    return json.dumps(data, indent=2)


@mcp.tool()
def insider_trades(ticker: str) -> str:
    """Get recent insider trading activity (buys and sells by executives).

    Args:
        ticker: Stock ticker symbol
    """
    data = get_insider_trades(ticker)
    return json.dumps(data, indent=2)


@mcp.tool()
def analyst_ratings(ticker: str) -> str:
    """Get analyst consensus ratings (buy/hold/sell) and price targets.

    Args:
        ticker: Stock ticker symbol
    """
    data = get_analyst_ratings(ticker)
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
