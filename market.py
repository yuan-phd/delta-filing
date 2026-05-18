"""
Delta Filing — Market Data Sources
====================================
Wrappers for financial market data:
  - yfinance: stock prices, ETF data (no key needed)
  - Finnhub: company news, insider trades, analyst ratings (free key)
  - Alpha Vantage: financial statements, ratios (free key)

Usage:
    python market.py
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


# ============================================================
#  yfinance — Stock prices and ETF data (no API key needed)
# ============================================================

import yfinance as yf


def get_stock_price(ticker: str, period: str = "1mo") -> dict:
    """Get recent stock price data.

    Args:
        ticker: Stock ticker (e.g., "AAPL")
        period: "1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"

    Returns:
        Dict with current price, change, and period high/low.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        raise ValueError(f"No price data found for {ticker}")

    current = hist["Close"].iloc[-1]
    period_start = hist["Close"].iloc[0]
    change_pct = ((current - period_start) / period_start) * 100
    high = hist["High"].max()
    low = hist["Low"].min()
    avg_volume = hist["Volume"].mean()

    return {
        "ticker": ticker,
        "current_price": round(current, 2),
        "period_start_price": round(period_start, 2),
        "change_pct": round(change_pct, 2),
        "period_high": round(high, 2),
        "period_low": round(low, 2),
        "avg_volume": int(avg_volume),
        "period": period,
    }


def get_key_metrics(ticker: str) -> dict:
    """Get key financial metrics for a company.

    Returns PE ratio, market cap, dividend yield, etc.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "ticker": ticker,
        "company_name": info.get("longName", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "forward_pe": info.get("forwardPE", "N/A"),
        "pb_ratio": info.get("priceToBook", "N/A"),
        "dividend_yield": info.get("dividendYield", "N/A"),
        "revenue": info.get("totalRevenue", "N/A"),
        "profit_margin": info.get("profitMargins", "N/A"),
        "roe": info.get("returnOnEquity", "N/A"),
        "debt_to_equity": info.get("debtToEquity", "N/A"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
    }


def compare_stocks(tickers: list[str], period: str = "3mo") -> list[dict]:
    """Compare price performance of multiple stocks."""
    results = []
    for ticker in tickers:
        try:
            data = get_stock_price(ticker, period)
            results.append(data)
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return results


# ============================================================
#  Finnhub — News, insider trades, analyst ratings
# ============================================================

import finnhub

_finnhub_client = None


def _get_finnhub() -> finnhub.Client:
    """Get or create Finnhub client."""
    global _finnhub_client
    if _finnhub_client is None:
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError(
                "FINNHUB_API_KEY not set. Get a free key at https://finnhub.io"
            )
        _finnhub_client = finnhub.Client(api_key=api_key)
    return _finnhub_client


def get_company_news(ticker: str, days: int = 7) -> list[dict]:
    """Get recent news articles about a company.

    Args:
        ticker: Stock ticker (e.g., "AAPL")
        days: Number of days to look back

    Returns:
        List of dicts with: date, headline, summary, source, url
    """
    client = _get_finnhub()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    news = client.company_news(ticker, _from=start, to=end)

    return [
        {
            "date": datetime.fromtimestamp(n["datetime"]).strftime("%Y-%m-%d"),
            "headline": n.get("headline", ""),
            "summary": n.get("summary", "")[:200],
            "source": n.get("source", ""),
            "url": n.get("url", ""),
        }
        for n in news[:10]  # Limit to 10 most recent
    ]


def get_insider_trades(ticker: str) -> list[dict]:
    """Get recent insider transactions.

    Returns list of insider buys/sells with amounts.
    """
    client = _get_finnhub()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    trades = client.stock_insider_transactions(ticker, start, end)
    data = trades.get("data", [])

    return [
        {
            "name": t.get("name", "Unknown"),
            "share": t.get("share", 0),
            "change": t.get("change", 0),
            "transaction_date": t.get("transactionDate", ""),
            "transaction_type": (
                "Buy" if (t.get("change", 0) or 0) > 0 else "Sell"
            ),
            "value": abs((t.get("change", 0) or 0) * (t.get("transactionPrice", 0) or 0)),
        }
        for t in data[:15]  # Limit to 15 most recent
    ]


def get_analyst_ratings(ticker: str) -> dict:
    """Get analyst consensus ratings and price targets."""
    client = _get_finnhub()

    # Recommendation trends (free tier)
    recs = client.recommendation_trends(ticker)
    latest = recs[0] if recs else {}

    # Price target (may require paid plan)
    try:
        target = client.price_target(ticker)
    except Exception:
        target = {}

    return {
        "ticker": ticker,
        "period": latest.get("period", "N/A"),
        "strong_buy": latest.get("strongBuy", 0),
        "buy": latest.get("buy", 0),
        "hold": latest.get("hold", 0),
        "sell": latest.get("sell", 0),
        "strong_sell": latest.get("strongSell", 0),
        "target_high": target.get("targetHigh", "N/A"),
        "target_low": target.get("targetLow", "N/A"),
        "target_mean": target.get("targetMean", "N/A"),
        "target_median": target.get("targetMedian", "N/A"),
    }


# ============================================================
#  Alpha Vantage — Financial statements
# ============================================================

import requests

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"


def _get_av(function: str, ticker: str) -> dict:
    """Make an Alpha Vantage API call."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "ALPHA_VANTAGE_API_KEY not set. Get a free key at "
            "https://www.alphavantage.co/support/#api-key"
        )

    params = {
        "function": function,
        "symbol": ticker,
        "apikey": api_key,
    }
    resp = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Alpha Vantage returns error messages in the response body
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")

    return data


def get_income_statement(ticker: str, quarters: int = 4) -> list[dict]:
    """Get quarterly income statement data.

    Returns revenue, net income, EPS for recent quarters.
    """
    data = _get_av("INCOME_STATEMENT", ticker)
    reports = data.get("quarterlyReports", [])[:quarters]

    return [
        {
            "quarter_end": r.get("fiscalDateEnding", ""),
            "revenue": r.get("totalRevenue", "N/A"),
            "gross_profit": r.get("grossProfit", "N/A"),
            "operating_income": r.get("operatingIncome", "N/A"),
            "net_income": r.get("netIncome", "N/A"),
            "eps": r.get("reportedEPS", "N/A"),
        }
        for r in reports
    ]


def get_balance_sheet(ticker: str) -> dict:
    """Get the most recent quarterly balance sheet."""
    data = _get_av("BALANCE_SHEET", ticker)
    reports = data.get("quarterlyReports", [])

    if not reports:
        return {"error": "No balance sheet data available"}

    r = reports[0]
    return {
        "quarter_end": r.get("fiscalDateEnding", ""),
        "total_assets": r.get("totalAssets", "N/A"),
        "total_liabilities": r.get("totalLiabilities", "N/A"),
        "total_equity": r.get("totalShareholderEquity", "N/A"),
        "cash": r.get("cashAndCashEquivalentsAtCarryingValue", "N/A"),
        "long_term_debt": r.get("longTermDebt", "N/A"),
        "current_assets": r.get("totalCurrentAssets", "N/A"),
        "current_liabilities": r.get("totalCurrentLiabilities", "N/A"),
    }


# ============================================================
#  Test all data sources
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Delta Filing — Market Data Test")
    print("=" * 60)

    # --- yfinance (no key needed) ---
    print("\n[1] yfinance: Apple stock price (last month)...")
    price = get_stock_price("AAPL", "1mo")
    print(f"    Price: ${price['current_price']}")
    print(f"    Change: {price['change_pct']}%")
    print(f"    Range: ${price['period_low']} — ${price['period_high']}")

    print("\n[2] yfinance: Apple key metrics...")
    metrics = get_key_metrics("AAPL")
    print(f"    Company: {metrics['company_name']}")
    print(f"    PE Ratio: {metrics['pe_ratio']}")
    print(f"    Market Cap: {metrics['market_cap']}")
    print(f"    Sector: {metrics['sector']}")

    print("\n[3] yfinance: Compare AAPL vs MSFT vs GOOGL (3 months)...")
    comp = compare_stocks(["AAPL", "MSFT", "GOOGL"], "3mo")
    for c in comp:
        if "error" not in c:
            print(f"    {c['ticker']}: {c['change_pct']:+.1f}%")
        else:
            print(f"    {c['ticker']}: {c['error']}")

    # --- Finnhub (needs key) ---
    if os.getenv("FINNHUB_API_KEY"):
        print("\n[4] Finnhub: Apple news (last 7 days)...")
        news = get_company_news("AAPL", days=7)
        for n in news[:3]:
            print(f"    [{n['date']}] {n['headline'][:80]}...")

        print("\n[5] Finnhub: Apple insider trades (last 90 days)...")
        trades = get_insider_trades("AAPL")
        for t in trades[:3]:
            print(f"    {t['name']}: {t['transaction_type']} "
                  f"({t['change']:+,} shares)")

        print("\n[6] Finnhub: Apple analyst ratings...")
        ratings = get_analyst_ratings("AAPL")
        print(f"    Buy: {ratings['strong_buy'] + ratings['buy']}")
        print(f"    Hold: {ratings['hold']}")
        print(f"    Sell: {ratings['sell'] + ratings['strong_sell']}")
        print(f"    Target: ${ratings['target_mean']}")
    else:
        print("\n[4-6] Finnhub: Skipped (FINNHUB_API_KEY not set)")

    # --- Alpha Vantage (needs key) ---
    if os.getenv("ALPHA_VANTAGE_API_KEY"):
        print("\n[7] Alpha Vantage: Apple income statement...")
        income = get_income_statement("AAPL", quarters=2)
        for q in income:
            print(f"    {q['quarter_end']}: Revenue={q['revenue']}, "
                  f"Net Income={q['net_income']}")

        print("\n[8] Alpha Vantage: Apple balance sheet...")
        bs = get_balance_sheet("AAPL")
        print(f"    Assets: {bs.get('total_assets', 'N/A')}")
        print(f"    Cash: {bs.get('cash', 'N/A')}")
        print(f"    Debt: {bs.get('long_term_debt', 'N/A')}")
    else:
        print("\n[7-8] Alpha Vantage: Skipped (ALPHA_VANTAGE_API_KEY not set)")

    print("\n" + "=" * 60)
    print("  Market data test complete.")
    print("=" * 60)
