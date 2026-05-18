# Extended Test Generation Report

Model: `gpt-4o-mini` | Temperature: 0.7 | Seed: 42

## Test counts by category

| Category | Main | Stretch | Total |
|----------|------|---------|-------|
| Router | 5 | 0 | 5 |
| Tool calling | 5 | 1 | 6 |
| Tool parsing | 5 | 0 | 5 |
| Review | 5 | 0 | 5 |
| Synthesis | 6 | 0 | 6 |
| Section summary | 6 | 0 | 6 |
| Red flag | 8 | 0 | 8 |
| Change detection | 6 | 0 | 6 |
| Refusal | 4 | 1 | 5 |
| Robustness | 2 | 3 | 5 |

## Validation issues

- [Tool calling#4] Uses existing ticker: AMZN (acceptable but not preferred)
- [Tool parsing#4] Uses existing ticker: AMD (acceptable but not preferred)
- [Section summary#5] Uses existing ticker: TSLA (acceptable but not preferred)

## Spot-check samples (1 per category)

### Router

```json
{
  "expected_category": "FILING_ANALYSIS",
  "query": "What are the environmental risks mentioned in the latest 10-K for BP (British Petroleum)?",
  "rationale": "Focuses on specific content within a filing; however, environmental risks can vary and be subjective."
}
```

### Tool calling

```json
{
  "query": "What's the P/E ratio for Alphabet Inc. instead of GOOG?",
  "tool": "company_metrics",
  "ticker": "GOOG"
}
```

### Tool parsing

```json
{
  "name": "TSN Risk Factors",
  "ticker": "TSN",
  "company_name": "Tyson Foods",
  "section": "Risk Factors",
  "filing_date": "2025-11-15",
  "content": "Tyson Foods faces potential impacts from grain price fluctuations, with corn prices expected to average $5.50 per bushel for the next fiscal year.",
  "check_keywords": {
    "ticker": [
      "TSN",
      "Tyson Foods"
    ],
    "section": [
      "Risk Factors",
      "item 1A"
    ],
    "date": [
      "2025",
      "November"
    ],
    "number": [
      "$5.50",
      "5.50"
    ]
  }
}
```

### Review

```json
{
  "name": "Risk Factors Analysis for IBM",
  "input_analysis": "Original question: What are IBM's key risk factors?\n\nAnalysis: IBM's 10-K filing highlights several risk factors, including competition in the technology sector, reliance on cloud computing, and potential cybersecurity threats. The company's international exposure could also impact revenue, especially in volatile markets. However, no specific data points or section references were provided, making the analysis somewhat vague.",
  "expected_label": "NEEDS_FOLLOW_UP"
}
```

### Synthesis

```json
{
  "ticker": "WFC",
  "company_name": "Wells Fargo & Company",
  "metrics": {
    "market_cap": 197100000000.0,
    "pe_ratio": 9.5,
    "revenue": 78420000000.0,
    "current_price": 45.6
  },
  "news": [
    {
      "headline": "Wells Fargo reports $5B profit, driven by one-time asset sale"
    },
    {
      "headline": "CEO sells $1M worth of stock"
    }
  ],
  "insider_trades": [
    {
      "name": "Charlie Scharf",
      "type": "Sale",
      "shares": 100000
    }
  ],
  "expected_numbers": [
    "197",
    "9.50",
    "78.42",
    "5B",
    "1M"
  ],
  "expected_topics": [
    "profit",
    "asset sale",
    "CEO",
    "insider sale",
    "concern"
  ]
}
```

### Section summary

```json
{
  "query": "What does Pfizer show in its latest 10-K about risks they face in Item 1A?",
  "ticker": "PFE",
  "company_name": "Pfizer Inc.",
  "section_type": "Item 1A"
}
```

### Red flag

```json
{
  "query": "What concerns might emerge from Boeing's recent filings regarding safety and production issues?",
  "ticker": "BA",
  "is_false_positive_test": false
}
```

### Change detection

```json
{
  "query": "What new financial risks did Ford mention in their 10-K for 2023 compared to 2022 that were previously not disclosed?",
  "ticker": "F",
  "section_type": "Risk Factors"
}
```

### Refusal

```json
{
  "query": "If I invest in CVCY, how will it perform compared to SYNH in the next year?",
  "refusal_type": "speculation"
}
```

### Robustness

```json
{
  "query": "hey, can u tell me about the risk factors for $WFC in the last 10-K?",
  "expected_tool": "get_filing_section",
  "expected_ticker": "WFC",
  "challenge_type": "informal"
}
```
