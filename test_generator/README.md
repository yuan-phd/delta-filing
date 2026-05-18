# Delta Filing Test Generator

Generates 50 new capability tests (35 → 85) for evaluating fine-tuned adapters.

## Workflow

```
generate_extended_tests.py  →  extended_tests.json  →  merge_tests.py  →  test_85_unified.py
       (GPT-4o-mini)           (human review)          (validates)        (deploy to Kaggle)
```

## Setup

```bash
# 1. Make sure these are in the same directory:
#    - generate_extended_tests.py
#    - merge_tests.py
#    - test_35_unified.py  (the original, used as template)

# 2. Install deps (if not already)
pip install openai python-dotenv

# 3. Make sure OPENAI_API_KEY is set
echo "OPENAI_API_KEY=sk-..." > .env
# OR: export OPENAI_API_KEY=sk-...
```

## Run

```bash
# Step 1: Generate tests (calls GPT-4o-mini ~10 times, ~$0.05, ~2 min)
python generate_extended_tests.py

# This produces:
#   extended_tests.json     — Generated tests by category
#   generation_report.md    — Human-readable summary

# Step 2: Review the report
cat generation_report.md

# Pay attention to:
#   - Validation issues (duplicates, ticker collisions)
#   - Spot-check samples for each category
#   - Distribution summary

# Step 3: If satisfied, merge into final test file
python merge_tests.py

# This produces:
#   test_85_unified.py      — Final file, syntax-checked

# Step 4: Verify counts
grep -c "rec(" test_85_unified.py
# Should report 80+ rec() calls for main tests
```

## What gets generated

| Category | Existing | New | Stretch | Total |
|----------|----------|-----|---------|-------|
| Router | 7 | 5 | 0 | 12 |
| Tool calling | 10 | 5 | 1 | 16 |
| Tool parsing | 3 | 5 | 0 | 8 |
| Review | 3 | 5 | 0 | 8 |
| Synthesis | 2 | 6 | 0 | 8 |
| Section summary | 2 | 6 | 0 | 8 |
| Red flag | 2 | 6 | 0 | 8 |
| Change detection | 2 | 6 | 0 | 8 |
| Refusal | 2 | 1 | 1 | 4 |
| Robustness | 2 | 0 | 3 | 5 |
| **Total** | **35** | **45** | **5** | **85** |

**Main = 80 tests** (reported as primary score)
**Stretch = 5 tests** (reported separately — beyond training distribution)

## Quality controls

The generator enforces:
- Each test uses a different ticker (no repeats within a category)
- Avoids tickers already in the original 35 tests
- Schema validation per category (rejects invalid JSON)
- Static checks before merge (duplicates, missing fields)
- Python syntax check after merge

What it does NOT do:
- Verify tests actually discriminate between adapters (do that on Kaggle)
- Validate factual correctness of generated content (e.g. company numbers)
- Replace human spot-checking — always review `generation_report.md`

## Tuning

Edit `generate_extended_tests.py`:
- `TEMPERATURE`: 0.7 default. Lower (0.3) for more consistent style, higher (1.0) for diversity.
- `SEED`: 42 default. Change for different test variants.
- `TARGETS` dict: per-category counts.
- `EXISTING_TICKERS`: tickers to avoid (already used in original 35).
- `TICKER_POOL`: candidate companies for generation.

## On Kaggle

After merge, copy `test_85_unified.py` to a Kaggle notebook. Edit the `ADAPTERS`
dict at the top to point to your adapter paths. Run.

Expected runtime: ~10 sec/test × 85 tests × N adapters. For 5 adapters: ~70 min.

## Troubleshooting

**"OPENAI_API_KEY not set"** — put it in `.env` or `export` it.

**Validation issues in report** — review the JSON, manually edit if needed,
re-run `merge_tests.py` (it doesn't re-call the API).

**Syntax check fails** — likely a string-escaping bug in generated content
(e.g. a quote inside a query). Open `extended_tests.json`, fix the offending
field, re-run merge.

**Want different test counts** — edit `TARGETS` in generate script, re-run both.
