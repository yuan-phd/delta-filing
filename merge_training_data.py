"""
Delta Filing — Merge & Prepare Training Data (PyTorch version)
================================================================
Merges all training data sources, balances categories, and outputs
in HuggingFace chat format for use with TRL's SFTTrainer.

Input files (in training_data/):
  sft_data_combined.jsonl         — 351 analysis Q&A pairs
  dpo_data_combined.jsonl         — 351 analysis DPO pairs
  sft_supplementary_fixed.jsonl   — supplementary (tool calling, router, etc.)
  dpo_tool_calling_fixed.jsonl    — tool calling + tool result DPO pairs

Output (in training_data/sft_pytorch/):
  train.jsonl, valid.jsonl, test.jsonl
  Each line: {"messages": [{"role": "system", "content": "..."}, ...]}

Usage:
    python merge_training_data.py
"""

import json
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path("training_data")
OUTPUT_DIR = DATA_DIR / "sft_pytorch"
OUTPUT_DIR.mkdir(exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "JPM",
    "V", "JNJ", "WMT", "PFE", "KO", "DIS", "NKE", "HD", "COST",
    "NFLX", "CRM", "ADBE", "ORCL", "AMD", "INTC", "BA", "CAT",
]

TICKER_TO_NAME = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google/Alphabet",
    "NVDA": "NVIDIA", "META": "Meta", "AMZN": "Amazon", "TSLA": "Tesla",
    "JPM": "JPMorgan", "V": "Visa", "JNJ": "Johnson & Johnson",
    "WMT": "Walmart", "PFE": "Pfizer", "KO": "Coca-Cola", "DIS": "Disney",
    "NKE": "Nike", "HD": "Home Depot", "COST": "Costco", "NFLX": "Netflix",
    "CRM": "Salesforce", "ADBE": "Adobe", "ORCL": "Oracle", "AMD": "AMD",
    "INTC": "Intel", "BA": "Boeing", "CAT": "Caterpillar",
}

SYSTEM_PROMPT = (
    "You are Delta Filing, a financial analyst AI specializing in SEC filing analysis. "
    "You analyze 10-K and 10-Q filings, detect year-over-year changes in risk disclosures, "
    "track management guidance accuracy, and flag potential red flags. "
    "Always reference specific filing sections, cite specific numbers and dates, "
    "provide analytical judgment, and note caveats. "
    "Do not give investment advice or predict stock prices."
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("=" * 60)
    print("  Delta Filing — Merge Training Data (PyTorch)")
    print("=" * 60)

    # --- Load analysis data ---
    print("\n[1] Loading analysis data...")
    analysis_raw = load_jsonl(DATA_DIR / "sft_data_combined.jsonl")
    print(f"    Loaded {len(analysis_raw)} analysis examples")

    # Convert to chat format
    analysis_chat = []
    for ex in analysis_raw:
        analysis_chat.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ],
            "category": ex.get("category", "analysis"),
        })

    # --- Load supplementary data (already in chat format) ---
    print("[2] Loading supplementary data...")
    supplementary = load_jsonl(DATA_DIR / "sft_supplementary_fixed.jsonl")
    print(f"    Loaded {len(supplementary)} supplementary examples")

    # --- Subsample tool_calling to 350 ---
    tool_calling = [ex for ex in supplementary if ex.get("category") == "tool_calling"]
    rest = [ex for ex in supplementary if ex.get("category") != "tool_calling"]
    sampled_tc = random.sample(tool_calling, min(350, len(tool_calling)))
    supplementary = rest + sampled_tc
    print(f"    After subsample: tool_calling {len(tool_calling)}→{len(sampled_tc)}, "
          f"total={len(supplementary)}")

    # --- Augment router data ---
    router_examples = [ex for ex in supplementary if ex.get("category") == "router"]
    augmented = []
    for ex in router_examples:
        user_msg = ex["messages"][1]["content"]
        for ticker in TICKERS:
            if ticker in user_msg:
                for _ in range(2):
                    new_ticker = random.choice([t for t in TICKERS if t != ticker])
                    new_name = TICKER_TO_NAME.get(new_ticker, new_ticker)
                    old_name = TICKER_TO_NAME.get(ticker, ticker)
                    new_user = user_msg.replace(ticker, new_ticker).replace(old_name, new_name)
                    augmented.append({
                        "messages": [
                            ex["messages"][0],
                            {"role": "user", "content": new_user},
                            ex["messages"][2],
                        ],
                        "category": "router",
                    })
                break
    supplementary = supplementary + augmented
    print(f"    Router augmentation: +{len(augmented)}, total={len(supplementary)}")

    # --- Merge ---
    all_data = analysis_chat + supplementary
    print(f"\n[3] Merged total: {len(all_data)} examples")

    # Category breakdown
    cats = {}
    for ex in all_data:
        cat = ex.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    print("    Category breakdown:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        pct = count * 100 // len(all_data)
        print(f"      {cat}: {count} ({pct}%)")

    # --- Shuffle and split ---
    random.shuffle(all_data)

    # Remove category field (not needed for training)
    clean_data = [{"messages": ex["messages"]} for ex in all_data]

    n = len(clean_data)
    train_end = int(n * 0.80)
    valid_end = int(n * 0.90)

    splits = {
        "train": clean_data[:train_end],
        "valid": clean_data[train_end:valid_end],
        "test": clean_data[valid_end:],
    }

    print(f"\n[4] Split: train={len(splits['train'])}, "
          f"valid={len(splits['valid'])}, test={len(splits['test'])}")

    # --- Save ---
    for name, data in splits.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"    Saved {path}")

    # --- Verify ---
    print("\n[5] Verification...")
    sample = clean_data[0]
    # Check no thinking tags in the raw data
    text = json.dumps(sample)
    has_think = "<think>" in text or "</think>" in text
    print(f"    Thinking tags in data: {has_think}")
    assert not has_think, "FATAL: thinking tags found in training data"
    print(f"    Sample user: {sample['messages'][1]['content'][:80]}...")
    print(f"    Sample asst: {sample['messages'][2]['content'][:80]}...")

    # --- DPO merge ---
    print(f"\n[6] Merging DPO data...")
    dpo_analysis = load_jsonl(DATA_DIR / "dpo_data_combined.jsonl")
    dpo_tool = load_jsonl(DATA_DIR / "dpo_tool_calling_fixed.jsonl")
    all_dpo = dpo_analysis + dpo_tool
    random.shuffle(all_dpo)

    dpo_path = DATA_DIR / "dpo_pytorch.jsonl"
    with open(dpo_path, "w") as f:
        for ex in all_dpo:
            f.write(json.dumps(ex) + "\n")
    print(f"    DPO: {len(all_dpo)} pairs → {dpo_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SFT: {len(all_data)} total → {OUTPUT_DIR}")
    print(f"  DPO: {len(all_dpo)} pairs → {dpo_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
