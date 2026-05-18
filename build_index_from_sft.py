"""
Delta Filing — Build FAISS index from SFT data
================================================
Builds FAISS index using the 351 real SEC filing analyses from
sft_data_combined.jsonl — the same data used to train the embedding model.

Why this instead of build_index_from_cache():
  - No network calls to SEC (cache may not cover all 15 tickers)
  - More data: 351 samples across 37 companies vs 30 samples (15 x 2)
  - All 4 section types: 1, 1A, 7, 7A
  - Same data distribution as embedding training (consistency)

Usage:
    python build_index_from_sft.py
"""

import json
import os
import sys

# Reuse mcp_rag.py classes
from mcp_rag import EmbeddingEngine, FilingVectorStore, INDEX_DIR

SRC = "training_data/sft_data_combined.jsonl"

# Section ID → human readable name
SECTION_NAMES = {
    "1":  "Item 1 (Business)",
    "1A": "Item 1A (Risk Factors)",
    "7":  "Item 7 (MD&A)",
    "7A": "Item 7A (Quantitative Risk)",
}


def main():
    if not os.path.exists(SRC):
        print(f"ERROR: {SRC} not found")
        sys.exit(1)

    print("=" * 60)
    print("  Building FAISS Index from SFT data")
    print("=" * 60)

    # Load SFT samples
    with open(SRC) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    valid = [s for s in samples if s.get("company") and s.get("section") and s.get("answer")]
    print(f"\n  Loaded {len(valid)} samples with company+section+answer")

    # Initialize engine + store
    engine = EmbeddingEngine()
    store = FilingVectorStore(engine)
    store.create_index()  # Start fresh, don't reuse old index

    # Index every sample as a separate vector
    indexed = 0
    for s in valid:
        ticker = s["company"]
        section_id = s["section"]
        section_name = SECTION_NAMES.get(section_id, f"Item {section_id}")
        filing_date = s.get("filing_date", "unknown")
        # Use answer text (real filing-based analysis)
        text = s["answer"]

        store.add(
            ticker=ticker,
            section=section_name,
            text=text,
            filing_date=filing_date,
            filing_type="10-K",
        )
        indexed += 1
        if indexed % 50 == 0:
            print(f"  Indexed {indexed}/{len(valid)}...")

    store.save()

    stats = store.stats()
    print(f"\n  Total indexed: {indexed}")
    print(f"  Unique companies: {stats['unique_companies']}")
    print(f"  Unique sections: {stats['unique_sections']}")
    print(f"  Companies: {', '.join(stats['companies'])}")
    print(f"\n  Saved to: {INDEX_DIR}/")


if __name__ == "__main__":
    main()
