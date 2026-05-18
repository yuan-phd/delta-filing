"""
Delta Filing — Embedding training data from REAL SFT data
==========================================================
Replaces SECTION_TOPICS (synthetic) with real SEC filing analysis from
sft_data_combined.jsonl.

Positive pair strategy: same SEC section, different companies.
  Example: AAPL 2025 Item 1A  ↔  NVDA 2025 Item 1A

This trains the embedding model to recognize that "Item 1A" (Risk Factors)
discussions cluster together regardless of company.

Usage:
    python generate_embedding_data.py

Output: training_data/embedding_pairs.jsonl  (~1500 pairs)
"""

import json
import os
import random
from collections import defaultdict

SRC = "training_data/sft_data_combined.jsonl"
OUT = "training_data/embedding_pairs.jsonl"

# Section name mapping for clearer labels
SECTION_NAMES = {
    "1":  "Item 1 (Business)",
    "1A": "Item 1A (Risk Factors)",
    "7":  "Item 7 (MD&A)",
    "7A": "Item 7A (Quantitative Risk)",
}

# Set random seed for reproducible pair generation
random.seed(42)


def main():
    # Load all SFT samples with company+section metadata
    with open(SRC) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    valid = [s for s in samples if s.get("company") and s.get("section") and s.get("answer")]
    print(f"Loaded {len(valid)} samples with company+section+answer")

    # Group by section
    by_section = defaultdict(list)
    for s in valid:
        by_section[s["section"]].append(s)

    print("\nSamples per section:")
    for sec, group in sorted(by_section.items()):
        n_companies = len(set(s["company"] for s in group))
        print(f"  {sec} ({SECTION_NAMES.get(sec, sec)}): {len(group)} samples, {n_companies} companies")

    # Generate positive pairs: same section, DIFFERENT companies
    # This forces the model to learn topic-level (Risk Factors, MD&A, etc.)
    # rather than company-level features
    pairs = []
    for section, group in by_section.items():
        # All possible (sample_i, sample_j) where i.company != j.company
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[i]["company"] != group[j]["company"]:
                    pairs.append({
                        "anchor":   group[i]["answer"],
                        "positive": group[j]["answer"],
                        "topic":    f"section_{section}",
                        "anchor_company":   group[i]["company"],
                        "positive_company": group[j]["company"],
                    })

    random.shuffle(pairs)

    # Cap at a reasonable size to keep training fast
    # 2000 pairs is plenty for a 22M model
    MAX_PAIRS = 2000
    if len(pairs) > MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]
        print(f"\nCapped to {MAX_PAIRS} pairs")

    # Write
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nGenerated {len(pairs)} positive pairs")
    print(f"Saved to: {OUT}")

    # Sanity check: distribution
    topic_counts = defaultdict(int)
    for p in pairs:
        topic_counts[p["topic"]] += 1
    print("\nFinal pair distribution by section:")
    for topic, n in sorted(topic_counts.items()):
        print(f"  {topic}: {n}")


if __name__ == "__main__":
    main()
