"""
Delta Filing — Training Data Generator
========================================
Generates SFT and DPO training data using real SEC filings + GPT-4o-mini.

Strategy:
  1. Fetch real filing sections from EDGAR (we already have this)
  2. Send sections to GPT-4o-mini with specific prompts
  3. Generate Q&A pairs in different categories:
     - Section summary
     - Change detection analysis
     - Red flag identification
     - Cross-reference analysis
  4. Save as JSONL for training

For DPO, we also generate "rejected" responses (bad answers)
by intentionally degrading the good answers.

Usage:
    python generate_training_data.py
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from edgar import get_section, get_filings
from diff import diff_sections, split_into_paragraphs

load_dotenv()
client = OpenAI()

OUTPUT_DIR = Path("training_data")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
#  Validation — check which companies/sections actually work
# ============================================================

def validate_company(company: str, filing_type: str = "10-K",
                     sections: list[str] = None) -> dict:
    """Check which sections can be fetched for both latest and previous filing.

    Returns dict mapping section_id to status:
        "both"    — both filings parsed, reasonable length → good for diff
        "latest"  — only latest filing works → good for summary/red_flag only
        "failed"  — can't parse → skip entirely
    """
    if sections is None:
        sections = ["1A", "7", "1", "7A"]

    results = {}
    MIN_CHARS = 500     # too short = parsing error
    MAX_CHARS = 100000  # too long = boundary error

    for section_id in sections:
        # Try latest filing
        try:
            s0 = get_section(company, filing_type, section_id, 0)
            len0 = len(s0["text"])
            if len0 < MIN_CHARS or len0 > MAX_CHARS:
                results[section_id] = "failed"
                continue
        except Exception:
            results[section_id] = "failed"
            continue

        # Try previous filing
        try:
            s1 = get_section(company, filing_type, section_id, 1)
            len1 = len(s1["text"])
            if len1 < MIN_CHARS or len1 > MAX_CHARS:
                results[section_id] = "latest"
            else:
                results[section_id] = "both"
        except Exception:
            results[section_id] = "latest"

    return results


# ============================================================
#  Prompt templates for generating training examples
# ============================================================

SUMMARY_PROMPT = """You are a financial analyst. Based on this SEC filing section,
generate a question-answer pair.

The QUESTION should be something an analyst would ask about this section.
The ANSWER should be a thorough analysis that:
- References specific content from the section
- Cites specific numbers, percentages, or dates when available
- Provides analytical judgment, not just summary
- Notes any concerns or notable items
- Is 150-300 words long

Filing section ({section_name}) from {company}'s {filing_type}, filed {filing_date}:

{section_text}

Respond in this exact JSON format:
{{"question": "your question here", "answer": "your detailed answer here"}}
"""

DIFF_PROMPT = """You are a financial analyst specializing in detecting changes between filings.

Here is a summary of changes between two consecutive {filing_type} filings for {company}:
- Filing dates: {date_old} → {date_new}
- Added paragraphs: {added}
- Removed paragraphs: {removed}
- Modified paragraphs: {modified}
- Unchanged paragraphs: {unchanged}

Sample of changes:
{changes_sample}

Generate a question-answer pair about these changes.
The QUESTION should ask about what changed and why it matters.
The ANSWER should:
- Highlight the most significant changes
- Explain why specific changes might be concerning or notable
- Reference specific before/after wording when relevant
- Be 150-300 words long

Respond in this exact JSON format:
{{"question": "your question here", "answer": "your detailed answer here"}}
"""

RED_FLAG_PROMPT = """You are a financial risk analyst reviewing SEC filings.

Based on this SEC filing section, generate a question-answer pair focused on
identifying potential red flags or concerns.

Filing section ({section_name}) from {company}'s {filing_type}, filed {filing_date}:

{section_text}

The QUESTION should ask about potential risks or concerns.
The ANSWER should:
- Identify specific concerning items in the text
- Explain why each item is potentially concerning
- Rate severity (high/medium/low) for each concern
- Suggest what to monitor going forward
- Be 150-300 words long

Respond in this exact JSON format:
{{"question": "your question here", "answer": "your detailed answer here"}}
"""

DEGRADE_PROMPT = """Take this detailed financial analysis and create a WORSE version.
The worse version should:
- Remove all specific numbers, dates, and section references
- Replace specific claims with vague generalizations
- Remove analytical judgment, keep only surface-level summary
- Be shorter (50-100 words)
- Sound generic, like it could apply to any company

Original analysis:
{good_answer}

Respond with ONLY the degraded text, nothing else."""


# ============================================================
#  Data generation functions
# ============================================================

def generate_qa_pair(prompt: str, max_retries: int = 3) -> dict | None:
    """Call GPT-4o-mini to generate a Q&A pair."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
            )
            text = response.choices[0].message.content.strip()

            # Clean up markdown code blocks if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    return None


def generate_degraded_answer(good_answer: str) -> str | None:
    """Generate a deliberately bad version of a good answer (for DPO)."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": DEGRADE_PROMPT.format(
                good_answer=good_answer
            )}],
            temperature=0.9,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    Degradation failed: {e}")
        return None


def generate_summary_examples(company: str, filing_type: str = "10-K",
                               sections: list[str] = None) -> list[dict]:
    """Generate section summary Q&A pairs from real filings."""
    if sections is None:
        sections = ["1A", "7"]  # Risk Factors and MD&A

    examples = []

    for section_id in sections:
        for filing_index in range(2):  # Latest and previous filing
            try:
                section = get_section(company, filing_type, section_id, filing_index)
            except Exception as e:
                print(f"    Could not fetch {company} {filing_type} section {section_id} "
                      f"index {filing_index}: {e}")
                continue

            # Truncate to avoid token limits
            section_text = section["text"][:4000]

            prompt = SUMMARY_PROMPT.format(
                section_name=section["section_name"],
                company=company,
                filing_type=filing_type,
                filing_date=section["filing_date"],
                section_text=section_text,
            )

            print(f"  Generating summary Q&A: {company} {section_id} "
                  f"({section['filing_date']})...")
            qa = generate_qa_pair(prompt)
            if qa:
                qa["category"] = "section_summary"
                qa["company"] = company
                qa["section"] = section_id
                qa["filing_date"] = section["filing_date"]
                examples.append(qa)

    return examples


def generate_diff_examples(company: str, filing_type: str = "10-K",
                            sections: list[str] = None) -> list[dict]:
    """Generate change detection Q&A pairs from filing diffs."""
    if sections is None:
        sections = ["1A"]

    examples = []

    for section_id in sections:
        try:
            section_new = get_section(company, filing_type, section_id, 0)
            section_old = get_section(company, filing_type, section_id, 1)
            diff = diff_sections(section_old, section_new)
        except Exception as e:
            print(f"    Could not diff {company} section {section_id}: {e}")
            continue

        # Create a sample of changes for the prompt
        changes_sample = ""
        for item in diff["added"][:2]:
            changes_sample += f"ADDED: {item['text'][:200]}...\n"
        for item in diff["removed"][:2]:
            changes_sample += f"REMOVED: {item['text'][:200]}...\n"
        for item in diff["modified"][:2]:
            changes_sample += (f"MODIFIED (similarity {item['similarity']}):\n"
                             f"  BEFORE: {item['before'][:150]}...\n"
                             f"  AFTER: {item['after'][:150]}...\n")

        s = diff["summary"]
        prompt = DIFF_PROMPT.format(
            company=company,
            filing_type=filing_type,
            date_old=diff["filing_a_date"],
            date_new=diff["filing_b_date"],
            added=s["added"],
            removed=s["removed"],
            modified=s["modified"],
            unchanged=s["unchanged"],
            changes_sample=changes_sample,
        )

        print(f"  Generating diff Q&A: {company} {section_id}...")
        qa = generate_qa_pair(prompt)
        if qa:
            qa["category"] = "change_detection"
            qa["company"] = company
            qa["section"] = section_id
            examples.append(qa)

    return examples


def generate_red_flag_examples(company: str, filing_type: str = "10-K",
                                sections: list[str] = None) -> list[dict]:
    """Generate red flag detection Q&A pairs."""
    if sections is None:
        sections = ["1A"]

    examples = []

    for section_id in sections:
        try:
            section = get_section(company, filing_type, section_id, 0)
        except Exception as e:
            print(f"    Could not fetch {company} section {section_id}: {e}")
            continue

        section_text = section["text"][:4000]

        prompt = RED_FLAG_PROMPT.format(
            section_name=section["section_name"],
            company=company,
            filing_type=filing_type,
            filing_date=section["filing_date"],
            section_text=section_text,
        )

        print(f"  Generating red flag Q&A: {company} {section_id}...")
        qa = generate_qa_pair(prompt)
        if qa:
            qa["category"] = "red_flag"
            qa["company"] = company
            qa["section"] = section_id
            examples.append(qa)

    return examples


# ============================================================
#  Main: generate training dataset
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch number (1 or 2)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Delta Filing — Training Data Generator (Batch {args.batch})")
    print("=" * 60)

    # 20 companies per batch, across different industries
    if args.batch == 1:
        companies = [
            # Tech
            "AAPL", "MSFT", "GOOGL", "NVDA", "META",
            "AMZN", "CRM", "INTC",
            # Finance
            "JPM", "GS", "BAC", "V",
            # Healthcare
            "JNJ", "PFE", "UNH",
            # Energy
            "XOM", "CVX",
            # Consumer
            "WMT", "KO", "DIS",
        ]
        sft_file = "sft_data_batch1.jsonl"
        dpo_file = "dpo_data_batch1.jsonl"
    else:
        companies = [
            # Tech
            "ORCL", "ADBE", "CSCO", "AMD", "QCOM",
            "TXN", "NOW", "SNOW",
            # Finance
            "MA", "C", "BLK", "AXP",
            # Healthcare
            "ABBV", "LLY", "MRK",
            # Energy
            "NEE", "SLB",
            # Consumer/Industrial
            "NKE", "MCD", "CAT",
        ]
        sft_file = "sft_data_batch2.jsonl"
        dpo_file = "dpo_data_batch2.jsonl"

    all_sections = ["1A", "7", "1", "7A"]

    # --- Phase 1: Validate all companies ---
    print("\n--- Phase 1: Validating companies ---")
    company_sections = {}  # {company: {section_id: "both"/"latest"/"failed"}}

    for company in companies:
        results = validate_company(company, sections=all_sections)
        company_sections[company] = results

        both = [s for s, v in results.items() if v == "both"]
        latest = [s for s, v in results.items() if v == "latest"]
        failed = [s for s, v in results.items() if v == "failed"]
        print(f"  {company}: diff={both}, summary_only={latest}, failed={failed}")

    # Count usable combinations
    total_diff = sum(
        1 for c in companies
        for s, v in company_sections[c].items() if v == "both"
    )
    total_summary = sum(
        1 for c in companies
        for s, v in company_sections[c].items() if v in ("both", "latest")
    )
    print(f"\n  Usable for diff: {total_diff} company-section pairs")
    print(f"  Usable for summary/red_flag: {total_summary} company-section pairs")

    # --- Phase 2: Generate data only from valid combinations ---
    print("\n--- Phase 2: Generating training data ---")

    all_sft_examples = []
    stats = {"section_summary": 0, "change_detection": 0, "red_flag": 0}

    for i, company in enumerate(companies):
        sections = company_sections[company]
        print(f"\n--- [{i+1}/{len(companies)}] {company} ---")

        # Summary: use sections where at least the latest filing works
        summary_ok = [s for s, v in sections.items() if v in ("both", "latest")]
        if summary_ok:
            summaries = generate_summary_examples(company, sections=summary_ok)
            all_sft_examples.extend(summaries)
            stats["section_summary"] += len(summaries)
        else:
            summaries = []

        # Diff: only use sections where BOTH filings parse correctly
        diff_ok = [s for s, v in sections.items() if v == "both"]
        if diff_ok:
            diffs = generate_diff_examples(company, sections=diff_ok)
            all_sft_examples.extend(diffs)
            stats["change_detection"] += len(diffs)
        else:
            diffs = []

        # Red flag: same as summary (only need latest filing)
        red_flag_ok = [s for s, v in sections.items() if v in ("both", "latest")]
        if red_flag_ok:
            red_flags = generate_red_flag_examples(company, sections=red_flag_ok)
            all_sft_examples.extend(red_flags)
            stats["red_flag"] += len(red_flags)
        else:
            red_flags = []

        print(f"  {company}: {len(summaries)} summaries, {len(diffs)} diffs, "
              f"{len(red_flags)} red flags")
        print(f"  Running total: {sum(stats.values())} examples "
              f"(S:{stats['section_summary']} D:{stats['change_detection']} "
              f"R:{stats['red_flag']})")

        time.sleep(0.5)

    # Save SFT data
    sft_path = OUTPUT_DIR / sft_file
    with open(sft_path, "w") as f:
        for ex in all_sft_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nSFT data saved: {sft_path} ({len(all_sft_examples)} examples)")

    # Category balance report
    print(f"\nCategory balance:")
    total = len(all_sft_examples)
    for cat, count in stats.items():
        pct = count / total * 100 if total else 0
        print(f"  {cat}: {count} ({pct:.0f}%)")

    # Generate DPO preference pairs
    print("\n--- Phase 3: Generating DPO preference pairs ---")
    dpo_examples = []

    for i, ex in enumerate(all_sft_examples):
        if (i + 1) % 10 == 0:
            print(f"  Degrading {i+1}/{len(all_sft_examples)}...")
        bad_answer = generate_degraded_answer(ex["answer"])
        if bad_answer:
            dpo_examples.append({
                "question": ex["question"],
                "chosen": ex["answer"],
                "rejected": bad_answer,
                "category": ex["category"],
                "company": ex.get("company", ""),
            })

    dpo_path = OUTPUT_DIR / dpo_file
    with open(dpo_path, "w") as f:
        for ex in dpo_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"DPO data saved: {dpo_path} ({len(dpo_examples)} examples)")

    print(f"\n{'=' * 60}")
    print(f"  Batch {args.batch} complete")
    print(f"  SFT examples: {len(all_sft_examples)}")
    print(f"  DPO pairs: {len(dpo_examples)}")
    print(f"  Balance: S:{stats['section_summary']} D:{stats['change_detection']} "
          f"R:{stats['red_flag']}")
    print(f"{'=' * 60}")
