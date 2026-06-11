#!/usr/bin/env python
"""Freeze control C for the register-shift control probe: a rejected-register
(verbose/generic) paraphrase of each extraction item, generated ONCE via GPT-4o-mini
using the DEGRADE_PROMPT recipe (the same recipe that produced DPO 'rejected'
answers), then written to disk. C is NOT level-comparable to gold (different
length/format) -- analyzed by trajectory direction only.

  python data_prep/gen_control_C.py
-> data_eval/track1_extraction_controlC.jsonl  {item_id, control_c}
"""
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data_eval", "track1_extraction.jsonl")
OUT = os.path.join(ROOT, "data_eval", "track1_extraction_controlC.jsonl")
load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"

# Mirrors generate_fixed_data.py TOOL_RESULT_DEGRADE_PROMPT (strips ticker/section/date,
# vague, generic) -- the extraction rejected-register.
DEGRADE = """Take this filing description and create a WORSE, rejected-style version.
The worse version should:
- Remove the company ticker
- Remove the specific section name (e.g. Item 1A, Item 7)
- Remove the filing date
- Replace specifics with vague generalizations
- Sound generic, like it could apply to any company's filing

Original:
{good}

Respond with ONLY the degraded text, nothing else."""


def good_answer(g):
    return (f"This is the {g['section']} of the SEC filing for ticker {g['ticker']}, "
            f"filed on {g['filing_date']}, containing that section's disclosures.")


def main():
    items = [json.loads(l) for l in open(SRC) if l.strip()]
    out = []
    for i, it in enumerate(items):
        r = client.chat.completions.create(
            model=MODEL, temperature=0.9, max_tokens=200,
            messages=[{"role": "user", "content": DEGRADE.format(good=good_answer(it["gold"]))}])
        out.append({"item_id": it["item_id"], "control_c": r.choices[0].message.content.strip()})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(items)}")
        time.sleep(0.05)
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    # quick hygiene: all present, non-empty, no leftover ticker/date leakage check is soft
    assert all(o["control_c"] for o in out), "empty control_c"
    print(f"wrote {len(out)} control-C paraphrases -> {OUT}")
    print("sample:", out[0]["control_c"][:120])


if __name__ == "__main__":
    sys.exit(main())
