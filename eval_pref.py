#!/usr/bin/env python
"""Eval harness (claude_code_spec.md §5) — generate, score, persist EVERY generation.

    python eval_pref.py --track track1 --condition dpo --seed 0 --task binary
    python eval_pref.py --track track1 --baseline --task json          # SFT baseline

For every item, persists to generations/{track}_{condition}_seed{S}_{task}.jsonl:
  {item_id, prompt, gold, raw_output, output_token_len,
   format_pass, normalization_applied, content_correct, verdict, ...}
No metric is computed without the underlying generations on disk (§5).

Aggregate per (track,condition,seed,task) -> results/{...}.json (§6).

Heavy model deps imported lazily (CUDA / 4-bit NF4 QLoRA, Kaggle). The
score+persist path (score_and_persist) is model-free and is exercised by the
local scorer dry-run.
"""
import argparse
import json
import os
import re
import statistics

import scorer as SC
import scorer_v2 as V

# Rigid eval prompts. NOTE (flagged for researcher review): §3a defines the
# binary task's output space as EXACTLY {COMPLETE, NEEDS_FOLLOW_UP} and §6's
# format gate is strict whole-string membership, so the binary eval prompt must
# demand exactly one label (no trailing reason). This differs from the looser
# REVIEW_SYSTEM_PROMPT used in the ablation notebook, which allowed
# "NEEDS_FOLLOW_UP: <reason>". This is the rigid-format end of the dissociation
# axis on purpose.
BINARY_SYSTEM_PROMPT = (
    "You are a quality reviewer for financial analysis. Review the analysis and "
    "respond with EXACTLY ONE WORD, either COMPLETE or NEEDS_FOLLOW_UP. "
    "Output only that single word. No explanation, no punctuation."
)
ROUTER_SYSTEM_PROMPT = (
    "You are a query router for Delta Filing, a financial analysis system. "
    "Classify the user's query into exactly one category. Respond with ONLY the "
    "category name, nothing else.\n\n"
    "Categories:\n"
    "- FILING_ANALYSIS: Questions about a specific company's filing content\n"
    "- FILING_DIFF: Questions comparing filings across time periods\n"
    "- COMPANY_DEEP_DIVE: Requests for comprehensive company analysis\n"
    "- SIMPLE_QUERY: Simple factual queries about filings\n"
    "- OUT_OF_SCOPE: Questions unrelated to SEC filings"
)
EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured fields from a SEC filing snippet. Respond with ONLY a "
    'JSON object with exactly these keys: {"ticker": "...", "section": "...", '
    '"filing_date": "YYYY-MM-DD"}. No prose, no code fences, nothing else.'
)

TASK_GEN = {
    "binary": {"system": BINARY_SYSTEM_PROMPT, "max_new_tokens": 16},
    "router": {"system": ROUTER_SYSTEM_PROMPT, "max_new_tokens": 16},
    "json": {"system": None, "max_new_tokens": 128},          # uses TOOL_SYSTEM_PROMPT
    "function_call": {"system": None, "max_new_tokens": 128},
    "extraction": {"system": EXTRACTION_SYSTEM_PROMPT, "max_new_tokens": 128},
}

EVAL_FILES = {  # full eval sets (~80 each) for the matrix
    ("track1", "binary"): "data_eval/track1_binary.jsonl",
    ("track1", "json"): "data_eval/track1_json.jsonl",
    ("track1", "router"): "data_eval/track1_router.jsonl",
    ("track1", "extraction"): "data_eval/track1_extraction.jsonl",
}
SMOKE_EVAL_FILES = {  # validated 5-item slice for the smoke gate
    ("track1", "binary"): "data_eval/track1_binary_smoke.jsonl",
    ("track1", "json"): "data_eval/track1_json_smoke.jsonl",
}


# ============================================================
# scoring + persistence (model-free; reused by the dry-run)
# ============================================================
def score_and_persist(track, condition, seed, task, items, gen_path, results_path,
                      tokenizer=None):
    """items: list of {item_id, prompt, gold, raw_output}. Score + write both files."""
    os.makedirs(os.path.dirname(gen_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    header = {"_header": True, "track": track, "condition": condition, "seed": seed,
              "task": task, "scorer_rules": SC.SCORER_RULES}

    rows = []
    with open(gen_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for it in items:
            raw = it["raw_output"]
            verdict, fields = SC.score(task, raw, it["gold"],
                                       prompt=it.get("prompt"), track=track)
            tok_len = (len(tokenizer.encode(raw)) if tokenizer is not None
                       else len(raw.split()))
            row = {
                "item_id": it["item_id"], "prompt": it["prompt"], "gold": it["gold"],
                "raw_output": raw, "output_token_len": tok_len,
                "raw_output_untruncated": it.get("raw_output_untruncated"),
                "format_pass": fields["format_pass"],
                "lenient_format_pass": fields["lenient_format_pass"],
                "leading_prose": fields["leading_prose"],
                "normalization_applied": fields["normalization_applied"],
                "content_correct": fields["content_correct"],
                "verdict": verdict,
            }
            f.write(json.dumps(row) + "\n")
            rows.append(row)

    n = len(rows)
    n_pass = sum(r["verdict"] == SC.VERDICT_PASS for r in rows)
    n_fb = sum(r["verdict"] == SC.VERDICT_FORMAT_BREAK for r in rows)
    n_re = sum(r["verdict"] == SC.VERDICT_REASONING_ERROR for r in rows)
    # lenient sensitivity: format breaks under the strict gate that the lenient
    # gate would have accepted (structured token present but not leading).
    n_fb_lenient = sum(not r["lenient_format_pass"] for r in rows)
    lens = [r["output_token_len"] for r in rows]
    results = {
        "track": track, "condition": condition, "seed": seed, "task": task, "n": n,
        "accuracy": n_pass / n if n else None,
        "format_break_rate": n_fb / n if n else None,            # strict (primary)
        "format_break_rate_lenient": n_fb_lenient / n if n else None,  # sensitivity
        "leading_prose_rate": sum(r["leading_prose"] for r in rows) / n if n else None,
        "reasoning_error_rate": n_re / n if n else None,
        "mean_output_len": statistics.mean(lens) if lens else None,
        "median_output_len": statistics.median(lens) if lens else None,
        "pct_normalized": sum(r["normalization_applied"] for r in rows) / n if n else None,
        "generations_file": gen_path,
        "scorer_rules": SC.SCORER_RULES,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    return results, rows


def score_and_persist_v2(track, condition, seed, task, items, gen_path, results_path,
                         tokenizer=None):
    """SCORER V2 (three-tier) scoring + persist — the FROZEN matrix/baseline scorer.
    Primary metric = semantic accuracy (PASS + SURFACE_FORM); strict = PASS only.
    Uses scorer_v2.score_v2 with the frozen canonicalization (section id \\d+[A-Z]?,
    case-fold, ignore appended titles). `accuracy` is aliased to semantic_accuracy so
    downstream curves report the semantic metric. Same scorer as run_baseline_v2 ->
    matrix-vs-baseline_v2 are like-for-like."""
    os.makedirs(os.path.dirname(gen_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    header = {"_header": True, "track": track, "condition": condition, "seed": seed,
              "task": task, "scorer": "v2", "canon_rules": V.CANON_RULES}
    rows, fields_list = [], []
    with open(gen_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for it in items:
            raw = it["raw_output"]
            tier, fields = V.score_v2(task, raw, it["gold"], prompt=it.get("prompt"), track=track)
            fields_list.append(fields)
            tok_len = (len(tokenizer.encode(raw)) if tokenizer is not None else len(raw.split()))
            row = {"item_id": it["item_id"], "prompt": it["prompt"], "gold": it["gold"],
                   "raw_output": raw, "output_token_len": tok_len,
                   "raw_output_untruncated": it.get("raw_output_untruncated"),
                   "format_pass": fields["format_pass"], "tier": tier,
                   "strict_pass": fields["strict_pass"], "semantic_pass": fields["semantic_pass"]}
            f.write(json.dumps(row) + "\n")
            rows.append(row)
    agg = V.aggregate(fields_list)
    n = len(rows)
    lens = [r["output_token_len"] for r in rows]
    results = {
        "track": track, "condition": condition, "seed": seed, "task": task, "n": n,
        "scorer": "v2",
        "accuracy": agg["semantic_accuracy"],            # PRIMARY (alias for downstream curves)
        "semantic_accuracy": agg["semantic_accuracy"],
        "strict_accuracy": agg["strict_accuracy"],       # secondary
        "format_break_rate": agg["format_break"] / n if n else None,
        "surface_form_rate": agg["surface_form"] / n if n else None,
        "semantic_error_rate": agg["semantic_error"] / n if n else None,
        "mean_output_len": statistics.mean(lens) if lens else None,
        "median_output_len": statistics.median(lens) if lens else None,
        "generations_file": gen_path, "canon_rules": V.CANON_RULES,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    return results, rows


# ============================================================
# generation (lazy heavy imports; CUDA / Kaggle)
# ============================================================
def adapter_path(args):
    import train_pref as T
    explicit = getattr(args, "adapter_path", None)
    if explicit:                       # eval a specific checkpoint dir
        return explicit
    if args.baseline:
        cfg = T.TRACK1 if args.track == "track1" else T.TRACK2
        # allow Kaggle path override for the SFT baseline adapter
        return getattr(args, "sft_adapter", None) or cfg["sft_adapter"]
    return os.path.join(args.output_base, f"{args.track}_{args.condition}_seed{args.seed}")


def load_model_and_tokenizer(args):
    import train_pref as T
    cfg = T.TRACK1 if args.track == "track1" else T.TRACK2
    tok = T.build_tokenizer(cfg)
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    src, local_only = T.resolve_model_source(cfg["model_id"])
    base = AutoModelForCausalLM.from_pretrained(
        src, quantization_config=bnb, trust_remote_code=True, device_map={"": 0},
        cache_dir=T.HF_HUB_CACHE, local_files_only=local_only)
    ap = adapter_path(args)
    model = PeftModel.from_pretrained(base, ap) if ap else base
    model.eval()
    return model, tok, cfg


# textual turn-boundary marker left over after special tokens are stripped
# (belt-and-suspenders if the model emits a role word without the special token)
_TURN_MARKER = re.compile(r"\n\s*(?:user|assistant|system)\s*\n", re.IGNORECASE)


def _ids_for(tok, *toks):
    """Token ids for the given special tokens that actually exist in the vocab."""
    ids = []
    unk = getattr(tok, "unk_token_id", None)
    for t in toks:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            ids.append(tid)
    return ids


def _eos_ids(tok):
    """End-of-turn ids to halt generation on: chat end-of-turn (<|im_end|>) + native eos."""
    ids = _ids_for(tok, "<|im_end|>")
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    return list(dict.fromkeys(ids))          # dedupe, preserve order


def _clean_completion(tok, gen_ids):
    """Truncate the generated completion at the first end-of-turn / new-role token,
    then decode. Returns (cleaned, untruncated)."""
    gen_ids = list(gen_ids)
    cut_set = set(_ids_for(tok, "<|im_end|>", "<|im_start|>"))
    if tok.eos_token_id is not None:
        cut_set.add(tok.eos_token_id)
    trunc = gen_ids
    for i, t in enumerate(gen_ids):
        if t in cut_set:
            trunc = gen_ids[:i]
            break
    cleaned = tok.decode(trunc, skip_special_tokens=True).strip()
    # secondary textual cut in case a role word leaked without its special token
    m = _TURN_MARKER.search(cleaned)
    if m:
        cleaned = cleaned[:m.start()].strip()
    untrunc = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return cleaned, untrunc


def _task_system(task):
    """The SAME system prompt the generation path uses for this task."""
    import train_pref as T
    g = TASK_GEN.get(task, {}).get("system")
    if g:
        return g
    return T.TOOL_SYSTEM_PROMPT if task in ("json", "function_call") else T.SYSTEM_PROMPT


def render_gold_seq(tok, task, prompt, gold, max_len=1024):
    """Render system+user+assistant(gold) exactly as the task's eval template builds
    it; return (ids, completion_start). Gold = enum label (router) or JSON (json/
    extraction)."""
    import torch
    gold_text = gold if isinstance(gold, str) else json.dumps(gold)
    pmsgs = [{"role": "system", "content": _task_system(task)},
             {"role": "user", "content": prompt}]
    full = tok.apply_chat_template(pmsgs + [{"role": "assistant", "content": gold_text}], tokenize=False)
    pt = tok.apply_chat_template(pmsgs, tokenize=False, add_generation_prompt=True)
    fid = tok.encode(full, max_length=max_len, truncation=True)
    pid = tok.encode(pt, max_length=max_len, truncation=True)
    cstart = 0
    for a, b in zip(fid, pid):
        if a != b:
            break
        cstart += 1
    return torch.tensor(fid, dtype=torch.long), cstart


def _generate(model, tok, task, items):
    """Free greedy generation + clean-completion for a list of items (model-agnostic)."""
    import torch
    import train_pref as T
    gen_cfg = TASK_GEN[task]
    sys_prompt = gen_cfg["system"] or (
        T.TOOL_SYSTEM_PROMPT if task in ("json", "function_call") else T.SYSTEM_PROMPT)
    eos_ids = _eos_ids(tok)                  # halt at <|im_end|> (not just native eos)
    out = []
    for it in items:
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": it["prompt"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=gen_cfg["max_new_tokens"],
                                 do_sample=False, eos_token_id=eos_ids,
                                 pad_token_id=tok.pad_token_id)
        gen_ids = gen[0][inp["input_ids"].shape[1]:].tolist()
        cleaned, untrunc = _clean_completion(tok, gen_ids)
        out.append({**it, "raw_output": cleaned, "raw_output_untruncated": untrunc})
    return out


def generate_items(args, items):
    """One-shot path (loads its own model). Used by run_baseline_audit."""
    model, tok, cfg = load_model_and_tokenizer(args)
    return _generate(model, tok, args.task, items), tok


class EvalSession:
    """Load the 4-bit base ONCE and swap LoRA adapters across checkpoints, instead
    of reloading the base per eval (cuts memory churn + time on the matrix).

    Create it AFTER training has freed its models, and tear it down before the next
    training run, so the base never coexists with the trainable model on the GPU.
    """
    def __init__(self, track):
        import torch
        import train_pref as T
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel  # noqa: F401  (used in _model_for)
        self.cfg = T.TRACK1 if track == "track1" else T.TRACK2
        self.tok = T.build_tokenizer(self.cfg)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        src, local_only = T.resolve_model_source(self.cfg["model_id"])
        self.base = AutoModelForCausalLM.from_pretrained(
            src, quantization_config=bnb, trust_remote_code=True, device_map={"": 0},
            cache_dir=T.HF_HUB_CACHE, local_files_only=local_only)
        self.base.eval()
        self.peft = None
        self.loaded = {}        # adapter_path -> adapter_name

    def _model_for(self, adapter_path):
        from peft import PeftModel
        if adapter_path is None:
            return self.base
        name = self.loaded.get(adapter_path)
        if name is None:
            name = f"ad{len(self.loaded)}"
            if self.peft is None:
                self.peft = PeftModel.from_pretrained(self.base, adapter_path, adapter_name=name)
            else:
                self.peft.load_adapter(adapter_path, adapter_name=name)
            self.loaded[adapter_path] = name
        self.peft.set_adapter(name)
        self.peft.eval()
        return self.peft

    def generate(self, task, items, adapter_path):
        model = self._model_for(adapter_path)
        return _generate(model, self.tok, task, items), self.tok

    def gold_logprob(self, task, items, adapter_path):
        """Teacher-forced per-token avg log-prob of the GOLD completion (secondary
        probability-layer signal). Gold rendered exactly as that task's eval
        template builds the assistant turn (json/extraction = JSON object, router
        = enum label). One forward per item -- near-free vs autoregressive gen."""
        import torch
        import pref_losses as PL
        model = self._model_for(adapter_path)
        dev = model.device
        out = []
        for it in items:
            ids, cstart = render_gold_seq(self.tok, task, it["prompt"], it["gold"])
            with torch.no_grad():
                lp = float(PL.compute_log_probs(model, ids, cstart, dev))
            out.append({"item_id": it["item_id"], "gold_logp": lp,
                        "gold_completion_len": len(ids) - cstart})
        return out


def load_eval_items(track, task, max_items=None, smoke=False):
    path = (SMOKE_EVAL_FILES if smoke else EVAL_FILES)[(track, task)]
    if not os.path.exists(path):
        raise SystemExit(f"NOT RUN: eval items {path} missing (run data_prep first).")
    items = [json.loads(l) for l in open(path) if l.strip()]
    return items[:max_items] if max_items else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["track1", "track2"], required=True)
    ap.add_argument("--condition", choices=["dpo", "dpo_clamp", "ipo", "cdpo"], default="dpo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", required=True, choices=["binary", "json", "function_call", "ifeval"])
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--output-base", default="adapters/pref")
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()

    cond = "baseline" if args.baseline else args.condition
    items = load_eval_items(args.track, args.task, args.max_items)
    gen_items, tok = generate_items(args, items)

    gen_path = f"generations/{args.track}_{cond}_seed{args.seed}_{args.task}.jsonl"
    results_path = f"results/{args.track}_{cond}_seed{args.seed}_{args.task}.json"
    results, _ = score_and_persist(args.track, cond, args.seed, args.task,
                                   gen_items, gen_path, results_path, tokenizer=tok)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
