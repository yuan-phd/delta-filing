#!/usr/bin/env python
"""Script 3 — same-backend fp16 gold-logp for ALL 3 tasks (json, router, extraction), on Mac.

Closes the cross-task / cross-backend gap: the matrix goldlogp files exist per task but were
produced on the Kaggle nf4 backend, and the Mac controls (mac_gold_controls.py) only scored
extraction. This teacher-forces every task's GOLD completion through the SFT baseline + each
seed's FINAL checkpoint in a single fp16 run, so Delta_gold = gold(final) - gold(SFT) is
clean and same-backend across json / router / extraction with no cross-session questions.

FAITHFUL METHOD — identical to the matrix gold scorer (kaggle_upload/eval_pref.py):
  * render_gold_seq : system+user+assistant(gold) via the SAME per-task system prompt the
                      generation/eval path uses (router/extraction prompts verbatim from
                      eval_pref; json -> train_pref.TOOL_SYSTEM_PROMPT). Replicated inline
                      here so the script imports only train_pref + pref_losses (no kaggle_upload
                      package, no scorer deps) — same numbers, zero import coupling.
  * teacher-forcing : pref_losses.compute_log_probs (completion-only, length-normalized per-
                      token mean log-prob) — the SAME function training uses for reference
                      log-probs and the metric the matrix logs as `gold_per_token_logp`.
  * gold_completion_len = len(ids) - completion_start  (matches matrix goldlogp files).
Only difference vs the matrix files: unquantized fp16/bf16 (Mac, no nf4), and all 3 tasks
emitted from one adapter load.

  python mac_gold_3task.py --seed all          # SFT + 8 seeds x 3 tasks (hours; ~2-3h CPU)
  python mac_gold_3task.py --seed 0            # one seed (reuses cached SFT reference)
  python mac_gold_3task.py --analyze-only      # rebuild Delta_gold tables from cached jsonl
Outputs -> analysis_ws/gold_3task/
             goldlogp_sft_{task}.jsonl
             goldlogp_seed{S}_final_{task}.jsonl     # rows: {item_id, gold_logp, gold_completion_len}
             gold_3task_summary.json
Total forwards: SFT (3x160=480) + 8 seeds x 3 tasks x 160 = 4320.
"""
import argparse
import json
import os
import time

import torch

import train_pref as T
import pref_losses as PL

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "analysis_ws", "gold_3task")
SFT_ADAPTER = T.TRACK1["sft_adapter"]                    # adapters/sft_pytorch
FINAL = "kaggle_result/final/seed{s}/adapters/pref/track1_dpo_seed{s}"   # final = top dir
TASKS = ["json", "router", "extraction"]
ITEMS_FILE = os.path.join(ROOT, "data_eval", "track1_{task}.jsonl")
N_BOOT = 10000
BOOT_SEED = 12345

# Per-task system prompts — VERBATIM from kaggle_upload/eval_pref.py so rendering matches the
# matrix gold scorer exactly. json/function_call fall through to train_pref.TOOL_SYSTEM_PROMPT.
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
_TASK_SYSTEM = {
    "json": T.TOOL_SYSTEM_PROMPT,
    "function_call": T.TOOL_SYSTEM_PROMPT,
    "router": ROUTER_SYSTEM_PROMPT,
    "extraction": EXTRACTION_SYSTEM_PROMPT,
}


def render_gold_seq(tok, task, prompt, gold, max_len=T.MAX_SEQ_LEN):
    """Render system+user+assistant(gold) exactly as eval_pref.render_gold_seq does;
    return (ids, completion_start). Gold = enum label (router) or JSON (json/extraction)."""
    gold_text = gold if isinstance(gold, str) else json.dumps(gold)
    pmsgs = [{"role": "system", "content": _TASK_SYSTEM[task]},
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


def out_path(tag, task):
    return os.path.join(OUT_DIR, f"goldlogp_{tag}_{task}.jsonl")


def load_cached(path):
    """Return {item_id: row} if a complete (160-row, header-excluded) file exists, else None."""
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if not r.get("_header")]
    if len(rows) == 160:
        return {r["item_id"]: r for r in rows}
    return None


def score_task(model, tok, task, items, device, label, path):
    """Teacher-force gold for all 160 items of one task through one adapter; write goldlogp file."""
    cached = load_cached(path)
    if cached is not None:
        print(f"  [{label}/{task}] cached -> {path} (skip)")
        return cached
    t0 = time.time()
    rows = []
    with torch.no_grad():
        for i, it in enumerate(items):
            ids, cstart = render_gold_seq(tok, task, it["prompt"], it["gold"])
            lg = float(PL.compute_log_probs(model, ids, cstart, device))
            rows.append({"item_id": it["item_id"], "gold_logp": lg,
                         "gold_completion_len": int(len(ids) - cstart)})
            if (i + 1) % 40 == 0 or (i + 1) == len(items):
                dt = time.time() - t0
                eta = dt / (i + 1) * (len(items) - i - 1)
                print(f"    [{label}/{task}] {i+1}/{len(items)} items  {dt:.0f}s  ETA {eta:.0f}s", flush=True)
    header = {"_header": True, "tag": label, "task": task,
              "metric": "gold_per_token_logp", "backend": "fp16_mac", "n": len(rows)}
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  [{label}/{task}] wrote -> {path}")
    return {r["item_id"]: r for r in rows}


# ---------------- analysis ----------------
def boot_ci(vals, lo=2.5, hi=97.5):
    import numpy as np
    rng = np.random.default_rng(BOOT_SEED)
    a = np.asarray(vals, float)
    m = a[rng.integers(0, len(a), size=(N_BOOT, len(a)))].mean(axis=1)
    return float(np.percentile(m, lo)), float(np.percentile(m, hi))


def analyze():
    import numpy as np
    sft = {task: load_cached(out_path("sft", task)) for task in TASKS}
    if any(sft[t] is None for t in TASKS):
        print("  analyze: SFT reference incomplete for some task — run a seed first.")
        return
    summary = {"reference": "sft_baseline", "backend": "fp16_mac",
               "metric": "Delta_gold = gold(final) - gold(sft), per-token length-normalized",
               "by_task": {}, "by_seed": {}}

    print("\n" + "=" * 80)
    print("SAME-BACKEND fp16 Delta_gold  (gold(final) - gold(SFT)), per task")
    print("reference = SFT baseline (same fp16 run)\n")

    # per-seed per-task Delta_gold (item-paired), with bootstrap CI on the item deltas
    per_task_seed_means = {task: [] for task in TASKS}
    for s in range(8):
        seed_block = {}
        have_any = False
        for task in TASKS:
            fin = load_cached(out_path(f"seed{s}_final", task))
            if fin is None:
                continue
            have_any = True
            ids = sorted(sft[task])
            deltas = [fin[i]["gold_logp"] - sft[task][i]["gold_logp"] for i in ids]
            m = float(np.mean(deltas)); lo, hi = boot_ci(deltas)
            seed_block[task] = {"mean_delta_gold": m, "ci95": [lo, hi],
                                "declines": hi < 0}
            per_task_seed_means[task].append(m)
        if have_any:
            summary["by_seed"][f"seed{s}"] = seed_block

    # per-task aggregate over seeds (mean of per-seed means + CI across seeds)
    print(f"  {'task':<12}{'n_seeds':>8}{'mean Dgold':>13}{'95% CI (across seeds)':>26}")
    for task in TASKS:
        ms = per_task_seed_means[task]
        if not ms:
            print(f"  {task:<12}{0:>8}   (no seeds cached)")
            continue
        agg = float(np.mean(ms))
        lo, hi = boot_ci(ms) if len(ms) > 1 else (ms[0], ms[0])
        summary["by_task"][task] = {"n_seeds": len(ms), "mean_delta_gold": agg,
                                    "ci95_across_seeds": [lo, hi],
                                    "per_seed_means": ms}
        print(f"  {task:<12}{len(ms):>8}{agg:>+13.4f}     [{lo:+.4f}, {hi:+.4f}]")

    # cross-task comparison at the aggregate level
    present = [t for t in TASKS if t in summary["by_task"]]
    if len(present) > 1:
        print("\n  Cross-task comparison (aggregate mean Delta_gold over seeds):")
        ranked = sorted(present, key=lambda t: summary["by_task"][t]["mean_delta_gold"])
        for t in ranked:
            d = summary["by_task"][t]["mean_delta_gold"]
            print(f"    {t:<12} Dgold={d:+.4f}   ({'gold deflates' if d < 0 else 'gold inflates'})")
        a, b = ranked[0], ranked[-1]
        gap = summary["by_task"][b]["mean_delta_gold"] - summary["by_task"][a]["mean_delta_gold"]
        print(f"    -> largest decline: {a};  smallest: {b};  spread = {gap:.4f} nats/token")
        summary["cross_task"] = {"most_decline": a, "least_decline": b, "spread": gap}

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(summary, open(os.path.join(OUT_DIR, "gold_3task_summary.json"), "w"), indent=2)
    print(f"\n  wrote {OUT_DIR}/gold_3task_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="all", help="0..7 or 'all'")
    ap.add_argument("--base-model", default=None, help="local Qwen3.5-4B dir (sets DELTA_MODEL_DIR)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--device", default="cpu", help="cpu (default) / mps / cuda")
    ap.add_argument("--n", type=int, default=None, help="limit items per task (debug)")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = list(range(8)) if args.seed == "all" else [int(args.seed)]

    if args.analyze_only:
        analyze(); return

    if args.base_model:
        os.environ["DELTA_MODEL_DIR"] = args.base_model
    device = torch.device(args.device or os.environ.get("DELTA_DEVICE")
                          or ("mps" if torch.backends.mps.is_available() else "cpu"))
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"device={device} dtype={args.dtype}  seeds={seeds}  tasks={TASKS}")

    # load all 3 tasks' items once
    items = {}
    for task in TASKS:
        rows = [json.loads(l) for l in open(ITEMS_FILE.format(task=task)) if l.strip()]
        if args.n:
            rows = rows[:args.n]
        items[task] = rows
        print(f"  {task}: {len(rows)} items")

    tok = T.build_tokenizer(T.TRACK1)

    # base (unquantized fp16) + PeftModel with named adapters (sft + each seed final)
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    src, local_only = T.resolve_model_source(T.TRACK1["model_id"])
    print(f"loading base {src} ({args.dtype}, unquantized) ...")
    base = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=dtype, trust_remote_code=True, local_files_only=local_only).to(device)
    base.eval()
    pm = PeftModel.from_pretrained(base, SFT_ADAPTER, adapter_name="sft")
    pm.eval()
    loaded = {"sft": SFT_ADAPTER}

    def model_for(path, name):
        if name not in loaded:
            pm.load_adapter(path, adapter_name=name)
            loaded[name] = path
        pm.set_adapter(name); pm.eval()
        return pm

    # SFT reference: all 3 tasks (computed/cached once, shared across seeds)
    model_for(SFT_ADAPTER, "sft")
    for task in TASKS:
        score_task(pm, tok, task, items[task], device, "sft", out_path("sft", task))

    # each requested seed @ final: load adapter once, score all 3 tasks through it
    for s in seeds:
        path = FINAL.format(s=s)
        assert os.path.isdir(path), f"missing final adapter: {path}"
        model_for(path, f"seed{s}")
        for task in TASKS:
            score_task(pm, tok, task, items[task], device,
                       f"seed{s}_final", out_path(f"seed{s}_final", task))

    analyze()


if __name__ == "__main__":
    main()
