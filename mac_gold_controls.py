#!/usr/bin/env python
"""Script 1 (priority) — full paired GOLD_PROTECTED statistic at n=8, on Mac (fp16/bf16).

Closes the #1 gap from the matrix analysis: the matrix logged gold-ONLY, so the probe's
PAIRED statistic stat = Delta_gold - Delta_mean(A,B) could only be evaluated on the 2 A1
probe seeds. This recomputes it for ALL 8 matrix seeds at the FINAL checkpoint by teacher-
forcing gold + control A + control B for the 160 extraction items, plus the SFT baseline.

FAITHFUL METHOD — reuses the frozen probe code verbatim:
  * renderers      : probe_register_shift.render_json / control_A_section / control_B_ticker
                     (control A = a different VALID section id; control B = a different ticker;
                     exactly the completions the A1 probe scored as logp_A / logp_B). The
                     controlC.jsonl file holds only control C, which the paired stat does NOT
                     use, so it is not needed here.
  * teacher-forcing: probe_register_shift.forward_full  (completion-only, length-normalized
                     per-token mean log-prob — identical to the training method
                     pref_losses.compute_log_probs and to the A1 probe).
  * decision rule  : register_decision.decide / bootstrap_ci / X=0.05  (DECISION_NOTE_register.md).
Only difference vs A1: unquantized fp16/bf16 (Mac, no nf4) and 8 seeds instead of 2.

  python mac_gold_controls.py --seed all          # all 8 seeds (hours)
  python mac_gold_controls.py --seed 0            # one seed (reuses cached SFT reference)
  python mac_gold_controls.py --analyze-only      # just re-run the stat from cached jsonl
Outputs -> analysis_ws/gold_controls/{sft.jsonl, seed{S}_final.jsonl, gold_controls_summary.json}
"""
import argparse
import json
import os
import time

import torch

import train_pref as T
import probe_register_shift as P
import register_decision as RD

ROOT = os.path.dirname(os.path.abspath(__file__))
ITEMS_FILE = os.path.join(ROOT, "data_eval", "track1_extraction.jsonl")
SFT_ADAPTER = T.TRACK1["sft_adapter"]                     # adapters/sft_pytorch
OUT_DIR = os.path.join(ROOT, "analysis_ws", "gold_controls")
FINAL = "kaggle_result/final/seed{s}/adapters/pref/track1_dpo_seed{s}"   # final = top dir


def gold_AB_renders(tok, item):
    """The 3 renders the paired stat needs (gold, A=wrong-section, B=wrong-ticker)."""
    g = item["gold"]
    return {
        "gold": P.render_json(tok, item["prompt"], g["ticker"], g["section"], g["filing_date"]),
        "A": P.render_json(tok, item["prompt"], g["ticker"], P.control_A_section(g["section"]), g["filing_date"]),
        "B": P.render_json(tok, item["prompt"], P.control_B_ticker(g["ticker"]), g["section"], g["filing_date"]),
    }


def score_checkpoint(model, tok, items, device, label):
    """Per-item {item_id, gold_logp, controlA_logp, controlB_logp} for one adapter."""
    rows = []
    t0 = time.time()
    for i, it in enumerate(items):
        r = gold_AB_renders(tok, it)
        lg, *_ = P.forward_full(model, *r["gold"], device)
        la, *_ = P.forward_full(model, *r["A"], device)
        lb, *_ = P.forward_full(model, *r["B"], device)
        rows.append({"item_id": it["item_id"], "gold_logp": lg,
                     "controlA_logp": la, "controlB_logp": lb})
        if (i + 1) % 20 == 0 or (i + 1) == len(items):
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(items) - i - 1)
            print(f"    [{label}] {i+1}/{len(items)} items  {dt:.0f}s  ETA {eta:.0f}s", flush=True)
    return rows


def load_cached(path):
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path)]
        if len(rows) == 160:
            return {r["item_id"]: r for r in rows}
    return None


def compute_for(model_factory, tok, items, device, label, out_path):
    cached = load_cached(out_path)
    if cached is not None:
        print(f"  [{label}] cached -> {out_path} (skip)")
        return cached
    print(f"  [{label}] computing {len(items)} items ...")
    model = model_factory()
    rows = score_checkpoint(model, tok, items, device, label)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  [{label}] wrote -> {out_path}")
    return {r["item_id"]: r for r in rows}


def analyze(seeds):
    """Paired stat per seed vs cached SFT reference + frozen decision mapping."""
    sft = load_cached(os.path.join(OUT_DIR, "sft.jsonl"))
    if sft is None:
        print("  analyze: no cached sft.jsonl — run a seed first."); return
    ids = sorted(sft)
    print("\n" + "=" * 74)
    print(f"PAIRED GOLD_PROTECTED STATISTIC  (X={RD.X}, ref=sft_baseline, frozen rule)\n")
    summary, n_gp = {}, 0
    have = []
    for s in seeds:
        fin = load_cached(os.path.join(OUT_DIR, f"seed{s}_final.jsonl"))
        if fin is None:
            continue
        have.append(s)
        dgold = [fin[i]["gold_logp"] - sft[i]["gold_logp"] for i in ids]
        dctrl = [0.5 * (fin[i]["controlA_logp"] + fin[i]["controlB_logp"])
                 - 0.5 * (sft[i]["controlA_logp"] + sft[i]["controlB_logp"]) for i in ids]
        stat = [g - c for g, c in zip(dgold, dctrl)]
        mean = sum(stat) / len(stat)
        lo, hi = RD.bootstrap_ci(stat)
        verdict = RD.decide(mean, lo, hi)
        n_gp += (verdict == "GOLD_PROTECTED")
        mg = sum(dgold) / len(dgold); mc = sum(dctrl) / len(dctrl)
        summary[f"seed{s}"] = {"mean_stat": mean, "ci95": [lo, hi], "verdict": verdict,
                               "mean_delta_gold": mg, "mean_delta_ctrlAB": mc}
        print(f"  seed{s}: stat={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"(Dgold={mg:+.4f} Dctrl={mc:+.4f})  -> {verdict}")
    print(f"\n  AGGREGATE: GOLD_PROTECTED on {n_gp}/{len(have)} seeds evaluated "
          f"(seeds {have}).")
    summary["_aggregate"] = {"gold_protected": n_gp, "n_seeds": len(have),
                             "seeds": have, "X": RD.X, "reference": "sft_baseline",
                             "method": "fp16/bf16 unquantized, paired stat Dgold-Dmean(A,B)"}
    json.dump(summary, open(os.path.join(OUT_DIR, "gold_controls_summary.json"), "w"), indent=2)
    print(f"  wrote {OUT_DIR}/gold_controls_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="all", help="0..7 or 'all'")
    ap.add_argument("--base-model", default=None, help="local Qwen3.5-4B dir (sets DELTA_MODEL_DIR)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--device", default=None, help="mps (default on M1) / cpu")
    ap.add_argument("--n", type=int, default=None, help="limit items (debug)")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = list(range(8)) if args.seed == "all" else [int(args.seed)]

    if args.analyze_only:
        analyze(list(range(8))); return

    if args.base_model:
        os.environ["DELTA_MODEL_DIR"] = args.base_model
    device = torch.device(args.device or os.environ.get("DELTA_DEVICE")
                          or ("mps" if torch.backends.mps.is_available() else "cpu"))
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"device={device} dtype={args.dtype}  seeds={seeds}")

    items = [json.loads(l) for l in open(ITEMS_FILE) if l.strip()]
    if args.n:
        items = items[:args.n]
    tok = T.build_tokenizer(T.TRACK1)

    # base (unquantized) + PeftModel with named adapters (sft + each seed final)
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

    # SFT reference (shared, computed/cached once)
    compute_for(lambda: model_for(SFT_ADAPTER, "sft"), tok, items, device,
                "sft_baseline", os.path.join(OUT_DIR, "sft.jsonl"))
    # each requested seed @ final
    for s in seeds:
        path = FINAL.format(s=s)
        assert os.path.isdir(path), f"missing final adapter: {path}"
        compute_for(lambda p=path, n=f"seed{s}": model_for(p, n), tok, items, device,
                    f"seed{s}_final", os.path.join(OUT_DIR, f"seed{s}_final.jsonl"))

    analyze(list(range(8)))   # report over whatever is cached on disk


if __name__ == "__main__":
    main()
