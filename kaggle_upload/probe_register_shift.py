#!/usr/bin/env python
"""Register-shift CONTROL probe (extended: + mass-logging + field-span deltas).

Per-token completion log-prob at 9 checkpoints (sft_baseline + margin1/3/5/final,
both seeds) on the full 160-item extraction suite, for FOUR completions per item
(gold / control A wrong-section / control B wrong-ticker / control C rejected-
register), plus KL(policy||reference=sft) over the gold completion.

ADDITIONS (all computed INSIDE the existing forwards -- no new forwards, runtime
unchanged; a forward counter asserts this):
 1. "Where does the mass go?" -> register_probe_topk_{backend}.jsonl : per gold
    position {entropy, gold prob+rank, top-5 competitor tokens}, plus per-position
    entropy at control C's positions as a contrast.
 2. Field-token-only deltas: per-token logp over ONLY the differing field span
    (gold vs A = section span; gold vs B = ticker span), each span length-
    normalized over its own tokens. Added to the main rows (4 scalars). The FROZEN
    verdict still uses the full-completion stat; the field stat is exploratory.

Backend: --quant = nf4 (Kaggle GPU, faithful). Default fp16 (local unit tests).
"""
import argparse
import json
import os
import re

import torch
import torch.nn.functional as F

import train_pref as T
import eval_pref as E

ROOT = os.path.dirname(os.path.abspath(__file__))
ITEMS_FILE = os.path.join(ROOT, "data_eval", "track1_extraction.jsonl")
CONTROL_C_FILE = os.path.join(ROOT, "data_eval", "track1_extraction_controlC.jsonl")
RUNS = {0: "kaggle_runs/seed0_delta-filing-aftersmoke-matrix-run1/adapters/pref/track1_dpo_seed0",
        1: "kaggle_runs/seed1_delta-filing-aftersmoke-matrix-run1/adapters/pref/track1_dpo_seed1"}
MARG = {0: {"margin1": 1.02, "margin3": 3.04, "margin5": 5.03, "final": 6.70},
        1: {"margin1": 1.01, "margin3": 3.10, "margin5": 5.04, "final": 7.65}}
SECTION_POOL = ["Item 1A", "Item 7", "Item 8", "Item 1", "Item 7A", "Item 3"]
TICKER_POOL = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "JPM"]
_FWD = [0]   # forward counter (asserts runtime unchanged)


def section_id(s):
    m = re.search(r"\d+[A-Z]?", str(s).upper())
    return m.group(0) if m else None


# ---------------- renderers ----------------
def _base_num(sec):
    m = re.match(r"\d+", section_id(sec) or "")
    return m.group(0) if m else None


def control_A_section(gold_section):
    """A different valid section whose BASE number differs from gold's, so the
    field span is clean (avoids prefix collisions like 'Item 1' vs 'Item 1A')."""
    gb = _base_num(gold_section)
    return next((s for s in SECTION_POOL if _base_num(s) != gb), "Item 99")


def control_B_ticker(gold_ticker):
    return next((t for t in TICKER_POOL if t.upper() != str(gold_ticker).upper()), "ZZZZ")


def _render(tok, user_prompt, completion, max_len=1024):
    pmsgs = [{"role": "system", "content": E.EXTRACTION_SYSTEM_PROMPT},
             {"role": "user", "content": user_prompt}]
    full_text = tok.apply_chat_template(pmsgs + [{"role": "assistant", "content": completion}], tokenize=False)
    prompt_text = tok.apply_chat_template(pmsgs, tokenize=False, add_generation_prompt=True)
    fid = tok.encode(full_text, max_length=max_len, truncation=True)
    pid = tok.encode(prompt_text, max_length=max_len, truncation=True)
    cstart = 0
    for a, b in zip(fid, pid):
        if a != b:
            break
        cstart += 1
    return torch.tensor(fid, dtype=torch.long), cstart


def render_json(tok, p, ticker, section, date):
    return _render(tok, p, json.dumps({"ticker": ticker, "section": section, "filing_date": date}))


def render_all(tok, item, control_c):
    g = item["gold"]
    return {
        "gold": render_json(tok, item["prompt"], g["ticker"], g["section"], g["filing_date"]),
        "A": render_json(tok, item["prompt"], g["ticker"], control_A_section(g["section"]), g["filing_date"]),
        "B": render_json(tok, item["prompt"], control_B_ticker(g["ticker"]), g["section"], g["filing_date"]),
        "C": _render(tok, item["prompt"], control_c),
    }


# ---------------- forward (one per call; returns everything from that single forward) ----------------
def forward_full(model, ids, cstart, device):
    """ONE forward. Returns (avg_logp, per_tok_logp[n], logsm[n,V], labels[n]).
    Completion is short, so [n,V] is bounded (not full-sequence)."""
    _FWD[0] += 1
    x = ids.unsqueeze(0).to(device)
    L = x.shape[1]
    n = L - cstart
    with torch.no_grad():
        logits = model(x).logits
    comp = logits[:, cstart - 1:L - 1, :].float().squeeze(0)        # [n, V]
    logsm = F.log_softmax(comp, dim=-1)                             # [n, V]
    labels = x[0, cstart:L]                                         # [n]
    per_tok = logsm.gather(-1, labels.unsqueeze(-1)).squeeze(-1)    # [n]
    return float(per_tok.sum() / n), per_tok, logsm, labels


def kl_over_positions(pol_logsm, ref_logsm):
    p = pol_logsm.exp()
    return float((p * (pol_logsm - ref_logsm)).sum(-1).mean())


def position_topk(logsm, labels, tok, topk=5):
    """Per gold position: entropy, gold prob+rank, top-5 competitors. From dist only."""
    probs = logsm.exp()
    ent = -(probs * logsm).sum(-1)                                  # [n]
    gold_lp = logsm.gather(-1, labels.unsqueeze(-1)).squeeze(-1)    # [n]
    gold_rank = (logsm > gold_lp.unsqueeze(-1)).sum(-1) + 1         # [n]
    vals, idx = logsm.topk(topk, dim=-1)
    rows = []
    for k in range(logsm.shape[0]):
        top5 = [[int(idx[k, j]), tok.decode([int(idx[k, j])]), float(vals[k, j].exp())]
                for j in range(topk)]
        rows.append({"position": k, "entropy": float(ent[k]),
                     "gold_token": int(labels[k]), "gold_str": tok.decode([int(labels[k])]),
                     "gold_prob": float(gold_lp[k].exp()), "gold_rank": int(gold_rank[k]),
                     "top5": top5})
    return rows


def entropy_per_pos(logsm):
    probs = logsm.exp()
    return [float(e) for e in (-(probs * logsm).sum(-1))]


def field_span(gold_comp, ctrl_comp):
    """LCP/LCS diff -> the differing middle span (start,end) in each completion."""
    g, c = list(map(int, gold_comp)), list(map(int, ctrl_comp))
    p = 0
    while p < len(g) and p < len(c) and g[p] == c[p]:
        p += 1
    s = 0
    while s < (len(g) - p) and s < (len(c) - p) and g[len(g) - 1 - s] == c[len(c) - 1 - s]:
        s += 1
    return (p, len(g) - s), (p, len(c) - s)


def span_mean(per_tok, span):
    a, b = span
    if b <= a:
        return None
    return float(per_tok[a:b].mean())


# ---------------- model loading ----------------
def load_base(quant, device):
    from transformers import AutoModelForCausalLM
    src, local_only = T.resolve_model_source(T.TRACK1["model_id"])
    if quant:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(src, quantization_config=bnb, device_map={"": 0},
                                                    trust_remote_code=True, local_files_only=local_only)
    else:
        base = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float16,
                                                    trust_remote_code=True, local_files_only=local_only).to(device)
    base.eval()
    return base


def checkpoint_list():
    cks = [("sft_baseline", T.TRACK1["sft_adapter"], 0.0)]
    for seed in (0, 1):
        for ck in ["margin1", "margin3", "margin5", "final"]:
            path = RUNS[seed] if ck == "final" else f"{RUNS[seed]}/ckpt_{ck}"
            cks.append((f"seed{seed}_{ck}", path, MARG[seed][ck]))
    return cks


# ---------------- unit tests (renderers + base ranking + field spans) ----------------
def unit_tests(tok, base, device, items, control_c, k=24):
    from peft import PeftModel
    print("=" * 80 + "\nUNIT TESTS\n" + "=" * 80)
    sub = items[:k]
    fld_ok = 0
    for it in sub:
        rr = render_all(tok, it, control_c[it["item_id"]])
        gcomp = rr["gold"][0][rr["gold"][1]:]
        acomp = rr["A"][0][rr["A"][1]:]
        bcomp = rr["B"][0][rr["B"][1]:]
        ga, ca = field_span(gcomp, acomp)
        gb, cb = field_span(gcomp, bcomp)
        a_decode = tok.decode([int(x) for x in acomp[ca[0]:ca[1]]])
        b_decode = tok.decode([int(x) for x in bcomp[cb[0]:cb[1]]])
        sec_id = section_id(it["gold"]["section"])
        a_ok = (ca[1] > ca[0]) and (section_id(a_decode) is not None)        # A span = a (wrong) section value
        b_ok = (cb[1] > cb[0]) and (control_B_ticker(it["gold"]["ticker"]).split()[0][:3].upper() in b_decode.upper()
                                    or len(b_decode.strip()) > 0)             # B span = ticker value, non-empty
        if a_ok and b_ok:
            fld_ok += 1
    print(f"  field span non-empty + decodes (A->section value, B->ticker value): {fld_ok}/{len(sub)}")

    sft = PeftModel.from_pretrained(base, T.TRACK1["sft_adapter"], adapter_name="sft")
    sft.set_adapter("sft"); sft.eval()
    ga = gb = 0; gold_lp = []; c_lp = []
    for it in sub:
        rr = render_all(tok, it, control_c[it["item_id"]])
        lg, _, _, _ = forward_full(sft, *rr["gold"], device)
        la, _, _, _ = forward_full(sft, *rr["A"], device)
        lb, _, _, _ = forward_full(sft, *rr["B"], device)
        lc, _, _, _ = forward_full(sft, *rr["C"], device)
        ga += (lg > la); gb += (lg > lb); gold_lp.append(lg); c_lp.append(lc)
    n = len(sub)
    print(f"  base ranks gold>A: {ga}/{n} ; gold>B: {gb}/{n}")
    print(f"  base gold logp mean {sum(gold_lp)/n:.3f} (high) ; control C baseline level {sum(c_lp)/n:.3f}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--unittest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--topk-out", default=None)
    args = ap.parse_args()
    backend = "nf4" if args.quant else "fp16"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu"))

    items = [json.loads(l) for l in open(ITEMS_FILE) if l.strip()]
    if args.n:
        items = items[:args.n]
    control_c = {o["item_id"]: o["control_c"] for o in (json.loads(l) for l in open(CONTROL_C_FILE))}
    assert all(it["item_id"] in control_c for it in items), "control C missing (run gen_control_C.py)"

    cks = checkpoint_list()
    expected_fwd = len(items) * (1 + len(cks) * 4)
    print(f"backend={backend} device={device} items={len(items)} checkpoints={len(cks)}")
    print(f"RUNTIME: ~{expected_fwd} forwards (={len(items)}x(1 ref + {len(cks)}x4)); "
          f"~{expected_fwd*0.3/60:.0f} min @0.3s/fwd on T4. Mass/field logging adds NO forwards.")

    tok = T.build_tokenizer(T.TRACK1)
    base = load_base(args.quant, device)
    _FWD[0] = 0
    unit_tests(tok, base, device, items, control_c, k=min(len(items), args.n or 24))
    if args.unittest:
        print(f"\n[forward counter] unit tests used {_FWD[0]} forwards. UNIT-TEST-ONLY: skipping trajectory.")
        return

    from peft import PeftModel
    pm = {"m": None}; loaded = {}
    def model_for(path, name):
        if pm["m"] is None:
            pm["m"] = PeftModel.from_pretrained(base, path, adapter_name=name)
        elif name not in loaded:
            pm["m"].load_adapter(path, adapter_name=name)
        loaded[name] = path; pm["m"].set_adapter(name); pm["m"].eval()
        return pm["m"]

    out_path = args.out or f"results/register_probe_{backend}.jsonl"
    topk_path = args.topk_out or f"results/register_probe_topk_{backend}.jsonl"
    os.makedirs("results", exist_ok=True)
    _FWD[0] = 0
    n_rows = 0
    with open(out_path, "w") as fout, open(topk_path, "w") as ftop:
        fout.write(json.dumps({"_header": True, "backend": backend, "n_items": len(items),
                               "checkpoints": [c[0] for c in cks], "X_nats": 0.05,
                               "reference": "sft_baseline"}) + "\n")
        ftop.write(json.dumps({"_header": True, "backend": backend,
                               "note": "per-position mass logging at gold + entropy contrast at C"}) + "\n")
        for it in items:
            rr = render_all(tok, it, control_c[it["item_id"]])
            gcomp = rr["gold"][0][rr["gold"][1]:]
            ga_g, ga_c = field_span(gcomp, rr["A"][0][rr["A"][1]:])   # section span (gold, A)
            gb_g, gb_c = field_span(gcomp, rr["B"][0][rr["B"][1]:])   # ticker span (gold, B)
            # reference gold dist (once per item)
            sft = model_for(T.TRACK1["sft_adapter"], "sft")
            _, _, ref_gold_dist, _ = forward_full(sft, *rr["gold"], device)
            for name, path, margin in cks:
                m = model_for(path, name)
                lg, pt_g, dist_g, lab_g = forward_full(m, *rr["gold"], device)
                la, pt_a, _, _ = forward_full(m, *rr["A"], device)
                lb, pt_b, _, _ = forward_full(m, *rr["B"], device)
                lc, _, dist_c, _ = forward_full(m, *rr["C"], device)
                kl = 0.0 if name == "sft_baseline" else kl_over_positions(dist_g, ref_gold_dist)
                seed = None if name == "sft_baseline" else int(name[4])
                fout.write(json.dumps({
                    "item_id": it["item_id"], "checkpoint": name, "seed": seed, "margin": margin,
                    "logp_gold": lg, "logp_A": la, "logp_B": lb, "logp_C": lc, "kl_gold": kl,
                    # field-span (length-normalized over each span's own tokens)
                    "logp_gold_secspan": span_mean(pt_g, ga_g), "logp_A_secspan": span_mean(pt_a, ga_c),
                    "logp_gold_tkspan": span_mean(pt_g, gb_g), "logp_B_tkspan": span_mean(pt_b, gb_c),
                }) + "\n")
                # mass logging (gold positions: full; C positions: entropy contrast)
                for r in position_topk(dist_g, lab_g, tok):
                    ftop.write(json.dumps({"item_id": it["item_id"], "checkpoint": name,
                                           "completion": "gold", **r}) + "\n")
                for pos, e in enumerate(entropy_per_pos(dist_c)):
                    ftop.write(json.dumps({"item_id": it["item_id"], "checkpoint": name,
                                           "completion": "C", "position": pos, "entropy": e}) + "\n")
                n_rows += 1
            del ref_gold_dist
    assert _FWD[0] == expected_fwd, f"forward count {_FWD[0]} != expected {expected_fwd} (runtime changed!)"
    print(f"\nwrote {n_rows} rows -> {out_path}")
    print(f"wrote mass log -> {topk_path}")
    print(f"[forward counter] {_FWD[0]} forwards == expected {expected_fwd}  (runtime UNCHANGED) ✓")
    print("Next: python register_decision.py --probe " + out_path)


if __name__ == "__main__":
    main()
