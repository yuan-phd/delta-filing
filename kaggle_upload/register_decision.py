#!/usr/bin/env python
"""FROZEN decision rule for the register-shift control probe (X = 0.05 nats/token).

Reads the per-item probe output (register_probe_{backend}.jsonl) and applies the
pre-registered mapping below to the PRIMARY axis. The mapping is fixed BEFORE the
nf4 run and is not adjustable after seeing data. See DECISION_NOTE_register.md.

Primary axis (per item, baseline -> final):
    Delta_gold   = logp_gold(final)  - logp_gold(baseline)
    Delta_mean   = mean(logp_A, logp_B)(final) - mean(...)(baseline)
    stat_i       = Delta_gold - Delta_mean
    report mean(stat) + bootstrap 95% CI over items, PER SEED.

Outcome mapping (X = 0.05):
    mean(stat) <= -X  AND  CI entirely < 0   -> TARGETED DISPLACEMENT  (two-layer thesis holds)
    |mean(stat)| < X   OR  CI spans 0         -> GLOBAL DRIFT           (bounded-null + global drift)
    mean(stat) >= +X  AND  CI entirely > 0   -> GOLD PROTECTED         (unexpected, report as-is)

Secondary axis (C, exploratory, direction + CI only, NO threshold):
    Delta_C = logp_C(final) - logp_C(baseline); rising while gold falls -> CHES-style
    evidence of mass toward the rejected register; falling/flat -> no such evidence.

KL readout: mean KL(policy||reference) over gold positions per checkpoint. Global
drift expected to show KL rising with a flat paired statistic.

  python register_decision.py --probe results/register_probe_nf4.jsonl
"""
import argparse
import json
from collections import defaultdict

X = 0.05            # nats/token, FROZEN
N_BOOT = 10000
BOOT_SEED = 12345   # fixed for reproducibility


def bootstrap_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def decide(mean_stat, lo, hi):
    if mean_stat <= -X and hi < 0:
        return "TARGETED_DISPLACEMENT"
    if mean_stat >= +X and lo > 0:
        return "GOLD_PROTECTED"
    return "GLOBAL_DRIFT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.probe) if l.strip()]
    rows = [r for r in rows if not r.get("_header")]

    # index by item -> checkpoint -> row
    by = defaultdict(dict)
    for r in rows:
        by[r["item_id"]][r["checkpoint"]] = r

    print(f"FROZEN decision rule  X={X} nats/token  (see DECISION_NOTE_register.md)")
    print(f"probe={args.probe}  items={len(by)}")
    kl_by_ck = defaultdict(list)
    for r in rows:
        kl_by_ck[r["checkpoint"]].append(r["kl_gold"])

    for seed in (0, 1):
        final = f"seed{seed}_final"
        stat, dgold, dmeanAB, dC, fstat = [], [], [], [], []
        for item, cks in by.items():
            if "sft_baseline" not in cks or final not in cks:
                continue
            b, f = cks["sft_baseline"], cks[final]
            dg = f["logp_gold"] - b["logp_gold"]
            dm = ((f["logp_A"] + f["logp_B"]) / 2) - ((b["logp_A"] + b["logp_B"]) / 2)
            stat.append(dg - dm); dgold.append(dg); dmeanAB.append(dm)
            dC.append(f["logp_C"] - b["logp_C"])
            # field-span stat (exploratory): advantage of gold over controls on the
            # differing field tokens only (section vs A, ticker vs B), baseline->final
            fk = ("logp_gold_secspan", "logp_A_secspan", "logp_gold_tkspan", "logp_B_tkspan")
            if all(f.get(k) is not None and b.get(k) is not None for k in fk):
                adv_f = ((f["logp_gold_secspan"] - f["logp_A_secspan"]) +
                         (f["logp_gold_tkspan"] - f["logp_B_tkspan"])) / 2
                adv_b = ((b["logp_gold_secspan"] - b["logp_A_secspan"]) +
                         (b["logp_gold_tkspan"] - b["logp_B_tkspan"])) / 2
                fstat.append(adv_f - adv_b)
        n = len(stat)
        if n == 0:
            print(f"\nseed{seed}: NO DATA"); continue
        ms = sum(stat) / n; lo, hi = bootstrap_ci(stat)
        verdict = decide(ms, lo, hi)
        print(f"\n=== seed{seed}  (n={n}) ===")
        print(f"  [absolute, load-bearing]  mean Delta_gold = {sum(dgold)/n:+.4f}  "
              f"mean Delta_mean(A,B) = {sum(dmeanAB)/n:+.4f}")
        print(f"  [PRIMARY paired]          mean(stat=Dgold-Dmean) = {ms:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  -> PRIMARY VERDICT: {verdict}")
        if fstat:
            mf = sum(fstat) / len(fstat); flo, fhi = bootstrap_ci(fstat)
            print(f"  [FIELD-SPAN stat, exploratory] mean = {mf:+.4f}  95% CI [{flo:+.4f}, {fhi:+.4f}]  "
                  f"(differing tokens only; verdict still uses full stat)")
        mc = sum(dC) / n; clo, chi = bootstrap_ci(dC)
        cdir = "RISING (CHES-style: mass toward rejected register)" if (mc > 0 and clo > 0) \
            else "FALLING/FLAT (no CHES evidence)" if (mc <= 0 or chi < 0 or clo < 0) else "AMBIGUOUS"
        print(f"  [SECONDARY C, exploratory] mean Delta_C = {mc:+.4f}  95% CI [{clo:+.4f}, {chi:+.4f}] -> {cdir}")
        kl_final = sum(kl_by_ck[final]) / len(kl_by_ck[final])
        print(f"  [KL readout]              mean KL(policy||ref) at final = {kl_final:.4f} nats/token")

    print("\nKL trajectory (mean over items):")
    for ck in ["sft_baseline", "seed0_margin1", "seed0_margin3", "seed0_margin5", "seed0_final",
               "seed1_margin1", "seed1_margin3", "seed1_margin5", "seed1_final"]:
        if kl_by_ck.get(ck):
            print(f"   {ck:<16} {sum(kl_by_ck[ck])/len(kl_by_ck[ck]):.4f}")


if __name__ == "__main__":
    main()
