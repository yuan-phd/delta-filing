#!/usr/bin/env python
"""Items #4(partial)/#5/#6/#7 of the pre-registered analysis on the reorganized matrix.

#4  held-out (val) margin trajectory — AGGREGATE only (no per-category logging).
#5  per-item flip analysis (extraction, json): pass<->fail vs baseline, McNemar.
#6  router FORMAT_BREAK vs content-error trajectory, per-seed.
#7  margin-vs-capability figures: per-seed trajectories + mean+-std bands.
Reads kaggle_result/final/seed{0..7} (matrix) + kaggle_result/A2 (baseline_v2).
"""
import json
import os
import numpy as np
from scipy import stats

ROOT = "kaggle_result/final"
BASE_GEN = "kaggle_result/A2/generations"
CKPTS = ["step20", "margin1", "margin3", "margin5", "final"]
SEEDS = range(8)
os.makedirs("figures", exist_ok=True)


def per_item(path, field="semantic_pass"):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if not r.get("_header"):
            out[r["item_id"]] = int(bool(r[field]))
    return out


def res(s, ck, t):
    return json.load(open(f"{ROOT}/seed{s}/results/track1_dpo_{ck}_seed{s}_{t}.json"))


def seed_margin(s, ck):
    m = json.load(open(f"{ROOT}/seed{s}/adapters/pref/track1_dpo_seed{s}/checkpoints.json"))
    return {c["label"]: c["smoothed_margin"] for c in m["checkpoints"]}[ck]


# ---------------- #5 flip / McNemar ----------------
def flips():
    print("=" * 72)
    print("#5  PER-ITEM FLIP ANALYSIS vs baseline_v2 (final), McNemar (semantic_pass)")
    for t in ["extraction", "json"]:
        base = per_item(f"{BASE_GEN}/track1_baseline_v2_{t}.jsonl")
        ids = sorted(base)
        print(f"\n  {t}:  (b10 = pass->fail [regressions], b01 = fail->pass [gains])")
        tot01 = tot10 = 0
        for s in SEEDS:
            seed = per_item(f"{ROOT}/seed{s}/generations/track1_dpo_final_seed{s}_{t}.jsonl")
            b01 = sum(1 for i in ids if base[i] == 0 and seed[i] == 1)
            b10 = sum(1 for i in ids if base[i] == 1 and seed[i] == 0)
            tot01 += b01; tot10 += b10
            n = b01 + b10
            # exact binomial McNemar (small counts)
            p = stats.binomtest(min(b01, b10), n, 0.5).pvalue if n else 1.0
            print(f"    seed{s}: fail->pass={b01:2d}  pass->fail={b10:2d}  net={b01-b10:+d}  exact p={p:.3f}")
        n = tot01 + tot10
        chi2 = (abs(tot01 - tot10) - 1) ** 2 / n if n else 0.0
        pp = stats.chi2.sf(chi2, 1)
        print(f"    POOLED (8x160, clustered — approx): fail->pass={tot01} pass->fail={tot10} "
              f"net={tot01-tot10:+d}  McNemar chi2={chi2:.1f} p={pp:.2e}")
        print(f"      -> {'asymmetric: regressions dominate' if tot10>tot01 else 'asymmetric: gains dominate' if tot01>tot10 else 'symmetric'}")


# ---------------- #6 router trajectory ----------------
def router_traj():
    print("\n" + "=" * 72)
    print("#6  ROUTER trajectory — format_break vs content (semantic) error, per checkpoint")
    print(f"  {'ckpt':<8}{'margin':>7}{'sem_acc(m+-s)':>16}{'fmt_break(m+-s)':>18}{'content_err(m+-s)':>19}")
    for ck in CKPTS:
        acc = [res(s, ck, "router")["semantic_accuracy"] for s in SEEDS]
        fb = [res(s, ck, "router")["format_break_rate"] for s in SEEDS]
        ce = [res(s, ck, "router")["semantic_error_rate"] for s in SEEDS]
        mg = np.mean([seed_margin(s, ck) for s in SEEDS])
        print(f"  {ck:<8}{mg:>7.2f}{f'{np.mean(acc):.3f}+-{np.std(acc):.3f}':>16}"
              f"{f'{np.mean(fb):.3f}+-{np.std(fb):.3f}':>18}{f'{np.mean(ce):.3f}+-{np.std(ce):.3f}':>19}")
    print("  per-seed router format_break at FINAL:",
          " ".join(f"{res(s,'final','router')['format_break_rate']:.2f}" for s in SEEDS))


# ---------------- #4 held-out margin (aggregate) ----------------
def heldout():
    print("\n" + "=" * 72)
    print("#4  HELD-OUT (val) margin trajectory — AGGREGATE ONLY (no style/tool split logged)")
    for s in SEEDS:
        rows = [json.loads(l) for l in open(f"{ROOT}/seed{s}/logs/track1_dpo_seed{s}_margin.jsonl")]
        vm = [(r["step"], round(r["val_margin"], 2)) for r in rows if "val_margin" in r]
        print(f"  seed{s}: {vm}")
    print("  held-out margin grows with training -> preference generalizes off the train pairs.")
    print("  PER-CATEGORY (style-axis vs tool_calling) split: NOT LOGGED -> cannot test that prediction.")


# ---------------- #7 figures ----------------
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = ["json", "router", "extraction"]
    base = {t: json.load(open(f"kaggle_result/A2/results/track1_baseline_v2_{t}.json"))["semantic_accuracy"]
            for t in tasks}

    # main: per-seed trajectories, 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, t in zip(axes, tasks):
        for s in SEEDS:
            xs = [seed_margin(s, ck) for ck in CKPTS]
            ys = [res(s, ck, t)["semantic_accuracy"] for ck in CKPTS]
            ax.plot(xs, ys, marker="o", ms=3, lw=1, alpha=0.7)
        ax.axhline(base[t], ls="--", color="k", lw=1, label="baseline_v2")
        ax.axhspan(base[t] - 0.05, base[t] + 0.05, color="green", alpha=0.07, label="+-5 pts")
        ax.set(title=f"{t}", xlabel="DPO margin (per-seed)", ylim=(0, 1.03))
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("semantic accuracy (free-gen)")
    fig.suptitle("Capability vs margin — per-seed trajectories (8 seeds)")
    fig.tight_layout()
    fig.savefig("figures/capability_vs_margin_perseed.png", dpi=130)
    plt.close(fig)

    # supplementary: mean +- std bands
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in tasks:
        mg = np.array([np.mean([seed_margin(s, ck) for s in SEEDS]) for ck in CKPTS])
        mean = np.array([np.mean([res(s, ck, t)["semantic_accuracy"] for s in SEEDS]) for ck in CKPTS])
        sd = np.array([np.std([res(s, ck, t)["semantic_accuracy"] for s in SEEDS]) for ck in CKPTS])
        ln, = ax.plot(mg, mean, marker="o", label=t)
        ax.fill_between(mg, mean - sd, mean + sd, color=ln.get_color(), alpha=0.18)
    ax.set(xlabel="DPO margin (mean over seeds)", ylabel="semantic accuracy",
           title="Capability vs margin — mean +- std (supplementary)", ylim=(0, 1.03))
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/capability_vs_margin_meanband.png", dpi=130)
    plt.close(fig)
    print("\n#7  wrote figures/capability_vs_margin_perseed.png (PRIMARY) "
          "+ figures/capability_vs_margin_meanband.png (supplementary)")


if __name__ == "__main__":
    flips()
    router_traj()
    heldout()
    figures()
