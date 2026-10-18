#!/usr/bin/env python3
"""Generate the three publication figures for the paper.

All data is transcribed from full_draft.md (Appendix A, F, G.3 and Tables 2/3/4).
Outputs PDF (vector) at column width into the same directory.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- global publication style -------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
})

GREEN = "#1b7837"   # extraction
BLUE  = "#2166ac"   # json
RED   = "#b2182b"   # router

# margins used as the x-axis for Figure 1 (mean router checkpoint margins, Table 2)
MARGINS = np.array([0.75, 1.09, 3.12, 5.13, 6.73])

# ---- Appendix A: per-seed semantic accuracy (step20, margin1, margin3, margin5, final) ----
EXTRACTION = np.array([
    [97.50, 97.50, 98.12, 96.25, 97.50],
    [98.75, 100.00, 98.75, 98.75, 94.38],
    [99.38, 99.38, 99.38, 98.12, 98.12],
    [98.12, 99.38, 98.12, 97.50, 98.12],
    [99.38, 100.00, 99.38, 95.62, 98.75],
    [100.00, 100.00, 99.38, 99.38, 95.00],
    [97.50, 99.38, 97.50, 97.50, 93.75],
    [99.38, 100.00, 100.00, 100.00, 100.00],
])
JSON = np.array([
    [82.50, 93.75, 93.12, 94.38, 92.50],
    [93.75, 95.62, 80.62, 89.38, 96.88],
    [88.75, 88.75, 93.12, 85.00, 95.62],
    [96.88, 95.00, 93.75, 93.75, 96.25],
    [91.88, 93.75, 76.88, 95.62, 93.75],
    [95.00, 95.00, 94.38, 96.25, 96.88],
    [90.62, 95.62, 86.88, 95.00, 94.38],
    [95.62, 93.75, 93.75, 94.38, 95.62],
])
ROUTER = np.array([
    [85.62, 78.12, 59.38, 61.88, 64.38],
    [87.50, 86.88, 61.25, 66.88, 76.88],
    [86.25, 83.75, 61.25, 60.62, 58.13],
    [83.75, 77.50, 65.00, 61.25, 63.12],
    [82.50, 80.62, 54.37, 69.38, 61.88],
    [84.38, 82.50, 44.38, 51.88, 48.75],
    [85.62, 83.75, 61.25, 64.38, 69.38],
    [85.62, 79.38, 62.50, 64.38, 66.25],
])
BASE = {"extraction": 97.50, "json": 98.75, "router": 93.75}


def figure1():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    series = [("extraction", EXTRACTION, GREEN), ("json", JSON, BLUE), ("router", ROUTER, RED)]
    for name, data, color in series:
        for seed in data:
            ax.plot(MARGINS, seed, color=color, alpha=0.28, linewidth=0.7, zorder=1)
        ax.plot(MARGINS, data.mean(axis=0), color=color, linewidth=2.2, zorder=3,
                solid_capstyle="round")
        ax.axhline(BASE[name], color=color, linestyle=(0, (4, 3)), linewidth=0.9,
                   alpha=0.7, zorder=2)
    ax.set_xlabel("Implicit reward margin ($\\beta$-scaled)")
    ax.set_ylabel("Semantic accuracy (%)")
    ax.set_xlim(0.4, 7.0)
    ax.set_ylim(40, 102)
    ax.set_xticks(MARGINS)
    ax.set_xticklabels([f"{m:.2f}" for m in MARGINS])
    legend_elems = [
        Line2D([0], [0], color=GREEN, lw=2.2, label="extraction (equivalent)"),
        Line2D([0], [0], color=BLUE, lw=2.2, label="json (small loss)"),
        Line2D([0], [0], color=RED, lw=2.2, label="router (catastrophic)"),
        Line2D([0], [0], color="0.4", lw=0.9, linestyle=(0, (4, 3)), label="baseline (step 0)"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", frameon=False, ncol=1,
              handlelength=1.8, borderaxespad=0.3)
    ax.annotate("damage window\n(margin $\\approx$1–3)", xy=(3.12, 58.7), xytext=(3.5, 47),
                fontsize=7, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    fig.tight_layout(pad=0.4)
    out = os.path.join(HERE, "fig1_trajectories.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---- Appendix F.1: per-seed chosen/rejected reward at margin1/3/5/final ----
CKPTS = ["margin1", "margin3", "margin5", "final"]
R_CHOSEN = np.array([
    [-0.044, -0.094, -0.001, 0.051],
    [-0.082, -0.060, 0.006, 0.055],
    [-0.039, -0.063, -0.033, 0.052],
    [-0.045, -0.052, -0.033, 0.060],
    [-0.054, -0.074, 0.013, 0.054],
    [-0.047, -0.100, 0.018, 0.059],
    [-0.051, -0.075, -0.023, 0.057],
    [-0.051, -0.057, 0.020, 0.032],
])
R_REJECTED = np.array([
    [-2.087, -5.569, -11.016, -14.518],
    [-2.365, -6.380, -10.146, -15.344],
    [-2.075, -5.714, -10.861, -15.909],
    [-2.024, -5.822, -9.320, -14.615],
    [-2.285, -6.235, -8.973, -12.302],
    [-2.014, -6.196, -9.233, -12.631],
    [-2.005, -5.799, -11.535, -13.346],
    [-2.052, -6.206, -10.166, -14.890],
])


def figure2():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(CKPTS))
    for seed in R_CHOSEN:
        ax.plot(x, seed, color=GREEN, alpha=0.28, linewidth=0.7, zorder=1)
    for seed in R_REJECTED:
        ax.plot(x, seed, color=RED, alpha=0.28, linewidth=0.7, zorder=1)
    ax.plot(x, R_CHOSEN.mean(axis=0), color=GREEN, linewidth=2.3, zorder=3,
            marker="o", markersize=4, label="chosen reward")
    ax.plot(x, R_REJECTED.mean(axis=0), color=RED, linewidth=2.3, zorder=3,
            marker="s", markersize=4, label="rejected reward")
    ax.axhline(0, color="0.6", linewidth=0.7, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(CKPTS)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Reward (per-token policy logp $-$ SFT logp)")
    ax.set_ylim(-17.5, 2.5)
    ax.legend(loc="lower left", frameon=False)
    # label the flat chosen line on the LEFT (away from the right-side inset)
    ax.annotate("chosen flat (nf4):\nretention term anchors", xy=(0.5, 0.0),
                xytext=(0.15, -3.2), fontsize=7, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7))
    ax.annotate("rejected collapse\ndrives full margin", xy=(2.0, -10.2),
                xytext=(0.95, -13.8), fontsize=7, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))
    # inset: mean per-pair rejected reward by axis at final (clear upper-right zone,
    # above the collapsing rejected lines and below the flat chosen line)
    iax = ax.inset_axes([0.66, 0.47, 0.30, 0.32])
    iax.bar([0, 1], [-16.226, -9.704], color=[BLUE, "#5aae61"], width=0.62)
    iax.set_xticks([0, 1])
    iax.set_xticklabels(["style", "tool"], fontsize=6.5)
    iax.set_title("mean per-pair rejected\nreward (nf4)", fontsize=6.5, pad=2)
    iax.tick_params(axis="y", labelsize=6)
    iax.axhline(0, color="0.6", lw=0.5)
    for s in ["top", "right"]:
        iax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    out = os.path.join(HERE, "fig2_reward_decomposition.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---- Appendix G.3: GOLD_PROTECTED paired stat with 95% CI ----
STAT = np.array([0.2561, 0.2425, 0.2125, 0.2389, 0.2953, 0.2172, 0.2477, 0.2666])
CI_LO = np.array([0.2480, 0.2348, 0.2053, 0.2316, 0.2870, 0.2108, 0.2403, 0.2574])
CI_HI = np.array([0.2644, 0.2502, 0.2198, 0.2462, 0.3038, 0.2236, 0.2553, 0.2756])
THRESH = 0.05


def figure3():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    x = np.arange(8)
    yerr = np.vstack([STAT - CI_LO, CI_HI - STAT])
    ax.bar(x, STAT, color=GREEN, alpha=0.85, width=0.66, zorder=2,
           yerr=yerr, capsize=2.5, error_kw=dict(elinewidth=0.9, capthick=0.9))
    ax.axhline(THRESH, color=RED, linestyle="--", linewidth=1.1, zorder=3)
    # label above the bars with an arrow into a between-bar gap (x=3.5), so it
    # never overlaps a bar
    ax.annotate("pre-registered threshold $X = 0.05$", xy=(3.5, THRESH),
                xytext=(3.5, 0.345), ha="center", va="center", fontsize=7.5,
                color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed{i}" for i in range(8)])
    ax.set_ylabel("Paired stat  $\\Delta_{\\rm gold}-\\Delta_{\\rm mean(A,B)}$")
    ax.set_xlabel("Seed")
    ax.set_ylim(0, 0.37)
    ax.set_title("GOLD_PROTECTED: 8/8 seeds (extraction, fp16)", fontsize=8.5)
    fig.tight_layout(pad=0.4)
    out = os.path.join(HERE, "fig3_gold_protected.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
