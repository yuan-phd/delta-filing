# Experimental Plan — Bounded vs Unbounded Preference Optimization and the Format-Dissociation Failure Mode

*Audience: the researcher (you). This is the "what and why." The companion file `claude_code_spec.md` is the "how" for execution.*

---

## 0. One-paragraph thesis (locked)

In small-model (3–4B) + LoRA agentic multi-task deployment, the alignment tax from preference optimization shows a **counterintuitive task dissociation**. An unbounded objective (DPO) lets the implicit reward margin grow without bound; the output distribution shifts toward the verbose, "preferred" style; and **rigid short-format tasks (binary classification) suffer a format-break collapse while complex structured outputs (JSON tool calls) survive intact** — in the *same* model, so it is not capability loss. Any bounded objective (IPO's quadratic penalty, cDPO's label smoothing) pins the margin and removes the collapse. The switch is **binary** (bounded vs unbounded), not a gradient of constraint strength.

The paper proves one causal chain:

> **loss boundedness → margin trajectory → output verbosity → format-break rate on rigid-short tasks**

---

## 1. The single risk this entire design exists to defeat

A sharp reviewer will say:

> "Output length ordering is: binary label (~2 tokens) < JSON call (~30 tokens) < analysis (hundreds). DPO drifts toward verbosity (this is the well-known DPO length-exploitation, Park 2024 / Rafailov 'Disentangling Length'). So *the shortest task dies hardest*. Your 'dissociation' is a restatement of length bias. No contribution."

**This attack currently succeeds against our data**, because the pattern (binary dies, JSON survives, analysis survives) is roughly length-ordered. The burden of proof is on us.

**The only thing that defeats it — and the only reason this paper is interesting — is this comparison being true:**

> Complex JSON (≈30 tokens, also a *short, exact, non-verbose* output) survives essentially intact, while the binary label (≈2 tokens) collapses.

If pure verbosity drift were the mechanism, it should damage **both** short tasks — a model drifting verbose should also struggle to emit a clean single-line JSON and stop. The fact that JSON survives means the model can still produce a constrained exact-format output **when the format has syntactic anchoring** (braces, quotes, keys), and fails **only** when the format is a bare token with no anchor. That reframes the finding from "shortest dies" (length bias, known) to **"syntactically-anchored formats are robust to verbosity drift; bare-token formats are defenseless"** (not length, and not previously characterized).

**Bottom line: the whole paper's value rests on the length-controlled analysis in §5 confirming this.** If it confirms → a sharp, modest, real finding. If it shows pure length monotonicity → we honestly narrow the claim to "an applied confirmation of DPO length exploitation in an agentic-SLM setting" (weaker, arXiv-only). We design to find out, not to assume.

---

## 2. Two-track structure

**Track 1 — in-domain (your SEC system).** The motivating, "in-the-wild" result. Same setup you already built; now run cleanly and at adequate scale.

**Track 2 — public (Llama-3.2-3B-Instruct + UltraFeedback).** Kills the "bespoke benchmark" objection and tests generality across model family *and* preference dataset. If the same dissociation appears here, it is not an artifact of your SEC data or of Qwen.

Both tracks measure the identical dissociation with the identical two-stage scoring.

---

## 3. Conditions compared

| Condition | Loss | Bounded? | Role |
|---|---|---|---|
| Baseline | none (SFT / instruct model as-is) | — | Reference point; measures pre-PO ability |
| DPO | `-logσ(β·m)` | No | The failure case |
| **DPO+clamp** | `-logσ(β·clamp(m,−c,c))` | **Yes, same loss family** | **Causal control** |
| IPO | `(clamp(m,−10,10) − 1/(2β))²` | Yes | Bounded via different mechanism |
| cDPO | `-(1−ε)logσ(m) − ε·logσ(−m)` | Yes | Bounded via different mechanism |

**Why DPO+clamp is non-negotiable.** DPO vs IPO differ in *two* things at once: boundedness *and* loss shape (logsigmoid vs quadratic). A reviewer will say we never isolated boundedness. DPO+clamp holds the loss family fixed (still logsigmoid) and changes *only* whether the margin is bounded. If clamping removes the collapse, we can upgrade the claim from "correlated with boundedness" to "**caused by** boundedness." It costs one line of code. It is the difference between a correlational and a causal paper.

**Baseline in each track:** Track 1 baseline = your existing SFT adapter; PO starts from SFT (reference = SFT). Track 2 baseline = Llama-3.2-3B-Instruct as-is; PO starts from it (reference = the instruct model).

---

## 4. Scoring protocol — fully programmatic, zero human annotation

Every generation is scored in two stages:

1. **Format gate.** Does the output conform to the required format?
   - Binary classification: stripped/normalized output ∈ the valid label set.
   - JSON / tool call: parses as JSON *and* contains the required keys/arguments.
   - IFEval: passes the official instruction verifier.
   - BoolQ: normalized output ∈ {yes, no}.
2. **Content** (only evaluated if the format gate passed): is the answer correct vs gold?

Each failure is tagged **FORMAT_BREAK** (failed stage 1) or **REASONING_ERROR** (passed stage 1, failed stage 2). This tag *is* the central evidence: our claim is that DPO's damage is overwhelmingly FORMAT_BREAK on the binary task and near-zero on JSON — i.e. distribution shift, not capability loss.

**Normalization rule must be explicit and logged.** Whether `COMPLETE.` (trailing period) counts as a pass changes the story. We define the rule (e.g. strip whitespace + trailing punctuation, case-insensitive exact match to label) and we **log how many passes required normalization**, so a reviewer can see the format-break rate is not an artifact of an overly strict regex.

This protocol is why **no human annotation is needed**: every axis of the core claim (binary, JSON, IFEval, BoolQ) has an objective programmatic verdict. Free-form analysis tasks (no unique answer) are **not** part of the core claim; if reported at all, they go in an appendix with an LLM-judge and a robustness note, never as primary evidence.

---

## 5. The length-controlled analysis (the hinge — §1)

Three things, in order of importance:

1. **Per-task mean output length, per condition.** Establishes that DPO inflates length (expected) and that bounded methods inflate less.
2. **The headline comparison.** DPO's FORMAT_BREAK rate on the binary task vs on the JSON task. The claim lives or dies on JSON-format-break staying low (model retains exact-format ability) while binary-format-break spikes.
3. **Within-length-bucket control.** Pool all generations, bucket by output length. Within comparable length buckets, compare FORMAT_BREAK rates of binary vs JSON. If JSON survives at lengths where binary breaks, the dissociation is **not** explained by length alone. This is the explicit rebuttal to the §1 attack, and it must be in the paper as a figure.

We report this analysis **even if it undercuts us.** If buckets show pure length monotonicity, we say so and narrow the claim. Designing to find out is the point.

---

## 6. Multi-seed and statistical honesty

- Track 1: **5 seeds** per condition. Track 2: **3 seeds** per condition.
- Report **mean ± std and every per-seed raw value** (full transparency table in appendix).
- With 3–5 seeds we **do not compute p-values** (underpowered, would be misleading). We report effect sizes and the raw spread, and we state plainly: *single architecture per track, N seeds, preliminary.*
- Multi-seed is the one thing that converts "anecdote" to "signal." It is orthogonal to everything else and cannot be skipped.

---

## 7. What is deliberately excluded, and why

- **GRPO** — not in the core experiments. It is a different paradigm (online RL vs offline preference loss), and we have **no single-objective GRPO control**, so any "multi-objective avoids the trade-off" claim is unsupported. Including it invites a fatal objection for zero benefit. At most: one sentence as a separate observation, no multi-seed, clearly flagged.
- **The KL k1→k3 estimator story** — it is textbook (Schulman 2020; TRL defaults to k3). Not a research contribution. It can be a standalone blog post; it does **not** go in this paper.
- **Human annotation** — designed out entirely (see §4).
- **Novelty posture** — we do **not** claim a new theory or to "refute" anything. The mechanism (unbounded margin → verbosity → OOD) is known. Our contribution is the **specific dissociation, the diagnostic protocol, and the applied agentic-SLM regime.** The README's strong phrasings ("refutes the strong version of the claim," "GRPO avoids the trade-off entirely") are removed.

---

## 8. Run matrix and compute reality

**Training runs:**

| Track | Conditions | Seeds | Runs |
|---|---|---|---|
| 1 (SEC, Qwen3.5-4B+SFT) | DPO, DPO+clamp, IPO, cDPO | 5 | 20 |
| 2 (Llama-3.2-3B-Instruct) | DPO, DPO+clamp, IPO, cDPO | 3 | 12 |
| | | **Total** | **32** + 2 baselines (eval-only) |

Each run: 1 epoch, a few-thousand preference pairs, 3–4B in 4-bit on T4 → roughly 1–4 h each. On Kaggle free tier (2×T4, ~30 h/week) this is a few weeks of wall-clock with queueing. **This is real work, not "mostly writing."** You said no time pressure — good, because this needs it.

**Eval** is cheap (inference + programmatic scoring), but must **save every generation** so the two-stage tagging and length analysis are possible. Your current pipeline discards generations — that is the first thing to fix.

---

## 9. Honest expected outcome

- **If §5 confirms the dissociation breaks length-monotonicity** and **§3's clamp removes the collapse**: a clean, modest, defensible workshop submission. Realistic acceptance at a competitive workshop ≈ coin-flip, dominated by reviewer draw and framing discipline.
- **If the hinge shows pure length effect**: still a legitimate arXiv preprint (an applied, controlled confirmation with a reusable diagnostic protocol), just a weaker claim. Either way the artifact demonstrates rigorous research execution — which is your actual goal.

The downside is hedged: a rigorously executed study is valuable on arXiv whether or not the workshop accepts it. The only true failure modes are sloppy execution or over-claiming — both designed out here.
