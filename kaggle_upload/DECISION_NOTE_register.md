# FROZEN decision note — register-shift control probe

Written BEFORE the nf4 GPU run. The mapping is fixed now; we do not decide which
outcome we got after seeing the data. Threshold X frozen from the existing fp16 probe.

## Question
Does the gold completion's log-prob fall **specifically**, or does **everything**
fall (global distribution drift not targeted at gold)?

## Frozen threshold
**X = 0.05 nats/token** (chosen from the existing fp16 probe; not adjustable after the nf4 run).

## Primary axis — gold vs mean(A, B)
Controls A (wrong valid section) and B (corrupted ticker) are well-formed, level-
comparable wrong completions. Per item, baseline (sft) → final:

    Delta_gold = logp_gold(final) − logp_gold(baseline)
    Delta_mean = mean(logp_A, logp_B)(final) − mean(logp_A, logp_B)(baseline)
    stat_i     = Delta_gold − Delta_mean

Report **mean(stat) + bootstrap 95% CI over items, per seed**.

| condition | verdict | meaning |
|---|---|---|
| mean(stat) ≤ −X **and** CI entirely < 0 | **TARGETED DISPLACEMENT** | likelihood displacement at the probability layer; two-layer thesis (displacement at prob layer, benign at decode layer) holds |
| \|mean(stat)\| < X **or** CI spans 0 | **GLOBAL DRIFT** | everything falls, gold not targeted; headline reverts to bounded-null + global drift (tied to KL). Publishable, weaker |
| mean(stat) ≥ +X **and** CI entirely > 0 | **GOLD PROTECTED** | controls fall ≥X more than gold; unexpected. Report as-is, no thesis rescue |

## Secondary axis — control C (exploratory, direction + CI only, NO threshold)
C = frozen rejected-register paraphrase (verbose/generic; NOT level-comparable to gold).

    Delta_C = logp_C(final) − logp_C(baseline)

- C **rising** from baseline while gold falls → CHES-style evidence that mass is
  moving toward the rejected register.
- C **falling/flat** → no such evidence.

Report mean Delta_C + bootstrap 95% CI; direction only.

## KL-from-reference readout
At each checkpoint: full-vocab KL(policy ‖ reference) averaged over the gold
completion's token positions, mean over the 160 items. Reference = sft_baseline
(the DPO reference). **Global drift is expected to show KL rising with a flat paired
statistic**; targeted displacement shows the paired statistic separating from 0.

## Caveat separation (report template)
- **Load-bearing (expected robust to backend):** the *absolute* gold-logp decline
  (fp16 already: seed0 −0.023→−0.210, seed1 −0.010→−0.082 per token). The nf4 run
  must **confirm this absolute decline reproduces**; it is reported separately.
- **Noise-sensitive:** gap/contrast numbers (stat, C, KL) — reported with CIs.

## Pre-registered renderings (frozen)
- gold: `{"ticker","section","filing_date"}` correct.
- control A: gold with a different valid section id (`probe_register_shift.control_A_section`).
- control B: gold with a different valid ticker (`control_B_ticker`).
- control C: `data_eval/track1_extraction_controlC.jsonl` (frozen, generated once).
All rendered into assistant-turn text via the eval template (`_render`).

## Unit tests (must pass before trusting numbers)
1. Each renderer yields well-formed, distinct completions (A.section ≠ gold, B.ticker ≠ gold).
2. Base model ranks gold > A and gold > B at baseline (C exempt; record its baseline level).
3. A base-passing item has high gold log-prob.
