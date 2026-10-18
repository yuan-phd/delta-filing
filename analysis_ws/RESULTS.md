# RESULTS — capability/margin decoupling (Track 1)

Source: free-generation eval (greedy) + exact-format scorer. Numbers from `results/` + `logs/`. No teacher-forcing.

## 7a Main table — FINAL checkpoint (mean±std over seeds)

| condition | extraction acc | extraction fmt_break | json acc | json fmt_break | router acc | router fmt_break |
|---|---|---|---|---|---|---|
| baseline | 0.975±0.000 | 0.000±0.000 | 0.988±0.000 | 0.000±0.000 | 0.938±0.000 | 0.000±0.000 |
| dpo | 0.970±0.021 | 0.010±0.017 | 0.952±0.015 | 0.002±0.004 | 0.636±0.077 | 0.106±0.070 |

## 7d Capability vs margin (the decoupling curve)

```
[dpo/extraction] baseline=0.975  m=1.1:acc=0.995  m=3.1:acc=0.988  m=5.1:acc=0.979  m=6.7:acc=0.970  max_drop=0.005 -> PRESERVED
[dpo/json] baseline=0.988  m=1.1:acc=0.939  m=3.1:acc=0.891  m=5.1:acc=0.930  m=6.7:acc=0.952  max_drop=0.097 -> PRESERVED
[dpo/router] baseline=0.938  m=1.1:acc=0.816  m=3.1:acc=0.587  m=5.1:acc=0.626  m=6.7:acc=0.636  max_drop=0.351 -> DEGRADED
```

## 7e Clamp causal readout (dpo vs dpo_clamp, final)

- NOT RUN: need both dpo and dpo_clamp final results.

## Figures

- analysis_ws/figures/capability_vs_margin_extraction.png
- analysis_ws/figures/capability_vs_margin_json.png
- analysis_ws/figures/capability_vs_margin_router.png
- analysis_ws/figures/margin_trajectories.png
