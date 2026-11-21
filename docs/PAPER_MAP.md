# Paper-to-file map

Repository map for *The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention* ([arXiv:2611.06476](https://arxiv.org/abs/2611.06476)). Back to the [README](../README.md#paper).

Paths are relative to the repository root. The tagged states [`paper-v1`](https://github.com/yuan-phd/delta-filing/tree/paper-v1) and [`paper-v2`](https://github.com/yuan-phd/delta-filing/tree/paper-v2) use the same paths.

Every path below is tracked in this repository.

| Paper component | Files |
|---|---|
| **Protocol** (pre-registration) | `PREREGISTRATION.md`; `notes/` (pre-registration notes): `notes/2026-06-11_note1_behavioral_branch_and_gate_rule.md` (Note 1), `notes/2026-06-18_note2_matrix_prereg_FROZEN.md` (Note 2, frozen), `notes/2026-06-11_note2_matrix_prereg_DRAFT_UNFROZEN.md` (Note 2 draft, kept for comparison), `notes/2026-06-11_note3_chronology_and_disclosures.md` (chronology and disclosures). Pre-registration commits: `e3be4d5` (Note 1 and Note 2 draft, 2026-06-11) and `62ab7c3` (Note 2 frozen, 2026-06-18). `register_decision.py` (frozen decision rule for the register probe) |
| **Training** | `train_pref.py` (length-normalized DPO + SFT retention), `pref_losses.py` (losses and assistant-only retention mask), `run_matrix.py` (training + eval matrix with margin-crossing checkpoints), `kaggle/matrix_kernel.py`, `kaggle/matrix_parallel_kernel.py`, `kaggle/make_model_dataset.py`, `kaggle/RUNBOOK.md`; preference data `training_data/dpo_pytorch.jsonl` (647 pairs); SFT initialization and reference adapter `adapters/sft_pytorch/` (the path `train_pref.py` loads; `kaggle_upload/adapters/sft_pytorch/` is a byte-identical copy in the Kaggle upload snapshot `kaggle_upload/`) |
| **Scoring** | `scorer_v2.py` (four-tier FORMAT_BREAK / PASS / SURFACE_FORM / SEMANTIC_ERROR), `eval_pref.py` (generation + scoring harness); step-0 baseline: `run_baseline_v2.py`, `kaggle/baseline_v2_kernel.py`, outputs in `kaggle_result/A2/` |
| **Suites** | `data_eval/` (see `data_eval/README.md`): frozen suites `data_eval/track1_extraction.jsonl`, `data_eval/track1_json.jsonl`, `data_eval/track1_router.jsonl`, Control C `data_eval/track1_extraction_controlC.jsonl`, and provenance `.bak` files; generation and adjudication: `data_prep/gen_track1_eval.py`, `data_prep/gen_track1_more.py`, `data_prep/expand_eval.py`, `data_prep/adjudicate_router.py`, `data_prep/gen_control_C.py`; hygiene audit `audit_eval_suite.py` with output in `results/hygiene_audit/` |
| **Adapters** | 40 confirmatory checkpoints (8 seeds × 5) under `kaggle_result/final/`: for each seed N in 0–7, `kaggle_result/final/seedN/adapters/pref/track1_dpo_seedN/` holds the final (step 180) adapter at its top level and `ckpt_step20/`, `ckpt_margin1/`, `ckpt_margin3/`, `ckpt_margin5/` subdirectories; per-seed generations, scored results and logs are in `kaggle_result/final/seedN/` |
| **Analyses** | Primary TOST: `primary_analysis.py`; secondaries: `remaining_analyses.py`, `build_results.py`; gold-logp: `gold_replication.py`, `mac_gold_3task.py`, `mac_gold_controls.py`; reward decomposition: `mac_reward_decomposition.py`, `kaggle_upload/kaggle_reward_decomposition_kernel.py`, `kaggle_upload/kaggle_fp16_train_decomp.py`; backend sensitivity: `mac_val_nf4_check.py`, `kaggle_upload/kaggle_val_nf4_98.py`; supplementary secondaries: `mac_supplementary.py`; register probe: `probe_register_shift.py`, `kaggle/register_probe_kernel.py`. Outputs: `analysis_ws/` (including `analysis_ws/PRIMARY_TOST.md` and `analysis_ws/RESULTS.md`), `kaggle_result/reward_decomposition_kernel/`, `kaggle_result/fp16_train_decomposition.json`, `kaggle_result/val_nf4_check.json`, `kaggle_result/val_nf4_98.json`, `kaggle_result/A1/`, `results/`, `RESULTS_track1_matrix.md` |
| **Figures** | `scripts/make_figures.py` regenerates Figures 1–3 (environment: `requirements-paper.txt`). Its output directory, `figures/paper/`, is generated and not tracked |

## Notes on specific files

Each note below was checked against the files it names.

### a. Early protocol and binding protocol

[`PREREGISTRATION.md`](../PREREGISTRATION.md) is the early protocol of the study: four loss conditions (`dpo`, `dpo_clamp`, `ipo`, `cdpo`), seeds 0 to 4, and suites of about 80 items per task (`n ≈ 80`). The binding protocol of the confirmatory study is Note 2, [`notes/2026-06-18_note2_matrix_prereg_FROZEN.md`](../notes/2026-06-18_note2_matrix_prereg_FROZEN.md). It is marked FROZEN, states that it supersedes the 2026-06-11 unfrozen draft, and specifies a single condition (length-normalized DPO, no clamp arm, IPO/cDPO deferred) over 8 seeds.

### b. Register-probe decision note

[`DECISION_NOTE_register.md`](../DECISION_NOTE_register.md) is the frozen decision rule for the register probe, with threshold X = 0.05 nats/token, written before the nf4 GPU run. Note 2 refers to it by name. The table above lists `register_decision.py` but not this file.

### c. Early result summaries

[`analysis_ws/RESULTS.md`](../analysis_ws/RESULTS.md) and [`CONSOLIDATED_REPORT.md`](../CONSOLIDATED_REPORT.md) were written from the nf4 matrix outputs (`train_pref.py` and `eval_pref.py` load the model in 4-bit nf4) before the fp16 analyses. `RESULTS.md` is the output of `analyze.py` over the matrix `results/` and `logs/`, and `CONSOLIDATED_REPORT.md` lists `analyze.py` among its scripts. `CONSOLIDATED_REPORT.md` was committed in `1021b40` (2026-06-21), before the first fp16/bf16 inference scripts (`573f41f`, "Mac fp16/bf16 inference scripts: paired GOLD_PROTECTED (n=8) + reward decomposition"). Some of their labels predate the analyses the paper reports. For example, `RESULTS.md` labels json `PRESERVED` (a max-drop rule from `analyze.py`), and `CONSOLIDATED_REPORT.md` lists the chosen/rejected reward decomposition as `NOT RUN`. The paper's verdicts come from the primary TOST (`analysis_ws/PRIMARY_TOST.md`) and the later analyses in the table.

### d. json seeds within ±5 points

The paper applies the seed-count criterion inclusively (|Δ| ≤ 5) and reports 7/8 json seeds within ±5 points. [`analysis_ws/PRIMARY_TOST.md`](../analysis_ws/PRIMARY_TOST.md) and [`analysis_ws/02_primary_tost.txt`](../analysis_ws/02_primary_tost.txt) (identical files) print 6/8. The reason is that seed 4 is exactly −5.00 points (0.9375 against a baseline of 0.9875), and `primary_analysis.py` computes `(0.9375 - 0.9875) * 100.0 = -5.000000000000004` in floating point, which fails `abs(diff) <= 5.0`. The json verdict is unaffected, because json also fails the item-level interval (90% bootstrap CI [−5.31, −1.95], outside ±5).

### e. Two Kaggle runbooks

[`kaggle_upload/RUNBOOK.md`](../kaggle_upload/RUNBOOK.md) contains all of [`kaggle/RUNBOOK.md`](../kaggle/RUNBOOK.md) plus a section "Script 2 — REWARD DECOMPOSITION (chosen vs rejected) runbook" for `kaggle_upload/kaggle_reward_decomposition_kernel.py`. `kaggle/RUNBOOK.md` does not have this section.

### f. `audit_eval.py` and `audit_eval_suite.py`

The paper names [`audit_eval.py`](../audit_eval.py), and the table lists [`audit_eval_suite.py`](../audit_eval_suite.py) for the hygiene audit. Both are model-free and use `scorer.py`.

- `audit_eval.py` is the scorer-side part of the task-inclusion audit in `PREREGISTRATION.md`. For json, router and extraction it checks whether the metric captures failure (synthetic degenerate, format-broken and content-wrong outputs), format fragility (trailing elaboration and leading prose), and class balance, item count, train/eval leakage against `training_data/dpo_pytorch.jsonl` and gold self-consistency. It prints to stdout and writes no file. Baseline accuracy and headroom are left to `run_baseline_audit.py`.
- `audit_eval_suite.py` is the hygiene audit of the frozen suites. It re-derives the degeneracy floors, item hygiene (prompt uniqueness, gold-as-output self-consistency, leakage against the preference pairs and the SFT data) and class/coverage balance, and writes `results/hygiene_audit/suite_hygiene.json` and `suite_hygiene.txt`. `data_eval/README.md` points to this script for the hygiene checks.
