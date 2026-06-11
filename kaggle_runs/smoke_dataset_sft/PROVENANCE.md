# Pilot SFT reference (from Kaggle dataset yuanmazax/delta-filing-smoke)

Downloaded 2026-06-11 to settle the SFT-checkpoint-identity question. These are the
exact files the batch-1 pilot DPO runs initialized from.

Verified byte-identical to repo `adapters/sft_pytorch/` and `kaggle_upload/adapters/sft_pytorch/`:
- adapter_model.safetensors : sha256 29a3794bf640336ac4622a6337437aba2a4b6c7da1c1868f255ec238e7585f02
- tokenizer_config.json     : sha256 9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d
- chat_template.jinja       : sha256 ce37839fe5d2adf271850c535f82c51deccb89e640667c3a5d6f2b30e65a6a11

=> pilot init == baseline_v2 init == matrix init (LoRA r=8, alpha=16, q/k/v/o_proj, base Qwen3.5-4B, peft 0.19.1).
