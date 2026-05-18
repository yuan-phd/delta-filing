# Delta Filing

A SEC filing intelligence system that **replaces GPT-4o-mini with a locally fine-tuned 4B model**. Trains and compares five alignment methods (SFT, DPO, IPO, cDPO, GRPO) on real 10-K data, identifies a critical KL-divergence implementation bug in GRPO, and deploys the resulting adapter behind a LangGraph multi-agent orchestrator.

## What it does

Given a query like *"What are Apple's risk factors in their latest 10-K?"*, the system routes the query through a LangGraph agent, retrieves the relevant SEC filing section via MCP tools, and produces an analytical summary — entirely on-device. The same workflow supports year-over-year filing comparisons, red flag detection, change tracking, and multi-source company deep-dives (filings + market data + insider trades).

Every LLM call goes through a fine-tuned Qwen3.5-4B + LoRA adapter running on Apple Silicon MPS. No OpenAI dependency.

## Highlights

### Five alignment methods, six hand-written loss functions

| Adapter | Method | Final score (87-test) | Notes |
|---------|--------|----------------------|-------|
| SFT | Supervised cross-entropy | 73/87 (84%) | Baseline |
| **DPO** | `-logσ(β·margin)` | **65/87 (75%)** | **Unbounded margin → catastrophic Review collapse (1/9)** |
| IPO | `(margin - 1/(2β))²` | 70/87 (80%) | Identity loss — bounded margin |
| cDPO | `-(1-ε)·logσ(m) - ε·logσ(-m)` | 69/87 (79%) | Label smoothing |
| **GRPO v2** | RL with k3 KL estimator | **76/87 (87%)** | **Best overall — multi-objective reward** |

Six losses implemented from scratch: DPO, IPO, cDPO, GRPO, InfoNCE (for embedding training), SFT retention. The five-method comparison surfaces a striking failure mode in unbounded DPO (below) that the bounded variants and RL avoid.

### Alignment tax severity tracks the boundedness of the preference objective

A persistent narrative in alignment research holds that fine-tuning on a small parameter budget (LoRA) forces a zero-sum trade-off between alignment quality and retained base capabilities. This project tested that claim with a controlled comparison across three DPO-family methods that differ only in how they constrain the preference margin.

The three methods training on the same data, with the same setup, with the same LoRA rank, differ only in margin constraint:

- **DPO** (`-logσ(β·margin)`) — margin is unbounded; loss decreases monotonically as margin → ∞
- **IPO** (`(margin - 1/(2β))²`) — margin pulled toward fixed target `1/(2β) = 1.0`; loss penalizes both under- and over-shooting
- **cDPO** — DPO with label smoothing `ε=0.1`, partial floor on the gradient

The empirical result tracks the boundedness ordering. During DPO training, the implicit reward margin grew from 0.02 at step 10 to 8.76 at step 520 — a 440× increase. The resulting adapter collapsed catastrophically on rigid-format tasks: 1/9 on Review (binary `COMPLETE`/`NEEDS_FOLLOW_UP` classification), 3/6 on Refusal. IPO held the margin near 1.0 (its target) and lost 3 points on tool-calling. cDPO showed an intermediate failure mode.

GRPO_v2, training on the same data and same LoRA rank but using a hand-designed multi-objective reward function (tool-call validity, section reference, numerical specificity, analytical judgment, format compliance), holds 15/15 on tool calling *and* leads on synthesis (7/8 vs SFT's 5/8). It avoids the trade-off entirely.

The interpretation: alignment tax magnitude in DPO-family methods is determined by how strongly the preference objective constrains the implicit reward margin. Unbounded DPO can over-optimize until format-rigid capabilities collapse. RL with an explicit, multi-objective reward signal can avoid the trade-off — but only if every important dimension is in the reward function. The LoRA parameter budget is not the binding constraint here.

### Diagnosed and fixed a KL-divergence implementation bug in GRPO

The initial GRPO implementation used `kl = pi_log_prob - ref_log_prob`. This is the log-ratio, not the KL divergence — it goes negative whenever the policy assigns lower probability to a token than the reference (which happens routinely as the policy learns). With β·kl added to the loss, a negative kl term *rewards* divergence from SFT. The policy drifted, lost SFT-acquired capabilities, and the resulting adapter scored 66/87 — worse than SFT baseline.

Fix: replaced with the k3 estimator from Schulman (2020), `kl = exp(log_ratio) - 1 - log_ratio` where `log_ratio = ref_log_prob - pi_log_prob`. k3 is unbiased and always non-negative — when the policy and reference agree, k3 ≈ 0; when they diverge, k3 grows fast and penalizes correctly. Added `clamp(log_ratio, -10, 10)` to guard against numerical explosion in the `exp()` term during the first few training steps.

After the fix, training-time KL values stayed in [0, 0.0006] throughout (vs. drifting to -1.08 with the bug), and the resulting adapter scored 76/87 — a +10 test improvement, the single largest delta in the project. The fact that the same data, same hyperparameters, and same reward function produced wildly different results based on this one-line KL implementation underscores how easily ablation studies can mis-attribute outcomes to algorithm choice when the underlying implementation is subtly broken.

### A custom 87-test evaluation suite catches issues real benchmarks miss

Built from scratch. Ten categories covering routing, tool calling, JSON parsing, multi-step synthesis, refusal, robustness. Includes deliberately false-positive red-flag tests (healthy companies like JNJ/PG/V) to detect over-alarming behavior. When binary pass/fail couldn't distinguish models, switched to a continuous *alarm word count* metric (JNJ alarm counts: SFT=4, DPO=3, IPO=6, cDPO=9, GRPO=5) — surfacing that DPO produces more confident analyses but cDPO over-alarms 2× as much as SFT. The suite caught DPO's catastrophic Review collapse (1/9) that aggregate accuracy alone would have understated — Tool parsing 8/8 + Review 1/9 in the same model is the diagnostic.

### End-to-end production replacement

`local_llm.py` is a duck-typed drop-in for `ChatOpenAI` with `.invoke()` and `.bind_tools()`. It loads Qwen3.5-4B + GRPO_v2 adapter, parses JSON tool calls from the SFT-trained model, and normalizes common parameter mistakes (e.g. the model occasionally emits `section_id="Risk Factors"` instead of `"1A"` — the wrapper maps this back to the SEC Item ID). Swapping `ChatOpenAI(model="gpt-4o-mini")` to `LocalLLM(adapter_path="./adapters/grpo_v2")` is a two-line change to `app.py`.

## Architecture

```
User query
   │
   ▼
┌─────────┐
│ Router  │ ── FILING_ANALYSIS / FILING_DIFF / COMPANY_DEEP_DIVE / SIMPLE_QUERY / OUT_OF_SCOPE
└────┬────┘
     │
 ┌───┴────────┬─────────────┬──────────┬─────────┐
 ▼            ▼             ▼          ▼         ▼
analyst   diff_agent   company      simple   decline
  ↕↑         ↕↑       deep_dive       ↕↑
tools ↔   tools ↔    (subgraph)    tools ↔
  ↓
review (HITL loop)
  ↓
END
```

All nodes use the same local LLM instance. Three MCP servers expose 12 tools:
- `mcp_edgar.py` — SEC filing retrieval, section parsing, year-over-year diff
- `mcp_market.py` — stock prices, company metrics, insider trades, analyst ratings
- `mcp_rag.py` — semantic search over indexed filing sections (FAISS + custom embedding)

The RAG component uses a fine-tuned MiniLM-L6-v2 (22M params) trained with InfoNCE on 2000 positive pairs from real SEC filing analyses, indexing 351 filing sections across 37 companies.

## Quick start

```bash
pip install -r requirements.txt

# Download Qwen3.5-4B base (HF will fetch on first run)
# Adapters are checked in under ./adapters/

# Build the RAG index (one-time, ~1 minute)
python generate_embedding_data.py
python train_embedding.py --train      # ~5 min on CPU/MPS
python build_index_from_sft.py         # ~1 min

# Run the agent
python app.py
```

Inference runs at ~5.6 tokens/sec on M1 Max MPS. Typical query latency: 30 seconds for a routing decision, 1–3 minutes for a full analysis.

## Project structure

```
delta-filing/
├── app.py                       # Main LangGraph orchestrator
├── local_llm.py                 # LocalLLM wrapper (drop-in for ChatOpenAI)
├── edgar.py / diff.py / market.py
├── mcp_edgar.py / mcp_market.py / mcp_rag.py   # MCP servers
├── adapters/
│   ├── sft_pytorch/             # SFT adapter
│   ├── dpo_ablation/            # IPO + cDPO from a single ablation script
│   └── grpo_v2/                 # GRPO adapter (deployed in production)
├── training_data/
│   ├── sft_data_combined.jsonl  # 351 real-filing analyses (37 companies, 4 sections)
│   ├── sft_supplementary_fixed.jsonl  # tool-calling examples
│   ├── dpo_pytorch.jsonl        # 647 preference pairs
│   └── embedding_pairs.jsonl    # 2000 positive pairs for InfoNCE
├── train_sft_kaggle.py          # SFT training (Kaggle T4)
├── train_dpo.py                 # DPO-family ablation (IPO + cDPO + DPO)
├── train_grpo_kaggle.py         # GRPO training (v2 with k3 KL fix)
├── train_embedding.py           # Custom embedding model (InfoNCE)
├── faiss_index/                 # FAISS vector store (351 sections)
├── models/financial_embeddings/ # Fine-tuned MiniLM weights
├── test_generator/              # Test suite + extension tooling
└── test/                        # Local 43-test smoke suite
```

## Detailed results

### 87-test main suite (across 5 adapters)

| Category          | SFT    | DPO    | IPO    | cDPO   | GRPO_v2 |
|-------------------|--------|--------|--------|--------|---------|
| Router            | 10/11  | 8/11   | 10/11  | 9/11   | 10/11   |
| Tool calling      | **15/15** | 14/15 | 12/15 | 12/15  | **15/15** |
| Tool parsing      | 6/8    | **8/8** | **8/8** | 6/8  | 6/8     |
| Review            | 8/9    | **1/9 ⚠️** | 7/9 | 6/9    | 8/9     |
| Synthesis         | 5/8    | 6/8    | 5/8    | 6/8    | **7/8** |
| Section summary   | 7/8    | 7/8    | 7/8    | **8/8** | 7/8     |
| Red flag          | 7/10   | **8/10** | 7/10 | 7/10   | **8/10** |
| Change detection  | 6/8    | 7/8    | 6/8    | **8/8** | 6/8     |
| Refusal           | **6/6** | 3/6   | 5/6    | 5/6    | **6/6** |
| Robustness        | 3/4    | 3/4    | 3/4    | 2/4    | 3/4     |
| **Total**         | **73** | **65** | **70** | **69** | **76**  |

DPO's Review collapse (1/9 vs others' 6-8/9) is the signature of unbounded preference optimization. The Review task asks the model to emit one of two tokens (`COMPLETE` or `NEEDS_FOLLOW_UP`); DPO retains the ability to generate well-formed JSON tool calls and free-form analysis, but loses the ability to follow this rigid two-class format. Tool parsing 8/8 + tool calling 14/15 confirms it isn't a capability collapse — it's a strategy shift induced by the unbounded margin objective.

### False-positive red flag detection (alarm word count, lower = better)

A continuous metric — counts occurrences of strong alarm words (concerning, severe, critical, misleading, etc.) in analyses of *healthy* companies. The model should not over-alarm on JNJ, PG, V.

| Company | SFT | DPO | IPO | cDPO | GRPO_v2 |
|---------|-----|-----|-----|------|---------|
| JNJ     | 4   | **3 ★** | 6 | 9    | 5       |
| PG      | 4   | 4   | 5   | 5    | **3 ★** |
| V       | 4   | 6   | 4   | 8    | 7       |
| **Avg** | **4.0** | 4.3 | 5.0 | 7.3 | 5.0    |

DPO has the lowest JNJ alarm count (3) — yet collapses on Review (1/9). Both signals are consistent with unbounded preference optimization: the model becomes confident in producing analytical paragraphs (lower alarm noise, better synthesis) but loses the format-following ability needed for binary classification. cDPO over-alarms 2× as much as SFT — label smoothing prevents extreme margins but adds noise to the analysis style.

### RAG retrieval quality

| Query | Top result |
|-------|-----------|
| "supply chain risks and manufacturing" | AVGO Item 1A (Risk Factors), sim=0.67 |
| "cybersecurity threats and data breaches" | DIS Item 1A (Risk Factors), sim=0.56 |
| "revenue growth and cloud services" | MSFT Item 7 (MD&A), sim=0.75 |
| "foreign currency exchange rate impact" | GOOGL Item 7A (Quantitative Risk), sim=0.64 |

All four queries retrieve the correct SEC Item type; same-topic / cross-topic margin = 0.093 (PASS, threshold 0.05).

## Known limitations

- **Inference speed**: 5.6 tok/s on M1 Max MPS. A full deep-dive query takes 1–3 minutes. Deployment on a GPU server (or via llama.cpp + GGUF) would improve this substantially. The slow speed is a hardware/deployment issue, not a model issue — the trained adapter is identical to what would run on a server.
- **`tool` role not in training data**: The SFT model wasn't trained with `tool`-role messages. The wrapper passes tool results back as a `user` turn prefixed with `[Tool result]`. Works in practice but not the cleanest design.
- **Section name vs ID**: The model occasionally generates `section_id="Risk Factors"` instead of `"1A"`. Handled by a normalization map in the wrapper (14 common name→ID mappings).
- **Three false-positive red flag tests fail across all adapters**: All four adapters over-alarm on healthy companies (JNJ/PG/V), and the binary pass/fail at threshold ≤3 alarm words can't distinguish them. The continuous alarm-count metric (above) is the meaningful signal.
- **`tool_calling` category at ceiling**: Both SFT and GRPO_v2 score 15/15 on tool calling. Additional ambiguous-intent test cases would be needed to differentiate further.
- **SFT-retention loss computes on full sequence**: Not just on assistant tokens. A minor inefficiency, not a correctness bug, but it dilutes the retention signal and likely contributed to the IPO/cDPO alignment tax magnitude.

## Acknowledgments

- Base model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- Embedding base: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- KL k3 estimator: [Schulman, "Approximating KL Divergence" (2020)](http://joschu.net/blog/kl-approx.html)
- Training infrastructure: Kaggle T4 (free tier)
- Data source: SEC EDGAR
