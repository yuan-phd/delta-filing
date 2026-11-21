# Agent system: details

Details of the SEC filing agent system. The overview, architecture diagram, router and MCP tables, and quick start are in the [README](../README.md#agent-system).

## Agent and tool loops

Three nodes (`analyst`, `diff_agent`, `simple`) form agent and tool loops. The agent emits a JSON tool call, a `ToolNode` dispatches it to `mcp_edgar` over MCP, the result returns as a `ToolMessage`, and the agent decides whether to call another tool or write the final answer. LangGraph's `tools_condition` handles the routing. With the local model, `local_llm.py` writes a text answer once a `ToolMessage` is in the history, so in practice the agent makes no further tool calls after the first result.

## Human-in-the-loop review

After the analyst finishes, a **review node** checks the output with the same local model acting as judge. If the analysis is judged shallow, the graph calls `interrupt()`:

- LangGraph **freezes the state** with the `MemorySaver` checkpointer
- The reviewer's suggestion is shown to the user at the CLI prompt
- The user chooses `yes` (accept the reviewer's instruction), `no` (skip and finish), or types a **custom improvement instruction**
- When the user responds, `interrupt()` resumes the graph exactly where it stopped
- The user's instruction is added to the analyst's message history as a new `HumanMessage`

A `max_iterations=3` guard prevents infinite loops whatever the reviewer or user prefers.

## Backend data

- **SEC EDGAR** (`edgar.py`): direct HTTP to SEC's public API with a `User-Agent` header (set in `edgar.py`; replace it with your own contact). Parses 10-K and 10-Q HTML to extract Item sections (1, 1A, 7, 7A, 8). All raw HTML is cached locally to respect SEC rate limits and speed up repeat queries.
- **Cross-year diff** (`diff.py`): paragraph-level matching between the same section of two filings. Identifies added, removed and modified paragraphs. It differs from a generic text diff because it is structured around SEC section semantics.
- **Market data** (`market.py`): yfinance for prices and fundamentals, Finnhub for news, insider trades and analyst ratings.
- **RAG** (`mcp_rag.py`): 351 SEC filing sections indexed in FAISS (384-d, `IndexFlatIP`), retrieved with a custom MiniLM-L6-v2 embedding model fine-tuned with InfoNCE on 2000 same-topic-different-company pairs.

## Local LLM wrapper

All LLM-calling nodes (router, analyst, diff, the deep-dive subgraph's ticker-extraction and synthesis nodes, simple, review) share a **single `LocalLLM` instance**. It is loaded once at startup: Qwen3.5-4B base + the GRPO LoRA adapter (`adapters/grpo_v2`), in float16 on MPS, CUDA or CPU.

`local_llm.py` is duck-typed to match `langchain_openai.ChatOpenAI`: it implements `.invoke()` and `.bind_tools()` with the same return types. Swapping the cloud LLM for the local one is a two-line change in `app.py`:

```python
from local_llm import LocalLLM
llm = LocalLLM(adapter_path="./adapters/grpo_v2")
```

The wrapper handles three subtleties of inference with a fine-tuned model:
1. The SFT model sometimes fails to emit `<|im_end|>`. The wrapper passes `stop_strings=["\nuser\n", "\nassistant\n", "<|im_end|>", "<|im_start|>"]` to `generate()` with a defensive `text.find()` fallback.
2. The model occasionally emits `section_id="Risk Factors"` instead of the SEC Item ID `"1A"`. A `SECTION_NAME_TO_ID` map normalizes it before tool dispatch.
3. The SFT training data has no `tool` role. The wrapper converts `ToolMessage` to a `user` turn prefixed `[Tool result]`.

## Project structure

```
delta-filing/
├── app.py                       # LangGraph orchestrator (router, agents, review, subgraph)
├── local_llm.py                 # LocalLLM wrapper — drop-in for ChatOpenAI
├── edgar.py / diff.py / market.py     # Backend data layer
├── mcp_edgar.py / mcp_market.py / mcp_rag.py    # Three MCP servers
├── adapters/
│   ├── sft_pytorch/             # SFT adapter (73/87)
│   ├── dpo_pytorch/             # DPO adapter (65/87)
│   ├── dpo_ablation/            # IPO + cDPO from controlled ablation
│   └── grpo_v2/                 # GRPO adapter (76/87, used by the agent) ★
├── training_data/               # SFT + DPO preference pairs + embedding pairs
├── train_sft_kaggle.py          # SFT (Kaggle T4×2, ~7h)
├── train_dpo.py                 # DPO standalone
├── train_grpo_kaggle.py         # GRPO with k3 KL estimator
├── train_embedding.py           # MiniLM-L6-v2 InfoNCE
├── faiss_index/                 # 351-vector RAG index
├── models/financial_embeddings/ # Fine-tuned MiniLM weights
├── notebook/kaggle/             # 9 Kaggle training/eval notebooks
├── notebook/kaggle_output/      # Training logs (incl. GRPO v1/v2 KL evidence)
├── test_generator/              # Test-suite generator + runner (see docs/ABLATION.md)
└── test/                        # Local pytest smoke suite
```

The paper's files are mapped in [PAPER_MAP.md](PAPER_MAP.md).
