"""
Delta Filing — Local LLM Wrapper
==================================
Drop-in replacement for ChatOpenAI in app.py. Loads Qwen3.5-4B + GRPO_v2
adapter and serves LangGraph nodes via duck-typed .invoke() and .bind_tools().

Design:
  - LocalLLM.invoke(messages) -> AIMessage (pure text)
  - LocalLLM.bind_tools(tools) -> LocalLLMWithTools
  - LocalLLMWithTools.invoke(messages) -> AIMessage with tool_calls (JSON parsed)

Tool calling strategy:
  - When messages don't contain ToolMessage: model generates JSON tool call,
    we parse it and populate AIMessage.tool_calls (LangChain compatible).
  - When messages contain ToolMessage: model generates text summary
    (no tool_calls), graph flow exits the loop.

This matches the SFT training data which is single-turn:
  user query -> JSON tool call (no multi-turn tool reasoning trained).

Usage in app.py:
    from local_llm import LocalLLM
    llm = LocalLLM(adapter_path="./adapters/grpo_v2")
    llm_with_tools = llm.bind_tools(tools)
"""

import json
import re
import uuid
from typing import Any

import torch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
#  Config
# ============================================================

MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_ADAPTER = "./adapters/grpo_v2"  # Best adapter from ablation
MAX_NEW_TOKENS = 500
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)


# ============================================================
#  LocalLLM — base wrapper for pure-text invoke()
# ============================================================

class LocalLLM:
    """Drop-in replacement for ChatOpenAI. Loads model once, reuses everywhere."""

    def __init__(self, adapter_path: str = DEFAULT_ADAPTER, max_new_tokens: int = MAX_NEW_TOKENS):
        print(f"  [LocalLLM] Loading base model on {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map={"": DEVICE},
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        print(f"  [LocalLLM] Loaded adapter from {adapter_path}")

        self.max_new_tokens = max_new_tokens

    # ---- Internal generation ----

    def _generate(self, messages_dict: list[dict], max_tokens: int | None = None) -> str:
        """Generate raw text from a list of role/content dicts.

        Note: SFT-trained model sometimes fails to emit <|im_end|> at end of turn.
        We use stop_strings to detect the start of a new turn ("user", "assistant")
        and trim the output to the first turn boundary.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages_dict, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Tell HF generate to stop when these strings appear in output
                stop_strings=["\nuser\n", "\nassistant\n", "<|im_end|>", "<|im_start|>"],
                tokenizer=self.tokenizer,
            )
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Defensive: trim anything after a turn boundary if stop_strings didn't catch it
        for marker in ["\nuser\n", "\nassistant\n", "<|im_end|>", "<|im_start|>"]:
            idx = text.find(marker)
            if idx >= 0:
                text = text[:idx].strip()
        return text

    # ---- Convert LangChain messages to role/content dicts ----

    @staticmethod
    def _to_dict(messages: list) -> list[dict]:
        """Convert LangChain message objects (or dicts) to {role, content} dicts."""
        result = []
        for m in messages:
            if isinstance(m, dict):
                result.append(m)
            elif isinstance(m, SystemMessage):
                result.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                result.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                result.append({"role": "assistant", "content": m.content or ""})
            elif isinstance(m, ToolMessage):
                # Represent tool result as a user turn so model sees it
                # (SFT model wasn't trained on a separate "tool" role)
                result.append({
                    "role": "user",
                    "content": f"[Tool result]\n{m.content}",
                })
            else:
                # Generic fallback: use .content attribute if present
                content = getattr(m, "content", str(m))
                result.append({"role": "user", "content": content})
        return result

    # ---- Public LangChain-compatible API ----

    def invoke(self, messages: list, **kwargs) -> AIMessage:
        """Pure text generation. Returns AIMessage with .content."""
        msg_dicts = self._to_dict(messages)
        text = self._generate(msg_dicts)
        return AIMessage(content=text)

    def bind_tools(self, tools: list) -> "LocalLLMWithTools":
        """Return a tool-aware wrapper. Tools are encoded into system prompt."""
        return LocalLLMWithTools(self, tools)


# ============================================================
#  LocalLLMWithTools — tool calling wrapper
# ============================================================

# Matches training data system prompt format
TOOL_PROMPT_HEADER = (
    "You are Delta Filing, a financial analyst AI specializing in SEC filing analysis. "
    "You have access to tools for retrieving SEC filings, financial data, news, and insider trades. "
    "When you need data, call the appropriate tool. When you have data, analyze it thoroughly.\n\n"
    "Available tools:\n\n"
)

TOOL_PROMPT_FOOTER = (
    "\nTo use a tool, respond with a JSON object:\n"
    '{"tool": "tool_name", "arguments": {"param": "value"}}\n'
    "Respond with ONLY the JSON, nothing else."
)


# Section name → SEC Item ID mapping.
# The SFT model sometimes generates section names (e.g. "Risk Factors")
# when training data uses Item IDs ("1A"). Normalize before dispatch.
SECTION_NAME_TO_ID = {
    # Item 1 — Business
    "business": "1",
    "item 1": "1",
    "item1": "1",
    # Item 1A — Risk Factors
    "risk factors": "1A",
    "riskfactors": "1A",
    "risk_factors": "1A",
    "item 1a": "1A",
    "item1a": "1A",
    # Item 1B — Unresolved Staff Comments
    "unresolved staff comments": "1B",
    "item 1b": "1B",
    # Item 7 — MD&A
    "md&a": "7",
    "mda": "7",
    "management's discussion": "7",
    "management discussion": "7",
    "management's discussion and analysis": "7",
    "item 7": "7",
    "item7": "7",
    # Item 7A — Quantitative & Qualitative Disclosures About Market Risk
    "market risk": "7A",
    "quantitative risk": "7A",
    "quantitative and qualitative disclosures": "7A",
    "item 7a": "7A",
    "item7a": "7A",
    # Item 8 — Financial Statements
    "financial statements": "8",
    "financials": "8",
    "item 8": "8",
}


def normalize_section_id(value: str) -> str:
    """Map free-form section names to SEC Item IDs. Pass through if already an Item ID."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return stripped
    # Already a valid Item ID like "1A", "7", "1B" — pass through
    if len(stripped) <= 3 and stripped[0].isdigit():
        return stripped.upper()
    # Try mapping the lowercased form
    key = stripped.lower().strip()
    return SECTION_NAME_TO_ID.get(key, stripped)


class LocalLLMWithTools:
    """Tool-aware wrapper. Decides per-turn whether to emit tool call or text."""

    def __init__(self, llm: LocalLLM, tools: list):
        self.llm = llm
        self.tools = tools
        self.tools_by_name = {t.name: t for t in tools}
        self.tool_prompt = self._build_tool_prompt(tools)

    @staticmethod
    def _build_tool_prompt(tools: list) -> str:
        """Build tool-listing system prompt matching SFT training data format.

        Training data format example:
            1. search_filings(ticker: str, filing_type: str = "10-K", count: int = 5)
               List recent SEC filings for a company.

        We reconstruct this from LangChain tool's args_schema dict.
        """
        json_type_to_py = {
            "string": "str", "integer": "int", "number": "float",
            "boolean": "bool", "array": "list", "object": "dict",
        }

        lines = []
        for i, t in enumerate(tools, start=1):
            # Get args from schema (LangChain MCP tools have args_schema as dict)
            schema = getattr(t, "args_schema", None) or {}
            if isinstance(schema, dict):
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
            else:
                # Pydantic model fallback (older LangChain)
                props = getattr(schema, "__fields__", {})
                required = {k for k, v in props.items() if getattr(v, "required", True)}

            # Build "(param: type = default)" signature
            params = []
            for pname, pinfo in props.items():
                if isinstance(pinfo, dict):
                    ptype = json_type_to_py.get(pinfo.get("type", "string"), "str")
                    default = pinfo.get("default")
                else:
                    ptype = "str"
                    default = None

                if pname in required and default is None:
                    params.append(f"{pname}: {ptype}")
                else:
                    default_str = f'"{default}"' if isinstance(default, str) else str(default)
                    params.append(f"{pname}: {ptype} = {default_str}")

            sig = f"{t.name}({', '.join(params)})"

            # First line of description
            desc = (t.description or "").strip().split("\n")[0]
            lines.append(f"{i}. {sig}\n   {desc}")

        return TOOL_PROMPT_HEADER + "\n\n".join(lines) + TOOL_PROMPT_FOOTER

    def _has_tool_result(self, messages: list) -> bool:
        """Check if messages contain a ToolMessage (i.e. we already called a tool)."""
        return any(isinstance(m, ToolMessage) for m in messages)

    def _parse_tool_call(self, text: str) -> dict | None:
        """Extract first JSON tool call from generated text. Returns None if no valid call."""
        # Try to find JSON in the response
        # Strategy: look for first { ... } block and try to parse
        # SFT training tells model to output JSON only, so this is usually clean
        text = text.strip()

        # Try direct parse first
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "tool" in obj and "arguments" in obj:
                return obj
        except json.JSONDecodeError:
            pass

        # Try to extract first {...} substring
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict) and "tool" in obj and "arguments" in obj:
                    return obj
            except json.JSONDecodeError:
                pass

        return None

    def invoke(self, messages: list, **kwargs) -> AIMessage:
        """
        Decide between tool call and text response based on message history.
        - No ToolMessage in history → expect tool call (or direct answer if no tool needed)
        - Has ToolMessage → summarize tool result as text (no more tool calls)
        """
        # Convert to dicts, but inject our tool-aware system prompt FIRST
        # Strip the original SystemMessage from app.py (we replace it with our trained format)
        msg_dicts = []
        had_tool_result = self._has_tool_result(messages)

        # Build system prompt: training format + (optionally) instruction to summarize
        sys_content = self.tool_prompt
        if had_tool_result:
            sys_content = (
                "You are Delta Filing, a financial analyst AI. You have just received tool results. "
                "Based on the tool results below, provide a thorough analytical answer to the user's "
                "original question. Reference specific filing sections, cite numbers and dates, "
                "and provide analytical judgment. Do NOT call any more tools — just write the answer."
            )

        msg_dicts.append({"role": "system", "content": sys_content})

        # Append user/assistant/tool turns (skip any SystemMessage from caller — we provided our own)
        for m in messages:
            if isinstance(m, SystemMessage):
                continue  # already replaced
            if isinstance(m, HumanMessage):
                msg_dicts.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                # Skip empty assistant turns (these are tool-call placeholders)
                if m.content:
                    msg_dicts.append({"role": "assistant", "content": m.content})
            elif isinstance(m, ToolMessage):
                msg_dicts.append({
                    "role": "user",
                    "content": f"[Tool result]\n{m.content}",
                })

        # Generate
        text = self.llm._generate(msg_dicts)

        # If we already had tool result, return text directly (no tool_calls)
        if had_tool_result:
            return AIMessage(content=text)

        # Otherwise try parsing as tool call
        parsed = self._parse_tool_call(text)
        if parsed is not None:
            tool_name = parsed["tool"]
            tool_args = parsed["arguments"]
            # Normalize known args that the SFT model sometimes mis-formats
            if isinstance(tool_args, dict) and "section_id" in tool_args:
                tool_args["section_id"] = normalize_section_id(tool_args["section_id"])
            # Validate tool exists
            if tool_name in self.tools_by_name:
                # Return AIMessage with tool_calls in LangChain format
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                    }],
                )
            else:
                # Model hallucinated a tool name — fall through to text response
                pass

        # Model emitted text instead of tool call — return as plain answer
        return AIMessage(content=text)


# ============================================================
#  Standalone test
# ============================================================

if __name__ == "__main__":
    # Quick smoke test
    print("Loading LocalLLM (this takes ~15s)...")
    llm = LocalLLM()

    print("\n--- Test 1: pure text invoke ---")
    response = llm.invoke([
        SystemMessage(content="Classify the query. Reply with one word: WEATHER or FILING."),
        HumanMessage(content="What are Apple's risk factors?"),
    ])
    print(f"Response: {response.content!r}")

    print("\n--- Test 2: simulated bind_tools (no real tools) ---")
    # Fake a minimal tool object
    class FakeTool:
        def __init__(self, name, description):
            self.name = name
            self.description = description

    fake_tools = [
        FakeTool("get_filing_section", "Get text of a specific section from a filing"),
        FakeTool("stock_price", "Get recent stock price"),
    ]
    llm_w_tools = llm.bind_tools(fake_tools)
    response = llm_w_tools.invoke([
        HumanMessage(content="Get Apple's risk factors section"),
    ])
    print(f"Has tool_calls: {bool(response.tool_calls)}")
    print(f"Content: {response.content!r}")
    if response.tool_calls:
        print(f"Tool calls: {response.tool_calls}")

    print("\n--- Test 3: simulated tool result handling ---")
    response = llm_w_tools.invoke([
        HumanMessage(content="Get Apple's risk factors section"),
        AIMessage(content="", tool_calls=[{"name": "get_filing_section", "args": {"ticker": "AAPL", "section_id": "1A"}, "id": "x"}]),
        ToolMessage(content="Apple's Item 1A discusses supply chain, foreign exchange, and litigation risks...", tool_call_id="x"),
    ])
    print(f"Has tool_calls: {bool(response.tool_calls)}")
    print(f"Content preview: {response.content[:200]!r}")
