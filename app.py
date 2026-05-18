"""
Delta Filing — Main Application (v2: Router + Custom State)
=============================================================
LangGraph agent with routing: different query types take different paths.

v1: Two nodes (agent ↔ tools), LLM decides everything
v2: Router classifies intent, different paths for different query types
    - FILING_ANALYSIS: multi-step filing analysis with tool calls
    - FILING_DIFF: change detection between filings
    - SIMPLE_QUERY: quick data lookup
    - OUT_OF_SCOPE: polite decline

Usage:
    python app.py
"""

import os
import asyncio
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from local_llm import LocalLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()


# ============================================================
#  Custom State — this is what flows between all nodes
# ============================================================

class DeltaState(TypedDict):
    # User input
    user_query: str
    route: str  # set by router: FILING_ANALYSIS, FILING_DIFF, SIMPLE_QUERY, OUT_OF_SCOPE, COMPANY_DEEP_DIVE

    # Conversation (LLM needs this for context)
    # Annotated with add_messages: LangGraph automatically APPENDS new messages
    # instead of replacing the whole list. This is critical for tool call chains.
    messages: Annotated[list, add_messages]

    # Analysis results (filled by different nodes)
    company: str  # extracted ticker, e.g. "AAPL"
    analysis_type: str  # what kind of analysis was requested

    # Loop control — for the analyst path
    iteration_count: int      # how many analysis rounds we've done
    needs_follow_up: bool     # does the review node think we need more depth?
    follow_up_instruction: str  # what to dig deeper on

    # Parallel execution results — each parallel agent writes to its own field
    financial_data: str    # filled by financial_agent
    news_data: str         # filled by news_agent
    insider_data: str      # filled by insider_agent


# ============================================================
#  Prompts — different roles for different nodes
# ============================================================

ROUTER_PROMPT = """You are a query classifier for a SEC filing analysis system.

Classify the user's query into exactly ONE category:

- FILING_ANALYSIS: user wants to read or analyze a specific section of a filing
  (e.g., "What are Apple's risk factors?", "Summarize Tesla's MD&A section")
  
- FILING_DIFF: user wants to compare filings across time periods, detect changes
  (e.g., "How did risk factors change?", "What's different from last year?")

- COMPANY_DEEP_DIVE: user wants a comprehensive overview of a company combining
  financial data, news, and insider activity
  (e.g., "Full analysis of Apple", "What's going on with Tesla?", "Deep dive MSFT")
  
- SIMPLE_QUERY: user wants a quick data lookup, stock price, or filing list
  (e.g., "List Apple's recent filings", "What filings does MSFT have?")
  
- OUT_OF_SCOPE: not related to SEC filings or financial analysis
  (e.g., "What's the weather?", "Write me a poem")

Reply with ONLY the category name, nothing else."""

ANALYST_PROMPT = """You are Delta Filing, a financial analyst AI specializing in SEC filing analysis.

You have tools to search filings, read specific sections, and compare sections across years.

Rules:
- Always reference specific filing sections (Item 1A, Item 7, etc.)
- Cite specific data points, dates, and numbers from the filings
- Flag concerning changes or risks you identify
- Note caveats and uncertainties
- Do NOT give investment advice or predict stock prices
- Be concise but thorough"""

DIFF_PROMPT = """You are Delta Filing, specializing in detecting changes between SEC filings.

You have a tool called diff_filing_sections that compares the same section across two 
consecutive filings and returns what was added, removed, and modified.

When presenting changes:
- Lead with the most significant changes
- Explain WHY a change might matter (e.g., new risk disclosure suggests emerging concern)
- Note the severity: minor wording tweaks vs substantive new content
- Compare the before/after text for modified paragraphs
- Do NOT give investment advice"""

DECLINE_MESSAGE = (
    "I'm Delta Filing, focused on SEC filing analysis. I can help you with:\n"
    "- Reading and analyzing 10-K/10-Q filing sections\n"
    "- Detecting changes between filings across years\n"
    "- Listing a company's recent SEC filings\n\n"
    "What would you like to know about a company's filings?"
)

REVIEW_PROMPT = """You are a quality reviewer for SEC filing analysis.
Review the analysis below and determine if it is thorough enough.

A good analysis MUST:
1. Reference specific filing sections (Item 1A, Item 7, etc.)
2. Cite specific numbers, dates, or quotes from the filing
3. Provide analytical judgment, not just summary
4. Address the user's original question directly

If the analysis is missing any of these, respond with:
NEEDS_FOLLOW_UP: [what specific additional analysis is needed]

If the analysis is thorough enough, respond with:
COMPLETE

Respond with ONLY one of these two formats."""

SYNTHESIS_PROMPT = """You are a financial analyst synthesizing a company deep dive report.

You have received data from three parallel research streams:
1. Financial data (stock price, PE ratio, market cap, margins)
2. Recent news (headlines and summaries from the past week)
3. Insider trading activity (executive buys and sells)

Combine all three into a concise, insightful overview. Structure your response as:
- Company snapshot (key metrics)
- Recent developments (news highlights)
- Insider activity (notable trades and what they might signal)
- Overall assessment (what the combined picture suggests)

Be specific with numbers. Note any concerning patterns."""


# ============================================================
#  Node functions
# ============================================================

def make_router_node(llm):
    """Create the router node — classifies user intent."""
    def router_node(state: DeltaState) -> dict:
        response = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=state["user_query"]),
        ])
        route = response.content.strip().upper()

        # Default to SIMPLE_QUERY if classification is unclear
        valid_routes = {"FILING_ANALYSIS", "FILING_DIFF", "SIMPLE_QUERY", "OUT_OF_SCOPE", "COMPANY_DEEP_DIVE"}
        if route not in valid_routes:
            route = "SIMPLE_QUERY"

        print(f"  [Router] → {route}")
        return {"route": route}

    return router_node


def make_analyst_node(llm_with_tools):
    """Create the filing analysis agent — reads and analyzes filing sections."""
    def analyst_node(state: DeltaState) -> dict:
        messages = [SystemMessage(content=ANALYST_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}  # add_messages will append automatically

    return analyst_node


def make_diff_node(llm_with_tools):
    """Create the diff agent — compares filings across years."""
    def diff_node(state: DeltaState) -> dict:
        messages = [SystemMessage(content=DIFF_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return diff_node


def make_simple_node(llm_with_tools):
    """Create the simple query agent — quick lookups."""
    def simple_node(state: DeltaState) -> dict:
        messages = [
            SystemMessage(content="Answer briefly using available tools. Be concise."),
        ] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return simple_node


def decline_node(state: DeltaState) -> dict:
    """Handle out-of-scope queries — no LLM needed, just a fixed message."""
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}


# --- Parallel execution nodes (Company Deep Dive) ---

def make_company_start_node(llm):
    """Extract the company ticker from the user's query.

    This node runs before the parallel agents fan out.
    It uses LLM to figure out which company the user is asking about.
    """
    def company_start(state: DeltaState) -> dict:
        response = llm.invoke([
            SystemMessage(content="Extract the stock ticker from this query. "
                                  "Reply with ONLY the ticker symbol, nothing else. "
                                  "Example: AAPL, MSFT, TSLA"),
            HumanMessage(content=state["user_query"]),
        ])
        ticker = response.content.strip().upper()
        print(f"  [Company Start] Ticker: {ticker}")
        return {"company": ticker}

    return company_start


def financial_agent(state: DeltaState) -> dict:
    """Fetch financial metrics and stock price. Runs in PARALLEL with other agents.

    Calls market.py directly (not through MCP) for simplicity.
    Each parallel agent writes to its OWN State field to avoid conflicts.
    """
    import json
    from market import get_key_metrics, get_stock_price

    ticker = state.get("company", "")
    print(f"  [Financial Agent] Fetching data for {ticker}...")

    try:
        metrics = get_key_metrics(ticker)
        price = get_stock_price(ticker, "1mo")
        result = {"metrics": metrics, "price": price}
    except Exception as e:
        result = {"error": str(e)}

    return {"financial_data": json.dumps(result, indent=2)}


def news_agent(state: DeltaState) -> dict:
    """Fetch recent company news. Runs in PARALLEL with other agents."""
    import json
    from market import get_company_news

    ticker = state.get("company", "")
    print(f"  [News Agent] Fetching news for {ticker}...")

    try:
        news = get_company_news(ticker, days=7)
        result = news
    except Exception as e:
        result = [{"error": str(e)}]

    return {"news_data": json.dumps(result, indent=2)}


def insider_agent(state: DeltaState) -> dict:
    """Fetch insider trading activity. Runs in PARALLEL with other agents."""
    import json
    from market import get_insider_trades

    ticker = state.get("company", "")
    print(f"  [Insider Agent] Fetching insider trades for {ticker}...")

    try:
        trades = get_insider_trades(ticker)
        result = trades
    except Exception as e:
        result = [{"error": str(e)}]

    return {"insider_data": json.dumps(result, indent=2)}


def make_synthesis_node(llm):
    """Combine results from all parallel agents into a unified report.

    This node runs AFTER all three parallel agents complete.
    LangGraph automatically waits for all incoming edges before running this.
    """
    def synthesis(state: DeltaState) -> dict:
        print(f"  [Synthesis] Combining results...")

        combined_data = (
            f"FINANCIAL DATA:\n{state.get('financial_data', 'N/A')}\n\n"
            f"RECENT NEWS:\n{state.get('news_data', 'N/A')}\n\n"
            f"INSIDER TRADES:\n{state.get('insider_data', 'N/A')}"
        )

        response = llm.invoke([
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(content=f"Company: {state.get('company', 'Unknown')}\n\n"
                                 f"User question: {state['user_query']}\n\n"
                                 f"Data collected:\n{combined_data}"),
        ])

        return {"messages": [AIMessage(content=response.content)]}

    return synthesis


def make_review_node(llm):
    """Create the review node — checks if analysis is thorough enough.

    This node creates the LOOP in the analyst path:
    - If analysis is incomplete → PAUSES and asks the human for confirmation
    - Human says yes → sends analyst back for more depth
    - Human says no → ends
    - If analysis is good enough OR max iterations reached → ends

    The pause is implemented with interrupt() — this is Human-in-the-Loop (HITL).
    When interrupt() is called, the entire graph freezes. The State is saved
    by the checkpointer. When the human responds, the graph resumes from
    exactly where it stopped.
    """
    def review_node(state: DeltaState) -> dict:
        iteration = state.get("iteration_count", 0) + 1
        max_iterations = 3

        # If we've already looped enough, stop regardless
        if iteration >= max_iterations:
            print(f"  [Review] Max iterations ({max_iterations}) reached. Finishing.")
            return {
                "iteration_count": iteration,
                "needs_follow_up": False,
                "follow_up_instruction": "",
            }

        # Get the last AI message (the analyst's response)
        last_ai_content = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_call_id"):
                last_ai_content = msg.content
                break

        if not last_ai_content:
            return {
                "iteration_count": iteration,
                "needs_follow_up": False,
                "follow_up_instruction": "",
            }

        # Ask the review LLM to evaluate
        review_input = [
            SystemMessage(content=REVIEW_PROMPT),
            HumanMessage(content=f"Original question: {state['user_query']}\n\n"
                                 f"Analysis to review:\n{last_ai_content}"),
        ]
        response = llm.invoke(review_input)
        review_result = response.content.strip()

        if review_result.startswith("NEEDS_FOLLOW_UP"):
            instruction = review_result.replace("NEEDS_FOLLOW_UP:", "").strip()
            print(f"  [Review] Round {iteration}: Needs more depth.")

            # === HITL: pause the graph and ask the human ===
            # interrupt() freezes everything. The value we pass becomes
            # visible to the main loop, which shows it to the user.
            # When the user responds, interrupt() returns their response.
            user_response = interrupt({
                "message": (
                    f"\n  [Review] The analysis could be improved:\n"
                    f"  {instruction}\n\n"
                    f"  Options: 'yes' to deep dive, 'no' to keep current analysis,\n"
                    f"  or type a custom instruction."
                ),
            })

            # === Graph resumes here after human responds ===
            if user_response.lower() in ("no", "n", "skip"):
                print(f"  [HITL] User chose to skip. Finishing.")
                return {
                    "iteration_count": iteration,
                    "needs_follow_up": False,
                    "follow_up_instruction": "",
                }
            else:
                # "yes" → use review's suggestion
                # anything else → use user's custom instruction
                if user_response.lower() in ("yes", "y"):
                    final_instruction = instruction
                else:
                    final_instruction = user_response

                print(f"  [HITL] User approved. Sending back to analyst.")
                return {
                    "iteration_count": iteration,
                    "needs_follow_up": True,
                    "follow_up_instruction": final_instruction,
                    "messages": [HumanMessage(
                        content=f"Please enhance your analysis: {final_instruction}"
                    )],
                }
        else:
            print(f"  [Review] Round {iteration}: Analysis is complete.")
            return {
                "iteration_count": iteration,
                "needs_follow_up": False,
                "follow_up_instruction": "",
            }

    return review_node


# ============================================================
#  Subgraphs — independent workflow units
# ============================================================

def build_company_subgraph(llm):
    """Build the Company Deep Dive as an independent subgraph.

    A subgraph is a complete graph (nodes + edges) compiled into a single unit.
    The parent graph treats it as ONE node — it doesn't see the internal structure.

    Before (5 nodes registered in parent graph):
        parent graph knows about: company_start, financial_agent,
        news_agent, insider_agent, synthesis

    After (1 node registered in parent graph):
        parent graph only sees: company_deep_dive
        The 5 nodes are hidden inside the subgraph

    Benefits:
        - Parent graph is cleaner (fewer nodes to manage)
        - Subgraph can be tested independently
        - Subgraph can be reused in different parent graphs
        - Teams can develop subgraphs independently
    """
    subgraph = StateGraph(DeltaState)

    # Register nodes (same nodes as before, but inside the subgraph)
    subgraph.add_node("company_start", make_company_start_node(llm))
    subgraph.add_node("financial_agent", financial_agent)
    subgraph.add_node("news_agent", news_agent)
    subgraph.add_node("insider_agent", insider_agent)
    subgraph.add_node("synthesis", make_synthesis_node(llm))

    # Entry point of the subgraph
    subgraph.set_entry_point("company_start")

    # Parallel fan-out and fan-in (same as before)
    subgraph.add_edge("company_start", "financial_agent")
    subgraph.add_edge("company_start", "news_agent")
    subgraph.add_edge("company_start", "insider_agent")

    subgraph.add_edge("financial_agent", "synthesis")
    subgraph.add_edge("news_agent", "synthesis")
    subgraph.add_edge("insider_agent", "synthesis")

    subgraph.add_edge("synthesis", END)

    return subgraph.compile()


# ============================================================
#  Graph construction
# ============================================================

def build_graph(llm, llm_with_tools, tools):
    """Build the full LangGraph with router, multiple paths, review loop, and subgraph.

    Graph structure (parent graph):

                          ┌──────────┐
                          │  router  │
                          └────┬─────┘
                               │
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          ▼
    analyst   diff_agent   company     simple     decline
      │↕         │↕       deep_dive     │↕          │
    tools_a    tools_d    (SUBGRAPH)   tools_s       │
      │          │           │           │          │
      ▼          │           │           │          │
    review       │           │           │          │
      │          │           │           │          │
      ▼          ▼           ▼           ▼          ▼
                            END

    Inside the company_deep_dive subgraph:
        company_start → [financial | news | insider] → synthesis
    """

    graph = StateGraph(DeltaState)

    # --- Register nodes ---
    graph.add_node("router", make_router_node(llm))

    # Filing analysis path (with review loop + HITL)
    graph.add_node("analyst", make_analyst_node(llm_with_tools))
    graph.add_node("tools_analyst", ToolNode(tools))
    graph.add_node("review", make_review_node(llm))

    # Diff path
    graph.add_node("diff_agent", make_diff_node(llm_with_tools))
    graph.add_node("tools_diff", ToolNode(tools))

    # Company deep dive — entire parallel pipeline is ONE subgraph node
    # The parent graph doesn't see the 5 internal nodes,
    # only this single "company_deep_dive" node
    graph.add_node("company_deep_dive", build_company_subgraph(llm))

    # Simple path
    graph.add_node("simple", make_simple_node(llm_with_tools))
    graph.add_node("tools_simple", ToolNode(tools))

    # Decline path
    graph.add_node("decline", decline_node)

    # --- Entry point ---
    graph.set_entry_point("router")

    # --- Router dispatches to different paths ---
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "FILING_ANALYSIS": "analyst",
            "FILING_DIFF": "diff_agent",
            "COMPANY_DEEP_DIVE": "company_deep_dive",  # routes to subgraph
            "SIMPLE_QUERY": "simple",
            "OUT_OF_SCOPE": "decline",
        },
    )

    # --- Analyst path: analyst ↔ tools_analyst → review → (loop or end) ---
    graph.add_conditional_edges(
        "analyst",
        tools_condition,
        {"tools": "tools_analyst", END: "review"},
    )
    graph.add_edge("tools_analyst", "analyst")
    graph.add_conditional_edges(
        "review",
        lambda state: "analyst" if state.get("needs_follow_up", False) else END,
    )

    # --- Diff path: diff_agent ↔ tools_diff ---
    graph.add_conditional_edges(
        "diff_agent",
        tools_condition,
        {"tools": "tools_diff", END: END},
    )
    graph.add_edge("tools_diff", "diff_agent")

    # --- Company deep dive: subgraph handles everything internally ---
    graph.add_edge("company_deep_dive", END)

    # --- Simple path: simple ↔ tools_simple ---
    graph.add_conditional_edges(
        "simple",
        tools_condition,
        {"tools": "tools_simple", END: END},
    )
    graph.add_edge("tools_simple", "simple")

    # --- Decline goes straight to END ---
    graph.add_edge("decline", END)

    return graph.compile(checkpointer=MemorySaver())


# ============================================================
#  Main
# ============================================================

async def run_agent():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_edgar.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            print(f"Loaded {len(tools)} tools from MCP server")

            llm = LocalLLM(adapter_path="./adapters/grpo_v2")
            llm_with_tools = llm.bind_tools(tools)

            app = build_graph(llm, llm_with_tools, tools)

            print("\n" + "=" * 60)
            print("  Delta Filing — SEC Filing Intelligence")
            print("  Routes: FILING_ANALYSIS | FILING_DIFF | COMPANY_DEEP_DIVE")
            print("  Type 'quit' to exit")
            print("=" * 60)

            thread_counter = 0

            while True:
                query = input("\nYou: ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    break
                if not query:
                    continue

                # Each query gets a unique thread_id for checkpointing
                thread_counter += 1
                config = {"configurable": {"thread_id": f"session-{thread_counter}"}}

                # Build initial state
                initial_state = {
                    "user_query": query,
                    "route": "",
                    "messages": [HumanMessage(content=query)],
                    "company": "",
                    "analysis_type": "",
                    "iteration_count": 0,
                    "needs_follow_up": False,
                    "follow_up_instruction": "",
                    "financial_data": "",
                    "news_data": "",
                    "insider_data": "",
                }

                print("")  # newline after "You: "

                # --- Run with streaming ---
                # astream yields updates as each node completes,
                # so we can show real-time progress instead of waiting in silence.
                final_result = None

                async for event in app.astream(initial_state, config, stream_mode="updates"):
                    # event is a dict: {node_name: node_output}
                    for node_name, node_output in event.items():
                        # Show progress based on which node just completed
                        if node_name == "router":
                            pass  # router already prints its own [Router] message
                        elif node_name == "company_start":
                            pass  # already prints [Company Start]
                        elif node_name in ("financial_agent", "news_agent", "insider_agent"):
                            pass  # already print their own messages
                        elif node_name == "synthesis":
                            pass  # already prints [Synthesis]
                        elif node_name in ("tools_analyst", "tools_diff", "tools_simple"):
                            print(f"  [Tools] Data retrieved.")
                        elif node_name == "review":
                            pass  # review prints its own messages

                        # Capture the final state
                        if "messages" in node_output:
                            final_result = node_output

                # Check if the graph is paused (HITL interrupt)
                state = await app.aget_state(config)
                while state.next:
                    # Graph is paused — there's an interrupt waiting for human input
                    interrupt_value = state.tasks[0].interrupts[0].value
                    print(interrupt_value["message"])

                    user_input = input("\n  Your decision: ").strip()
                    if not user_input:
                        user_input = "yes"

                    # Resume with streaming too
                    async for event in app.astream(
                        Command(resume=user_input), config, stream_mode="updates"
                    ):
                        for node_name, node_output in event.items():
                            if node_name in ("tools_analyst", "tools_diff", "tools_simple"):
                                print(f"  [Tools] Data retrieved.")
                            if "messages" in node_output:
                                final_result = node_output

                    state = await app.aget_state(config)

                # Print the final response
                if final_result and "messages" in final_result:
                    msgs = final_result["messages"]
                    # Find the last message with content
                    for msg in reversed(msgs) if isinstance(msgs, list) else [msgs]:
                        if hasattr(msg, "content") and msg.content:
                            print(f"\n{msg.content}")
                            break
                else:
                    print("[No response generated]")


if __name__ == "__main__":
    asyncio.run(run_agent())
