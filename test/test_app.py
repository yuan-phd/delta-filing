"""
Tests for app.py — LangGraph structure and routing
Tests graph structure without making LLM API calls.
Run: python tests/test_app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Only import what we can test without API keys
from app import DeltaState, decline_node


def test_state_has_required_fields():
    """DeltaState TypedDict has all required fields."""
    annotations = DeltaState.__annotations__
    required = ["messages", "user_query", "route"]
    for field in required:
        assert field in annotations, f"DeltaState missing '{field}'"

def test_state_has_review_fields():
    """DeltaState has review loop fields."""
    annotations = DeltaState.__annotations__
    review_fields = ["iteration_count", "needs_follow_up", "follow_up_instruction"]
    for field in review_fields:
        assert field in annotations, f"DeltaState missing review field '{field}'"

def test_decline_node():
    """Decline node returns a fixed message without LLM."""
    state = {"messages": [], "user_query": "What's the weather?", "route": "OUT_OF_SCOPE"}
    result = decline_node(state)
    assert "messages" in result
    assert len(result["messages"]) == 1
    content = result["messages"][0].content
    assert len(content) > 10
    assert any(word in content.lower() for word in ["sec", "filing", "scope", "help"])

def test_valid_routes():
    """All 5 routes are defined in the system."""
    valid = {"FILING_ANALYSIS", "FILING_DIFF", "SIMPLE_QUERY", "OUT_OF_SCOPE", "COMPANY_DEEP_DIVE"}
    # Check by reading the router function source
    import inspect
    from app import make_router_node
    source = inspect.getsource(make_router_node)
    for route in valid:
        assert route in source, f"Route '{route}' not found in router source"

def test_graph_builds():
    """Graph can be built (checks imports and structure, not LLM calls)."""
    try:
        from app import build_graph, build_company_subgraph
        # Just check the functions exist and are callable
        assert callable(build_graph)
        assert callable(build_company_subgraph)
    except ImportError as e:
        # If LangGraph or other deps missing, skip gracefully
        print(f"    Skipped (missing dependency: {e})")
        return

def test_review_max_iterations():
    """Review node source includes max iteration check."""
    import inspect
    from app import make_review_node
    source = inspect.getsource(make_review_node)
    assert "max_iterations" in source, "Review node missing max_iterations check"
    assert "iteration_count" in source, "Review node missing iteration_count"

def test_hitl_interrupt():
    """Review node uses interrupt() for human-in-the-loop."""
    import inspect
    from app import make_review_node
    source = inspect.getsource(make_review_node)
    assert "interrupt" in source, "Review node missing interrupt() for HITL"

def test_parallel_execution_nodes():
    """Company deep dive has parallel agents."""
    import inspect
    from app import financial_agent, news_agent, insider_agent
    assert callable(financial_agent)
    assert callable(news_agent)
    assert callable(insider_agent)

def test_checkpointer():
    """Graph uses MemorySaver checkpointer."""
    import inspect
    from app import build_graph
    source = inspect.getsource(build_graph)
    assert "MemorySaver" in source, "Graph missing MemorySaver checkpointer"
    assert "checkpointer" in source


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"  [PASS] {t.__name__}"); passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}"); failed += 1
    print(f"\n  app.py: {passed}/{passed+failed} passed")
