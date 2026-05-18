"""
Tests for training data quality
Run: python tests/test_data_quality.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_data")


def load_jsonl(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Source data tests ---

def test_sft_source_exists():
    """sft_data_combined.jsonl exists and has expected count."""
    data = load_jsonl(os.path.join(DATA_DIR, "sft_data_combined.jsonl"))
    assert data is not None, "sft_data_combined.jsonl not found"
    assert len(data) == 351, f"Expected 351 examples, got {len(data)}"

def test_sft_source_format():
    """Each SFT source example has required fields."""
    data = load_jsonl(os.path.join(DATA_DIR, "sft_data_combined.jsonl"))
    if data is None: return
    for i, ex in enumerate(data):
        assert "question" in ex, f"Example {i} missing 'question'"
        assert "answer" in ex, f"Example {i} missing 'answer'"
        assert "category" in ex, f"Example {i} missing 'category'"
        assert len(ex["question"]) > 10, f"Example {i} question too short"
        assert len(ex["answer"]) > 20, f"Example {i} answer too short"

def test_sft_source_categories():
    """SFT source has expected categories."""
    data = load_jsonl(os.path.join(DATA_DIR, "sft_data_combined.jsonl"))
    if data is None: return
    cats = set(ex["category"] for ex in data)
    expected = {"section_summary", "red_flag", "change_detection"}
    assert expected.issubset(cats), f"Missing categories: {expected - cats}"

def test_dpo_source_exists():
    """dpo_data_combined.jsonl exists."""
    data = load_jsonl(os.path.join(DATA_DIR, "dpo_data_combined.jsonl"))
    assert data is not None, "dpo_data_combined.jsonl not found"
    assert len(data) > 300, f"Expected 300+ DPO pairs, got {len(data)}"

def test_dpo_source_format():
    """Each DPO example has question, chosen, rejected."""
    data = load_jsonl(os.path.join(DATA_DIR, "dpo_data_combined.jsonl"))
    if data is None: return
    for i, ex in enumerate(data):
        assert "question" in ex, f"DPO {i} missing 'question'"
        assert "chosen" in ex, f"DPO {i} missing 'chosen'"
        assert "rejected" in ex, f"DPO {i} missing 'rejected'"
        assert len(ex["chosen"]) > len(ex["rejected"]) * 0.3, \
            f"DPO {i}: chosen suspiciously shorter than rejected"


# --- Merged data tests ---

def test_merged_train_exists():
    """Merged train.jsonl exists in sft_pytorch or sft."""
    for subdir in ["sft_pytorch", "sft"]:
        path = os.path.join(DATA_DIR, subdir, "train.jsonl")
        if os.path.exists(path):
            data = load_jsonl(path)
            assert len(data) > 800, f"Expected 800+ train, got {len(data)}"
            return
    assert False, "No train.jsonl found in sft_pytorch/ or sft/"

def test_merged_format():
    """Merged data has messages format."""
    for subdir in ["sft_pytorch", "sft"]:
        path = os.path.join(DATA_DIR, subdir, "train.jsonl")
        if os.path.exists(path):
            data = load_jsonl(path)
            for i, ex in enumerate(data[:10]):
                assert "messages" in ex, f"Example {i} missing 'messages'"
                msgs = ex["messages"]
                assert len(msgs) >= 2, f"Example {i}: need at least user+assistant"
                roles = [m["role"] for m in msgs]
                assert "user" in roles, f"Example {i}: no user message"
                assert "assistant" in roles, f"Example {i}: no assistant message"
            return

def test_no_thinking_tags():
    """No <think> tags in any training data."""
    for subdir in ["sft_pytorch", "sft"]:
        path = os.path.join(DATA_DIR, subdir, "train.jsonl")
        if os.path.exists(path):
            data = load_jsonl(path)
            for i, ex in enumerate(data):
                text = json.dumps(ex)
                assert "<think>" not in text, f"Example {i} contains <think>"
                assert "</think>" not in text, f"Example {i} contains </think>"
            return

def test_no_empty_content():
    """No messages with empty content."""
    for subdir in ["sft_pytorch", "sft"]:
        path = os.path.join(DATA_DIR, subdir, "train.jsonl")
        if os.path.exists(path):
            data = load_jsonl(path)
            for i, ex in enumerate(data):
                for j, msg in enumerate(ex["messages"]):
                    assert msg.get("content", "").strip(), \
                        f"Example {i}, message {j} has empty content"
            return

def test_no_duplicate_questions():
    """No duplicate user questions in training data."""
    for subdir in ["sft_pytorch", "sft"]:
        path = os.path.join(DATA_DIR, subdir, "train.jsonl")
        if os.path.exists(path):
            data = load_jsonl(path)
            questions = []
            for ex in data:
                for msg in ex["messages"]:
                    if msg["role"] == "user":
                        questions.append(msg["content"][:100])
            duplicates = len(questions) - len(set(questions))
            dup_rate = duplicates / len(questions) if questions else 0
            assert dup_rate < 0.05, f"Duplicate rate {dup_rate:.1%} exceeds 5%"
            return

def test_valid_test_split_no_overlap():
    """Train and valid/test sets don't overlap."""
    for subdir in ["sft_pytorch", "sft"]:
        train_path = os.path.join(DATA_DIR, subdir, "train.jsonl")
        valid_path = os.path.join(DATA_DIR, subdir, "valid.jsonl")
        if os.path.exists(train_path) and os.path.exists(valid_path):
            train = load_jsonl(train_path)
            valid = load_jsonl(valid_path)
            train_qs = set(json.dumps(ex["messages"][1]["content"][:100]) for ex in train if len(ex["messages"]) > 1)
            valid_qs = set(json.dumps(ex["messages"][1]["content"][:100]) for ex in valid if len(ex["messages"]) > 1)
            overlap = train_qs & valid_qs
            assert len(overlap) == 0, f"Train/valid overlap: {len(overlap)} examples"
            return


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"  [PASS] {t.__name__}"); passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}"); failed += 1
    print(f"\n  data quality: {passed}/{passed+failed} passed")
