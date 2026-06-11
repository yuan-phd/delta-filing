"""Two-stage deterministic scorer (claude_code_spec.md §6) — the core measurement.

Pure functions, no LLM, no randomness. score() returns a verdict in
{PASS, FORMAT_BREAK, REASONING_ERROR} plus the per-item fields the eval harness
persists alongside every generation.

Stage 1 = FORMAT GATE (is the output in the required shape?).
Stage 2 = CONTENT (given valid format, is it correct?).

SYMMETRIC GATE (code-review fix): both gates require the output, after stripping
leading whitespace (and an optional opening code fence for JSON), to BEGIN with
the required structured token; trailing content after it is allowed. This avoids
asymmetric strictness (prose-tolerant JSON vs prose-intolerant binary) that could
manufacture the dissociation.
  binary : must start with a DELIMITED label  -> "COMPLETE: reason" passes,
           "COMPLETED" fails (wrong token), "Based on... COMPLETE" is FORMAT_BREAK.
  json   : must start with a valid JSON object (or fenced JSON) -> "{...} note"
           passes, "Sure: {...}" is FORMAT_BREAK.
A `leading_prose` flag and a LENIENT gate (token allowed anywhere) are recorded as
a sensitivity metric; the strict symmetric gate is the PRIMARY verdict driver.

normalization_applied is True for ANY normalization needed to accept the output:
surrounding whitespace, trailing punctuation, case folding, or an opening code
fence (JSON).
"""
import json
import re

VERDICT_PASS = "PASS"
VERDICT_FORMAT_BREAK = "FORMAT_BREAK"
VERDICT_REASONING_ERROR = "REASONING_ERROR"

BINARY_LABELS = {
    "track1": {"COMPLETE", "NEEDS_FOLLOW_UP"},
    "track2": {"YES", "NO"},   # boolq
}
# router: closed-set 5-class enum (different format family from JSON; degeneracy
# floored at 1/5 = 0.20, so collapse-to-constant is visible on accuracy).
ROUTER_LABELS = {"FILING_ANALYSIS", "FILING_DIFF", "COMPANY_DEEP_DIVE",
                 "SIMPLE_QUERY", "OUT_OF_SCOPE"}
# extraction: flat JSON schema with a date-format constraint (fragile multi-key).
EXTRACTION_KEYS = ("ticker", "section", "filing_date")

SCORER_RULES = {
    "gate": "SYMMETRIC begins-with: output (after lstrip + optional opening code "
            "fence for JSON) must START with the required structured token; "
            "trailing content allowed.",
    "binary_strict": "starts with a delimited label (negative lookahead on "
                     "[A-Za-z0-9_], case-insensitive) so COMPLETED != COMPLETE; "
                     "'COMPLETE: reason' passes, 'Based on... COMPLETE' breaks.",
    "binary_lenient": "(sensitivity) label present as a delimited token anywhere.",
    "json_strict": "starts with a balanced {...} (after optional ```json fence) "
                   "that parses and has tool/name + arguments; '{...} note' passes, "
                   "'Sure: {...}' breaks.",
    "json_lenient": "(sensitivity) any balanced {...} anywhere parses with required keys.",
    "leading_prose": "True iff lenient passes but strict fails (structured token "
                     "present but not at the start).",
    "normalization_applied": "True if accepting the output needed surrounding "
                             "whitespace strip, trailing punctuation strip, case "
                             "folding, or an opening code-fence strip.",
    "matches_gold.function_call": "tool name exact + every gold argument value-normalized-equal.",
    "matches_gold.binary": "normalized label == gold label (track2 boolq maps yes/no to bool).",
}


# ============================================================
# normalize / helpers
# ============================================================
def normalize(raw):
    """Strip surrounding whitespace. Returns (normalized, changed?)."""
    s = raw.strip()
    return s, (s != raw)


def first_json_span(raw):
    """Extract the first balanced {...} substring. Returns the span or None.

    Brace-counting that respects string literals + escapes so braces inside JSON
    string values don't unbalance the span.
    """
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return None


def try_json_parse(span):
    if span is None:
        return None
    try:
        obj = json.loads(span)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _tool_name(obj):
    for k in ("tool", "name", "function", "function_name"):
        if k in obj:
            return obj[k]
    return None


def required_keys_present(obj, task):
    """JSON-family gate: which keys must be present for valid format."""
    if task in ("function_call", "json"):
        has_name = _tool_name(obj) is not None
        has_args = ("arguments" in obj) or ("parameters" in obj)
        return has_name and has_args
    if task == "extraction":
        return all(k in obj for k in EXTRACTION_KEYS)
    return obj is not None


def _args(obj):
    return obj.get("arguments", obj.get("parameters", {})) or {}


def _norm_value(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    try:
        return float(s)
    except ValueError:
        return s


def _norm_date(v):
    """Normalize a date to YYYY-MM-DD; fall back to string-normalized if unparseable."""
    import datetime
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y",
                "%d %B %Y", "%d %b %Y", "%Y.%m.%d"):
        try:
            return datetime.datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s.lower()


# ============================================================
# format gates (symmetric begins-with) + lenient sensitivity
# ============================================================
def binary_gate(raw, label_set):
    """Symmetric begins-with gate for the binary task."""
    ls = raw.lstrip()
    strict_label, matched_text = None, None
    for lab in sorted(label_set):                       # deterministic order
        m = re.match(re.escape(lab) + r"(?![A-Za-z0-9_])", ls, re.IGNORECASE)
        if m:
            strict_label, matched_text = lab, m.group(0)
            break
    strict_pass = strict_label is not None

    lenient_label = None
    for lab in sorted(label_set):
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(lab) + r"(?![A-Za-z0-9_])",
                     raw, re.IGNORECASE):
            lenient_label = lab
            break
    lenient_pass = lenient_label is not None

    norm = False
    if strict_pass:
        norm = (raw.strip() != raw) or (matched_text != matched_text.upper())
        # trailing punctuation immediately after the label (legacy binary rule)
        after = ls[len(matched_text):]
        if after[:1] in (".", ",", ";", ":", "!"):
            norm = True
    return {
        "strict_pass": strict_pass, "lenient_pass": lenient_pass,
        "leading_prose": lenient_pass and not strict_pass,
        "label": strict_label or lenient_label,
        "normalization_applied": norm,
    }


_FENCE = re.compile(r"```[A-Za-z0-9]*[ \t]*\r?\n?")


def json_gate(raw, task):
    """Symmetric begins-with gate for the json/function_call task."""
    ls = raw.lstrip()
    fence_stripped = False
    body = ls
    fm = _FENCE.match(ls)
    if fm:
        body = ls[fm.end():]
        fence_stripped = True

    strict_obj = None
    if body.lstrip().startswith("{"):
        obj = try_json_parse(first_json_span(body))
        if obj is not None and required_keys_present(obj, task):
            strict_obj = obj
    strict_pass = strict_obj is not None

    lenient_obj = try_json_parse(first_json_span(raw))
    lenient_pass = lenient_obj is not None and required_keys_present(lenient_obj, task)

    norm = (raw.strip() != raw) or fence_stripped
    return {
        "strict_pass": strict_pass, "lenient_pass": lenient_pass,
        "leading_prose": lenient_pass and not strict_pass,
        "parsed": strict_obj if strict_pass else lenient_obj,
        "normalization_applied": norm,
    }


# ============================================================
# content matching (§6 stage 2)
# ============================================================
def matches_gold(task, parsed, gold, track="track1"):
    if task == "binary":
        if track == "track2":
            gold_yes = gold if isinstance(gold, bool) else \
                str(gold).strip().lower() in ("yes", "true", "1")
            return (parsed.upper() == "YES") == gold_yes
        return parsed.upper() == str(gold).strip().upper()

    if task == "router":
        return parsed.upper() == str(gold).strip().upper()

    if task == "extraction":
        for k in EXTRACTION_KEYS:
            if k not in parsed:
                return False
            gv, pv = gold[k], parsed[k]
            if k == "filing_date":
                if _norm_date(pv) != _norm_date(gv):
                    return False
            elif _norm_value(pv) != _norm_value(gv):
                return False
        return True

    if task in ("function_call", "json"):
        gold_name = _tool_name(gold) if isinstance(gold, dict) else gold.get("tool")
        if _tool_name(parsed) != gold_name:
            return False
        gold_args = _args(gold) if isinstance(gold, dict) else {}
        pred_args = _args(parsed)
        for k, gv in gold_args.items():
            if k not in pred_args or _norm_value(pred_args[k]) != _norm_value(gv):
                return False
        return True

    raise ValueError(f"matches_gold: unsupported task {task!r}")


# ============================================================
# main entry (§6)
# ============================================================
def score(task, raw, gold, prompt=None, track="track1"):
    """Return (verdict, fields). Pure + deterministic. Strict gate drives the
    verdict; lenient_format_pass is recorded as a sensitivity metric."""
    fields = {
        "format_pass": False,           # strict (primary)
        "lenient_format_pass": False,   # sensitivity
        "leading_prose": False,
        "normalization_applied": False,
        "content_correct": None,
        "parsed": None,
    }

    # ---- stage 1: FORMAT GATE ----
    if task == "binary":
        g = binary_gate(raw, BINARY_LABELS[track])
        parsed = g["label"]
        fields["parsed"] = g["label"]
    elif task == "router":
        g = binary_gate(raw, ROUTER_LABELS)      # same begins-with enum gate, 5 labels
        parsed = g["label"]
        fields["parsed"] = g["label"]
    elif task in ("function_call", "json", "extraction"):
        g = json_gate(raw, task)
        parsed = g["parsed"]
        fields["parsed"] = g["parsed"]
    elif task == "ifeval":
        il = gold.get("instruction_id_list") if isinstance(gold, dict) else None
        kw = gold.get("kwargs") if isinstance(gold, dict) else None
        ok = ifeval_verifier(prompt, raw, il or [], kw or [])
        fields.update(format_pass=ok, lenient_format_pass=ok)
        if not ok:
            return VERDICT_FORMAT_BREAK, fields
        fields["content_correct"] = True
        return VERDICT_PASS, fields
    else:
        raise ValueError(f"score: unsupported task {task!r}")

    fields["format_pass"] = g["strict_pass"]
    fields["lenient_format_pass"] = g["lenient_pass"]
    fields["leading_prose"] = g["leading_prose"]
    fields["normalization_applied"] = g["normalization_applied"]

    if not fields["format_pass"]:
        return VERDICT_FORMAT_BREAK, fields

    # ---- stage 2: CONTENT ----
    correct = matches_gold(task, parsed, gold, track=track)
    fields["content_correct"] = bool(correct)
    return (VERDICT_PASS if correct else VERDICT_REASONING_ERROR), fields


_IFEVAL_REGISTRY = None


def _load_ifeval_registry():
    """Lazily import the vendored official IFEval verifier (third_party/) and
    ensure nltk punkt data is present. Cached after first load."""
    global _IFEVAL_REGISTRY
    if _IFEVAL_REGISTRY is not None:
        return _IFEVAL_REGISTRY
    import os
    import sys
    tp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
    if tp not in sys.path:
        sys.path.insert(0, tp)
    try:
        import nltk
        for pkg in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            try:
                nltk.data.find(pkg)
            except LookupError:
                try:
                    nltk.download(pkg.split("/")[-1], quiet=True)
                except Exception:
                    pass
        from instruction_following_eval import instructions_registry
    except Exception as e:
        raise RuntimeError(
            "NOT RUN: vendored IFEval verifier unavailable "
            f"(third_party/instruction_following_eval): {e}")
    _IFEVAL_REGISTRY = instructions_registry
    return instructions_registry


def ifeval_verifier(prompt, response, instruction_id_list, kwargs):
    """Official IFEval STRICT instruction-following check (Track-2 only).

    Mirrors google-research instruction_following_eval strict scoring: every
    instruction in `instruction_id_list` must be followed by `response`. `kwargs`
    is the per-instruction argument list aligned with instruction_id_list.

    Per §0, do not fabricate: if the verifier cannot load, _load_ifeval_registry
    raises rather than guess a pass/fail.
    """
    reg = _load_ifeval_registry()
    results = []
    for idx, iid in enumerate(instruction_id_list):
        cls = reg.INSTRUCTION_DICT[iid]
        instr = cls(iid)
        kws = (kwargs[idx] if idx < len(kwargs) else None) or {}
        instr.build_description(**kws)
        args = instr.get_instruction_args()
        if args and "prompt" in args:
            instr.build_description(prompt=prompt)
        ok = bool(response and response.strip() and instr.check_following(response))
        results.append(ok)
    return all(results) if results else False
