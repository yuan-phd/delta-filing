#!/usr/bin/env python
"""Dry-run the preference-loss math + the §2a retention-mask fix on a tiny batch,
printing intermediate values (claude_code_spec.md §0 rule 5).

Runs fully on CPU -- no bitsandbytes, no CUDA, no base-model download. The
tokenizer is loaded from the local SFT adapter dir so this is offline.

  python dryrun_loss.py
"""
import json

import torch
import torch.nn.functional as F

import pref_losses as PL

TOK_DIR = "adapters/sft_pytorch"   # local tokenizer.json (no model weights needed)
PREF_DATA = "training_data/dpo_pytorch.jsonl"

SYSTEM_PROMPT = (
    "You are Delta Filing, a financial analyst AI specializing in SEC filing analysis. "
    "You analyze 10-K and 10-Q filings, detect year-over-year changes in risk disclosures, "
    "track management guidance accuracy, and flag potential red flags. "
    "Always reference specific filing sections, cite specific numbers and dates, "
    "provide analytical judgment, and note caveats. "
    "Do not give investment advice or predict stock prices."
)
TOOL_SYSTEM_PROMPT = (
    "You are Delta Filing, a financial analyst AI specializing in SEC filing analysis. "
    "You have access to tools for retrieving SEC filings, financial data, news, and insider trades. "
    "When you need data, call the appropriate tool. When you have data, analyze it thoroughly.\n\n"
    'To use a tool, respond with JSON: {"tool": "name", "arguments": {...}}\n'
    "Respond with ONLY the JSON, nothing else."
)
SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] | trim }}<|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


class StubLM(torch.nn.Module):
    """Minimal CausalLM stub that mimics HF's interface (.logits, .loss with the
    standard next-token shift and ignore_index=-100). Lets us numerically show
    that the §2a mask actually changes the retention CE on CPU."""
    def __init__(self, vocab, dim=16, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, labels=None):
        logits = self.head(self.emb(input_ids))
        loss = None
        if labels is not None:
            sl = logits[:, :-1, :].reshape(-1, logits.size(-1))
            ll = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(sl, ll, ignore_index=-100)
        return type("O", (), {"logits": logits, "loss": loss})


def hr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def part1_preference_loss():
    hr("PART 1 — §2c preference-loss inventory on 3 synthetic pairs")
    print(f"beta={PL.DEFAULT_BETA}  CLAMP_C={PL.DEFAULT_CLAMP_C}  cDPO eps={PL.DEFAULT_CDPO_EPS}")
    print("(inputs are length-normalized avg log-probs: pi_chosen, pi_rejected, ref_*)")
    # (pi_chosen, pi_rejected, ref_chosen, ref_rejected, note)
    cases = [
        (-1.20, -1.80, -1.50, -1.50, "chosen preferred -> margin>0 (healthy)"),
        (-2.40, -0.80, -1.50, -1.50, "rejected preferred -> margin<0 (DPO blows up)"),
        (-1.51, -1.49, -1.50, -1.50, "near tie -> margin~0"),
    ]
    for i, (pc, pr, rc, rr, note) in enumerate(cases):
        print(f"\n-- case {i}: {note}")
        print(f"   pi_chosen={pc}  pi_rejected={pr}  ref_chosen={rc}  ref_rejected={rr}")
        margin = None
        row = {}
        for cond in PL.CONDITIONS:
            loss, margin = PL.preference_loss(pc, pr, rc, rr, condition=cond)
            row[cond] = float(loss)
        print(f"   margin = beta*((pc-rc)-(pr-rr)) = {margin:+.4f}")
        for cond in PL.CONDITIONS:
            print(f"     {cond:<10} loss = {row[cond]:.4f}")

    hr("PART 1b — unbounded(DPO) vs bounded(DPO+clamp) as margin -> -inf")
    print("Shows the causal control: clamp saturates the loss/gradient, DPO does not.")
    print(f"{'pi_chosen':>10}{'margin':>10}{'DPO':>12}{'DPO+clamp':>12}{'IPO':>10}{'cDPO':>10}")
    for pc in [-1.5, -3.5, -7.5, -15.5, -25.5]:
        # ref symmetric, rejected fixed -> drive margin very negative
        pr, rc, rr = -1.5, -1.5, -1.5
        vals = {}
        margin = None
        for cond in PL.CONDITIONS:
            loss, margin = PL.preference_loss(pc, pr, rc, rr, condition=cond)
            vals[cond] = float(loss)
        print(f"{pc:>10.1f}{margin:>10.2f}{vals['dpo']:>12.4f}"
              f"{vals['dpo_clamp']:>12.4f}{vals['ipo']:>10.4f}{vals['cdpo']:>10.4f}")
    # analytic check
    c = PL.DEFAULT_CLAMP_C
    sat = float(-F.logsigmoid(torch.tensor(-c)))
    print(f"\n   analytic saturation: DPO+clamp loss -> -logsigmoid(-C) = {sat:.4f} at C={c}")
    print(f"   (every row's DPO+clamp value above is capped at this once margin <= -C)")


def part2_retention_mask():
    hr("PART 2 — §2a retention-mask fix on 3 REAL preference examples")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)
    tok.chat_template = SIMPLE_CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"tokenizer: {TOK_DIR}  vocab_size={tok.vocab_size}")

    raw = [json.loads(l) for l in open(PREF_DATA) if l.strip()]
    # pick a short tool-calling pair + two longer analysis pairs
    picks = []
    for cat in ("tool_calling", "section_summary", "red_flag"):
        for ex in raw:
            if ex.get("category") == cat:
                picks.append(ex)
                break

    # Pre-encode so the stub embedding is sized to the true max token id
    # (Qwen has special/added tokens above tokenizer.vocab_size).
    encoded = []
    for ex in picks:
        sp = TOOL_SYSTEM_PROMPT if ex.get("category") == "tool_calling" else SYSTEM_PROMPT
        prompt_msgs = [{"role": "system", "content": sp},
                       {"role": "user", "content": ex["question"]}]
        chosen_msgs = prompt_msgs + [{"role": "assistant", "content": ex["chosen"]}]
        chosen_text = tok.apply_chat_template(chosen_msgs, tokenize=False)
        chosen_ids = tok.encode(chosen_text, max_length=1024, truncation=True)
        prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(prompt_text, max_length=1024, truncation=True)
        encoded.append((ex, chosen_ids, prompt_ids))
    max_id = max(max(c) for _, c, _ in encoded)
    stub = StubLM(max_id + 1)

    for i, (ex, chosen_ids, prompt_ids) in enumerate(encoded):
        prompt_len = len(prompt_ids)
        clean_prefix = chosen_ids[:prompt_len] == prompt_ids

        chosen_t = torch.tensor(chosen_ids, dtype=torch.long)
        labels = PL.build_assistant_labels(chosen_t, prompt_len)
        span = PL.assert_assistant_mask(labels, chosen_t, prompt_len)

        # decode the supervised (unmasked) span to prove it == assistant content
        sup_ids = [int(t) for t, l in zip(chosen_ids, labels.tolist()) if l != -100]
        sup_decoded = tok.decode(sup_ids)

        # OLD (buggy) full-seq CE vs NEW assistant-only CE — numeric proof mask is live
        full_labels = chosen_t.clone()
        x = chosen_t.unsqueeze(0)
        with torch.no_grad():
            old_ce = float(stub(x, labels=full_labels.unsqueeze(0)).loss)
            new_ce = float(stub(x, labels=labels.unsqueeze(0)).loss)

        print(f"\n-- example {i}  category={ex.get('category')}")
        print(f"   total tokens         = {len(chosen_ids)}")
        print(f"   prompt_len (masked)  = {prompt_len}")
        print(f"   assistant span len   = {span}")
        print(f"   clean token prefix?  = {clean_prefix}")
        print(f"   #unmasked labels     = {int((labels != -100).sum())}  "
              f"(assertion #unmasked == span: PASS)")
        print(f"   OLD full-seq CE      = {old_ce:.4f}  (supervised {len(chosen_ids)-1} tokens)")
        print(f"   NEW assistant-only CE= {new_ce:.4f}  (supervised {span} tokens)")
        print(f"   supervised span starts with: {sup_decoded[:80]!r}")
        # sanity: supervised text should match the assistant content head
        head_ok = ex["chosen"].strip()[:30] in sup_decoded
        print(f"   supervised text == assistant content head? {head_ok}")


def part3_completion_logprobs():
    hr("PART 3 — compute_log_probs is COMPLETION-ONLY (no prompt-length artifact)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)
    tok.chat_template = SIMPLE_CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ex = next(json.loads(l) for l in open(PREF_DATA) if l.strip())
    sp = TOOL_SYSTEM_PROMPT if ex.get("category") == "tool_calling" else SYSTEM_PROMPT
    prompt_msgs = [{"role": "system", "content": sp}, {"role": "user", "content": ex["question"]}]
    chosen_msgs = prompt_msgs + [{"role": "assistant", "content": ex["chosen"]}]
    chosen_ids = tok.encode(tok.apply_chat_template(chosen_msgs, tokenize=False),
                            max_length=1024, truncation=True)
    prompt_ids = tok.encode(tok.apply_chat_template(prompt_msgs, tokenize=False,
                            add_generation_prompt=True), max_length=1024, truncation=True)
    completion_start = len(prompt_ids)
    x = torch.tensor(chosen_ids, dtype=torch.long)

    stub = StubLM(max(chosen_ids) + 1)
    dev = torch.device("cpu")
    comp_only = float(PL.compute_log_probs(stub, x, completion_start, dev))

    # manual reference: full-seq avg (the OLD behavior) for contrast
    with torch.no_grad():
        logits = stub(x.unsqueeze(0)).logits
    tlp = torch.gather(F.log_softmax(logits[:, :-1, :].float(), -1),
                       -1, x[1:].unsqueeze(0).unsqueeze(-1)).squeeze(-1)
    full_avg = float(tlp.sum() / (len(chosen_ids) - 1))
    comp_manual = float(tlp[:, completion_start - 1:].sum() / (len(chosen_ids) - completion_start))

    print(f"  total tokens L            = {len(chosen_ids)}")
    print(f"  completion_start          = {completion_start}  (prompt tokens EXCLUDED)")
    print(f"  completion length (denom) = {len(chosen_ids) - completion_start}")
    print(f"  compute_log_probs() value = {comp_only:.6f}")
    print(f"  manual completion-only    = {comp_manual:.6f}   (matches: {abs(comp_only-comp_manual)<1e-5})")
    print(f"  OLD full-seq avg (denom={len(chosen_ids)-1}) = {full_avg:.6f}  "
          f"<- differs => prompt length no longer affects the score")
    print("  (stub weights are random; absolute values are meaningless -- the point")
    print("   is the denominator/span: completion-only vs full-sequence.)")


if __name__ == "__main__":
    part1_preference_loss()
    part2_retention_mask()
    part3_completion_logprobs()
    print("\nDRY-RUN COMPLETE (loss math + retention mask + completion-only logprobs).")
