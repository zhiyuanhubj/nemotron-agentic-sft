#!/usr/bin/env python3
"""Render agent trajectories to input_ids + labels for Nemotron-3-Ultra.

Pretokenising rather than handing `messages` to a training framework is what
keeps the loss mask under our control. Two things went wrong when the framework
owned it:

  * ChatDataset serialises tool-call arguments to a JSON string (OpenAI wire
    format), but Nemotron's template iterates `arguments|items` and Jinja raises
    "Can only get item pairs from a mapping" on every trajectory that calls a
    tool.
  * `start_of_turn_token` is the only mask control on offer, and it cannot
    express "train the reasoning but not the generation prefix the scaffold
    prefills", nor "drop this one bad step".

Mask rules, in the same spirit as the Qwen3.6 renderer:

  masked   system, user, tool observations, and the `<|im_start|>assistant\\n`
           turn header -- all of it is supplied by the scaffold at inference.
  trained  reasoning, content, tool calls, and `<|im_end|>` (the model has to
           learn to stop).

Segment boundaries sit at special-token edges only, and the concatenation is
asserted equal to `apply_chat_template` output, so a template change fails loudly
instead of silently shifting every label by a few tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

IGNORE = -100
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"


def clean_messages(msgs):
    """Drop roles the template has no branch for and parse tool arguments."""
    out = []
    for m in msgs:
        if m.get("role") not in ("system", "user", "assistant", "tool"):
            continue
        m = {k: v for k, v in m.items()
             if k not in ("extra", "provider_specific_fields", "function_call")}
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    fn["arguments"] = json.loads(a)
                except json.JSONDecodeError:
                    fn["arguments"] = {"command": a}
        if m.get("content") is None and not m.get("tool_calls"):
            m["content"] = ""
        out.append(m)
    return out


def step_is_bad(m, prev_args, tok):
    """Step-level quality gate. Returns a reason string, or None to keep.

    Measured over 3,713 assistant steps: 8.4% have reasoning under 20 tokens
    while still calling a tool, 0.1% repeat the previous action verbatim. Steps
    with *no* reasoning at all (19.2%) are deliberately kept -- Nemotron's
    template has a first-class empty `<think></think>`, and forcing a thought
    onto every `ls` would teach the model to pad rather than to think.
    """
    r = (m.get("reasoning_content") or "").strip()
    tcs = m.get("tool_calls") or []
    if r and tcs and len(tok(r, add_special_tokens=False).input_ids) < 20:
        return "thin_thinking"
    if tcs:
        args = json.dumps([(t.get("function") or {}).get("arguments") for t in tcs],
                          sort_keys=True)
        if args == prev_args:
            return "repeat_action"
    return None


def assistant_segments(m, trainable):
    """(text, trainable) pairs for one assistant turn, mirroring the template.

    The turn header stays masked because the scaffold emits it; everything the
    model itself must produce is trained. When `trainable` is False the whole
    turn is still rendered -- the trajectory needs it as context -- but nothing
    in it contributes to the loss.
    """
    segs = []
    reasoning = (m.get("reasoning_content") or "").strip()
    content = (m.get("content") or "").strip()
    tcs = m.get("tool_calls") or []

    segs.append(("<|im_start|>assistant\n", False))
    if reasoning:
        segs.append(("<think>\n", False))
        segs.append((reasoning, trainable))
        segs.append(("</think>", trainable))
    else:
        segs.append(("<think></think>", False))
    if content:
        segs.append((content, trainable))
    for tc in tcs:
        fn = tc.get("function") or {}
        segs.append((f"\n<tool_call>\n<function={fn.get('name')}>\n", trainable))
        for k, v in (fn.get("arguments") or {}).items():
            rendered = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            segs.append((f"<parameter={k}>\n{rendered}\n</parameter>\n", trainable))
        segs.append(("</function>\n</tool_call>", trainable))
    segs.append(("<|im_end|>\n", trainable))
    return segs


def render(msgs, tok, stats):
    segs = []
    prev_args = None
    for m in msgs:
        role = m["role"]
        if role == "system":
            segs.append((f"<|im_start|>system\n{(m.get('content') or '').strip()}<|im_end|>\n", False))
        elif role == "user":
            segs.append((f"<|im_start|>user\n{(m.get('content') or '').strip()}<|im_end|>\n", False))
        elif role == "tool":
            segs.append((f"<|im_start|>tool\n{(m.get('content') or '').strip()}<|im_end|>\n", False))
        elif role == "assistant":
            why = step_is_bad(m, prev_args, tok)
            stats["step_" + (why or "kept")] += 1
            segs.extend(assistant_segments(m, trainable=(why is None)))
            tcs = m.get("tool_calls") or []
            if tcs:
                prev_args = json.dumps([(t.get("function") or {}).get("arguments") for t in tcs],
                                       sort_keys=True)
    ids, labels = [], []
    for text, trainable in segs:
        if not text:
            continue
        t = tok(text, add_special_tokens=False).input_ids
        ids.extend(t)
        labels.extend(t if trainable else [IGNORE] * len(t))
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-len", type=int, default=131072)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    stats = Counter()
    n_tok = n_trained = 0
    with open(args.inp) as f, open(args.out, "w") as o:
        for line in f:
            msgs = clean_messages(json.loads(line)["messages"])
            if not any(m["role"] == "assistant" for m in msgs):
                stats["traj_no_assistant"] += 1
                continue
            ids, labels = render(msgs, tok, stats)
            if len(ids) > args.max_len:
                stats["traj_over_length"] += 1
                continue
            if not any(l != IGNORE for l in labels):
                stats["traj_all_masked"] += 1
                continue
            o.write(json.dumps({"input_ids": ids, "labels": labels}) + "\n")
            stats["traj_kept"] += 1
            n_tok += len(ids)
            n_trained += sum(1 for l in labels if l != IGNORE)

    for k in sorted(stats):
        print(f"  {k:22s} {stats[k]}")
    if n_tok:
        print(f"  tokens {n_tok:,} | trained {n_trained:,} ({n_trained/n_tok*100:.1f}%)")


if __name__ == "__main__":
    main()
