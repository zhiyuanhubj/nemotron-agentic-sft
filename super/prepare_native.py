"""Native full-string tokenization and assistant-only labels from the production pipeline."""

import copy
import json
import re
import numpy as np


def normalize(row):
    trajectory = row.get("normalized_trajectory") or row["trajectory"]
    messages = copy.deepcopy(trajectory["messages"])
    if row.get("normalized_trajectory"):
        steps = row["trajectory"]["steps"]
        assert not any(s.get("user_content") for s in steps[1:]), (
            "Unhandled later user feedback"
        )
        assert steps[0]["user_content"] and steps[0]["system_prompt"]
        messages = [
            {"role": "system", "content": steps[0]["system_prompt"]},
            {"role": "user", "content": steps[0]["user_content"]},
        ] + messages
    result = []
    for message in messages:
        role = message["role"]
        if role == "exit":
            continue  # Harness bookkeeping, never a model training target.
        assert role in ("system", "user", "assistant", "tool"), role
        m = {
            k: v
            for k, v in message.items()
            if k
            in (
                "role",
                "content",
                "reasoning_content",
                "tool_calls",
                "tool_call_id",
                "name",
            )
        }
        m["content"] = m.get("content") or ""
        assert isinstance(m["content"], str), "Unexpected multimodal input"
        for call in m.get("tool_calls") or []:
            f = call.get("function", call)
            if isinstance(f.get("arguments"), str):
                f["arguments"] = json.loads(f["arguments"])
            assert isinstance(f.get("arguments", {}), dict)
        # Structural template tokens in literal content would make marker masks ambiguous.
        encoded = json.dumps(m, ensure_ascii=False)
        assert "<|im_start|>" not in encoded and "<|im_end|>" not in encoded
        result.append(m)
    assert result[0]["role"] == "system" and result[1]["role"] == "user"
    return result


def tokenize(messages, tokenizer, return_layout=False):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        truncate_history_thinking=False,
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(encoded["input_ids"], dtype=np.int32)
    # Verify the exact input sequence against the native tokenizer path, including BPE boundaries.
    assert ids.tolist() == tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
        truncate_history_thinking=False,
    )
    labels = np.full(len(ids), -100, dtype=np.int32)
    spans = list(re.finditer(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.S))
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert len(spans) == len(assistants)
    offsets = np.asarray(encoded["offset_mapping"])
    supervised = np.zeros(len(text), dtype=np.bool_)
    for match, message in zip(spans, assistants):
        start, end = match.start(1), match.end()
        body = match.group(1)
        # The opening think tag is a generation prefix. Redacted/empty reasoning
        # stays context-only; genuine reasoning, actions and EOS remain targets.
        if body.startswith("<think></think>"):
            start += len("<think></think>")
        elif body.startswith("<think>\n"):
            start += len("<think>\n")
        elif body.startswith("<think>"):
            start += len("<think>")
        supervised[start:end] = True
    # Label a token only if its entire character span is a target. Boundary
    # tokens crossing masked text are conservatively ignored, never retokenized.
    prefix = np.concatenate(([0], np.cumsum(supervised, dtype=np.int64)))
    starts, ends = offsets[:, 0], offsets[:, 1]
    mask = (ends > starts) & ((prefix[ends] - prefix[starts]) == (ends - starts))
    labels[mask] = ids[mask]
    assert np.any(labels != -100)
    if return_layout:
        boundaries = [
            int(np.searchsorted(offsets[:, 0], match.start())) for match in spans
        ]
        assert all(
            offsets[i, 0] == match.start() for i, match in zip(boundaries, spans)
        )
        return ids, labels, boundaries
    return ids, labels
