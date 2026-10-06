"""Split long trajectories without duplicating supervised targets."""

import numpy as np


def windows(ids, labels, boundaries, max_length=8192, context_tokens=16384):
    """Supervise every original target token exactly once with rolling context.

    Prefer cuts between complete assistant/tool rounds. Reuse the initial
    system/task prefix and up to 16K recent context; repeated context is masked.
    A single oversized round falls back to a token boundary, never dropping it.
    """
    if len(ids) <= max_length:
        return [
            (
                ids,
                labels,
                {
                    "source_start": 0,
                    "source_end": len(ids),
                    "context_start": 0,
                    "prefix_tokens": 0,
                    "partial_round_boundary": False,
                },
            )
        ]
    prefix = boundaries[0]
    assert prefix < max_length // 2, "Task prefix too large for rolling windows"
    assert not np.any(labels[:prefix] != -100)
    overlap_budget = min(context_tokens, (max_length - prefix) // 4)
    start, result = prefix, []
    source_target_count = int((labels != -100).sum())
    while start < len(ids):
        # Overlap only complete preceding rounds when one fits inside 16K.
        context_starts = [b for b in boundaries if max(prefix, start - overlap_budget) <= b < start]
        context = context_starts[0] if context_starts else start
        if context == start and start not in boundaries:
            context = max(prefix, start - overlap_budget)
        limit = min(len(ids), context + max_length - prefix)
        end = limit
        partial = False
        if limit < len(ids):
            candidates = [b for b in boundaries if start < b <= limit]
            if candidates:
                end = candidates[-1]
            else:
                partial = True
        assert end > start
        chunk_ids = np.concatenate([ids[:prefix], ids[context:end]])
        chunk_labels = np.concatenate(
            [np.full(prefix + start - context, -100, np.int32), labels[start:end]]
        )
        assert len(chunk_ids) == len(chunk_labels) <= max_length
        if np.any(chunk_labels != -100):
            result.append(
                (
                    chunk_ids,
                    chunk_labels,
                    {
                        "source_start": start,
                        "source_end": end,
                        "context_start": context,
                        "prefix_tokens": prefix,
                        "partial_round_boundary": partial,
                    },
                )
            )
        start = end
    assert sum(int((x[1] != -100).sum()) for x in result) == source_target_count
    return result
