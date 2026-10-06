"""Capped task repeats, slightly fewer JS attempts, long-trajectory windows,
and independent SWE-bench Pro Verified reference validation.
"""

import collections
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from prepare_native import normalize, tokenize
import os

ROOT = Path(os.environ.get("SOURCE_ROOT", "raw"))
MODEL = os.environ["NEMOTRON_MODEL"]
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "131072"))
FILES = {
    "01_clean_thinking_and_actions": "all_successful_trials.jsonl.gz",
    "02_clean_actions_dirty_thinking_removed": "all_successful_trials.jsonl.gz",
    "05_swe_rebench_v2": "all_reward1_without_successful_hack.jsonl.gz",
    "05_swebench_pro_verified": "all_reward1_without_successful_hack.jsonl.gz",
}

REFERENCE = Path(os.environ["REFERENCE_JSONL"])
BENCHMARK = Path(os.environ["BENCHMARK_JSONL"])
OUT = Path(os.environ.get("PREPARED_DIR", "prepared"))


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_training():
    grouped = collections.defaultdict(list)
    stats = {
        "input_by_group": {},
        "action_duplicates": [],
        "cap_exclusions": [],
        "js_adjustment_exclusions": [],
    }
    for priority, (group, filename) in enumerate(FILES.items()):
        n = 0
        for line in gzip.open(ROOT / "data" / group / filename, "rt"):
            row = json.loads(line)
            messages = normalize(row)
            actions = []
            for m in messages:
                if m["role"] == "assistant":
                    for c in m.get("tool_calls") or []:
                        f = c.get("function", c)
                        actions.append(
                            {"name": f["name"], "arguments": f.get("arguments", {})}
                        )
            key = {
                "task": row["task"],
                "group": group,
                "language": row["language"],
                "trial": row.get("trial"),
                "run": row.get("run"),
                "source_line": n + 1,
                "sha256": digest(messages),
                "action_sha256": digest(actions or messages),
                "priority": priority,
            }
            grouped[row["task"]].append((key, messages))
            n += 1
        stats["input_by_group"][group] = n
    selected = {}
    for task, records in grouped.items():
        records.sort(key=lambda x: (x[0]["priority"], x[0]["sha256"]))
        unique, seen = [], set()
        for entry in records:
            if entry[0]["action_sha256"] in seen:
                stats["action_duplicates"].append(entry[0])
            else:
                unique.append(entry)
                seen.add(entry[0]["action_sha256"])
        first = unique[0]
        # Prefer a second distinct action trajectory from another requested group.
        rest = sorted(
            unique[1:],
            key=lambda x: (
                x[0]["group"] == first[0]["group"],
                x[0]["priority"],
                x[0]["sha256"],
            ),
        )
        selected[task] = [first] + rest[:1]
        stats["cap_exclusions"].extend(x[0] for x in rest[1:])
    # Keep every task. Reduce only second JS attempts, aiming near 28.5% of
    # trajectories; this is a modest adjustment, not an exact benchmark match.
    non_js = sum(len(v) for v in selected.values() if v[0][0]["language"] != "js")
    js_tasks = sorted(
        [t for t, v in selected.items() if v[0][0]["language"] == "js"],
        key=lambda t: digest(t),
    )
    available_js = sum(len(selected[t]) for t in js_tasks)
    desired_js = max(len(js_tasks), min(available_js, round(non_js * 0.285 / 0.715)))
    extra_slots = desired_js - len(js_tasks)
    for task in js_tasks:
        if len(selected[task]) == 2:
            if extra_slots:
                extra_slots -= 1
            else:
                stats["js_adjustment_exclusions"].append(selected[task].pop()[0])
    records = [entry for task in sorted(selected) for entry in selected[task]]
    assert len(selected) == len(grouped)
    assert max(collections.Counter(e[0]["task"] for e in records).values()) <= 2
    stats["candidate_tasks"] = len(grouped)
    stats["selected_trajectories"] = len(records)
    stats["selected_languages"] = dict(
        collections.Counter(e[0]["language"] for e in records)
    )
    print(
        "SELECTION",
        json.dumps({k: v for k, v in stats.items() if not isinstance(v, list)}),
        flush=True,
    )
    return records, set(grouped), stats


def reference_messages(row):
    result = []
    for i, step in enumerate(row["trajectory"]["steps"]):
        if i == 0:
            result.append({"role": "system", "content": step["system_prompt"]})
        if step.get("user_content"):
            result.append({"role": "user", "content": step["user_content"]})
        assistant = copy.deepcopy(step["assistant_content"])
        assistant["role"] = "assistant"
        result.append(assistant)
        observations = step.get("observation") or []
        calls = assistant.get("tool_calls") or []
        assert len(observations) <= len(calls)
        for call, observation in zip(calls, observations):
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": observation["content"],
                }
            )
    return normalize({"trajectory": {"messages": result}})


def load_validation(training_tasks, size=128):
    benchmark = {x["instance_id"]: x for x in map(json.loads, BENCHMARK.open())}
    candidates = collections.defaultdict(list)
    for line in gzip.open(REFERENCE, "rt"):
        row = json.loads(line)
        task = row["task_id"]
        if task in training_tasks:
            continue
        if (
            row.get("correct") is not True
            or row.get("status") != "completed"
            or row.get("error")
        ):
            continue  # Some exported "resolved" rows still have harness failures.
        assert task in benchmark
        candidates[benchmark[task]["repo_language"]].append(row)
    language_counts = collections.Counter(
        x["repo_language"] for x in benchmark.values()
    )
    exact = {
        lang: size * count / len(benchmark) for lang, count in language_counts.items()
    }
    quotas = {lang: math.floor(count) for lang, count in exact.items()}
    for lang in sorted(quotas, key=lambda k: (-(exact[k] - quotas[k]), k))[
        : size - sum(quotas.values())
    ]:
        quotas[lang] += 1
    result = []
    for language, number in quotas.items():
        rows = sorted(
            candidates[language],
            key=lambda x: digest("v41-independent-validation:" + x["task_id"]),
        )
        assert len(rows) >= number, (language, len(rows))
        for row in rows[:number]:
            task = row["task_id"]
            messages = reference_messages(row)
            key = {
                "task": task,
                "language": language,
                "group": "independent_swebench_pro_verified",
                "reference_model": row["model"],
                "reference_correct": True,
                "reference_run": row["run_id"],
                "sha256": digest(messages),
            }
            result.append((key, messages))
    assert len(result) == size and len({x[0]["task"] for x in result}) == size
    assert not training_tasks.intersection(x[0]["task"] for x in result)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "validation_benchmark_tasks.jsonl").open("w") as f:
        for key, _ in result:
            # Exact task IDs also allow a later independent agent evaluation.
            f.write(
                json.dumps(
                    {
                        "instance_id": key["task"],
                        "language": key["language"],
                        "repo": benchmark[key["task"]]["repo"],
                    }
                )
                + "\n"
            )
    return result


def windows(ids, labels, boundaries):
    """Supervise every original target token exactly once with rolling context.

    Prefer cuts between complete assistant/tool rounds. Reuse the initial
    system/task prefix and up to 16K recent context; repeated context is masked.
    A single oversized round falls back to a token boundary, never dropping it.
    """
    if len(ids) <= MAX_LENGTH:
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
    assert prefix < MAX_LENGTH // 2, "Task prefix too large for rolling windows"
    assert not np.any(labels[:prefix] != -100)
    overlap_budget = min(16384, (MAX_LENGTH - prefix) // 4)
    start, result = prefix, []
    source_target_count = int((labels != -100).sum())
    while start < len(ids):
        # Overlap only complete preceding rounds when one fits inside 16K.
        context_starts = [
            b for b in boundaries if max(prefix, start - overlap_budget) <= b < start
        ]
        context = context_starts[0] if context_starts else start
        if context == start and start not in boundaries:
            context = max(prefix, start - overlap_budget)
        limit = min(len(ids), context + MAX_LENGTH - prefix)
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
        assert len(chunk_ids) == len(chunk_labels) <= MAX_LENGTH
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


def write_split(name, records, tokenizer):
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    lengths, counts, offsets = [], [], [0]
    all_ids, all_labels, meta, long_records = [], [], [], []
    with (OUT / (name + ".jsonl")).open("w") as raw:
        for index, (key, messages) in enumerate(records):
            ids, labels, boundaries = tokenize(messages, tokenizer, return_layout=True)
            parts = windows(ids, labels, boundaries)
            raw.write(
                json.dumps(
                    {"messages": messages, "provenance": key}, ensure_ascii=False
                )
                + "\n"
            )
            if len(parts) > 1:
                long_records.append(
                    {**key, "original_tokens": len(ids), "segments": len(parts)}
                )
            for part_index, (part_ids, part_labels, layout) in enumerate(parts):
                key_part = {
                    **key,
                    **layout,
                    "segment": part_index,
                    "segments": len(parts),
                    "trajectory_tokens": len(ids),
                    "tokens": len(part_ids),
                    "target_tokens": int((part_labels != -100).sum()),
                }
                meta.append(key_part)
                all_ids.append(part_ids)
                all_labels.append(part_labels)
                offsets.append(offsets[-1] + len(part_ids))
                lengths.append(len(part_ids))
                counts.append(key_part["target_tokens"])
            if (index + 1) % 20 == 0:
                print(
                    name, "trajectories", index + 1, "segments", len(meta), flush=True
                )
    with (directory / "manifest.jsonl").open("w") as f:
        for entry in meta:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    np.save(directory / "input_ids.npy", np.concatenate(all_ids).astype(np.uint32))
    np.save(directory / "labels.npy", np.concatenate(all_labels))
    np.save(directory / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    return {
        "rows": len(meta),
        "trajectories": len(records),
        "tasks": len({x[0]["task"] for x in records}),
        "tokens": sum(lengths),
        "target_tokens": sum(counts),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "trajectory_languages": dict(
            collections.Counter(x[0]["language"] for x in records)
        ),
        "task_languages": dict(
            collections.Counter(
                {x[0]["task"]: x[0]["language"] for x in records}.values()
            )
        ),
        "segment_languages": dict(collections.Counter(x["language"] for x in meta)),
        "groups": dict(collections.Counter(x[0]["group"] for x in records)),
        "long_trajectories_retained": long_records,
        "partial_round_windows": sum(x["partial_round_boundary"] for x in meta),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not any(OUT.iterdir()), "Use a fresh PREPARED_DIR"
    training, training_tasks, selection = load_training()
    validation = load_validation(training_tasks)
    tok = AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, local_files_only=True
    )
    summary = {
        "version": 2,
        "source_root": str(ROOT),
        "max_length": MAX_LENGTH,
        "selection": selection,
        "validation": {
            "source": str(REFERENCE),
            "source_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
            "type": "held-out Verified tasks with successful independent teacher reference trajectories; loss is not agent pass rate",
        },
        "splits": {},
        "policy": "max 2 distinct action trajectories per task; all selected tasks train; modest JS reduction; no length exclusion; rolling windows supervise targets once",
    }
    summary["splits"]["train"] = write_split("train", training, tok)
    summary["splits"]["validation"] = write_split("validation", validation, tok)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    import data_gate

    data_gate.publish_ready(OUT, stats=summary)
    print(
        "PREPARE_V2_COMPLETE",
        json.dumps(
            {
                k: {a: b for a, b in v.items() if a != "long_trajectories_retained"}
                for k, v in summary["splits"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
