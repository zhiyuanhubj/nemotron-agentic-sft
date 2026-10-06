"""Prepare task-disjoint messages JSONL for full-parameter agentic SFT."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import data_gate
import numpy as np
from prepare_native import normalize, tokenize
from rolling_windows import windows


def load_records(path):
    """Read plain or compressed JSONL; task IDs group related trajectories."""
    opener = gzip.open if path.suffix == ".gz" else open
    records, seen = [], set()
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task = row.get("task_id", row.get("task"))
                if not isinstance(task, str) or not task.strip():
                    raise ValueError("Each record requires a nonempty string task_id")
                messages = normalize(row)
                if not any(m["role"] == "assistant" for m in messages):
                    raise ValueError("Each record requires an assistant response")
            except (AttributeError, KeyError, TypeError, ValueError, AssertionError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            digest = hashlib.sha256(
                json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            # Exact duplicate conversations must not cross the split boundary.
            if digest in seen:
                raise ValueError(f"{path}:{line_number}: duplicate conversation")
            seen.add(digest)
            records.append(
                (
                    {
                        "task": task,
                        "source": str(path.resolve()),
                        "source_line": line_number,
                        "sha256": digest,
                    },
                    messages,
                )
            )
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def split_records(records, validation_fraction, seed):
    tasks = sorted(
        {key["task"] for key, _ in records},
        key=lambda task: hashlib.sha256(f"{seed}:{task}".encode()).hexdigest(),
    )
    if len(tasks) < 2:
        raise ValueError("Automatic splitting requires at least two distinct tasks")
    count = max(1, min(len(tasks) - 1, round(len(tasks) * validation_fraction)))
    validation_tasks = set(tasks[:count])
    train = [r for r in records if r[0]["task"] not in validation_tasks]
    validation = [r for r in records if r[0]["task"] in validation_tasks]
    return train, validation


def check_disjoint(train, validation):
    overlap = {k["task"] for k, _ in train} & {k["task"] for k, _ in validation}
    if overlap:
        raise ValueError(f"Training and validation share task IDs: {sorted(overlap)[:10]}")
    if {k["sha256"] for k, _ in train} & {k["sha256"] for k, _ in validation}:
        raise ValueError("Training and validation contain identical conversations")


def write_split(directory, records, tokenizer, max_length, context_tokens):
    """Stream token buffers to disk, then materialize memory-mappable arrays."""
    directory.mkdir(parents=True)
    offsets, targets = [0], 0
    raw_ids, raw_labels = directory / "ids.tmp", directory / "labels.tmp"
    with (
        raw_ids.open("wb") as ids_file,
        raw_labels.open("wb") as labels_file,
        (directory / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
    ):
        for index, (key, messages) in enumerate(records):
            try:
                ids, labels, boundaries = tokenize(messages, tokenizer, return_layout=True)
                parts = windows(ids, labels, boundaries, max_length, context_tokens)
            except (ValueError, AssertionError) as exc:
                raise ValueError(f"{key['source']}:{key['source_line']}: {exc}") from exc
            for segment, (part_ids, part_labels, layout) in enumerate(parts):
                # The collator removes the first label during the causal shift.
                if part_labels[0] != -100:
                    raise ValueError("Window begins with a target lacking causal context")
                count = int((part_labels != -100).sum())
                part_ids.astype(np.uint32).tofile(ids_file)
                part_labels.astype(np.int32).tofile(labels_file)
                offsets.append(offsets[-1] + len(part_ids))
                targets += count
                manifest.write(
                    json.dumps(
                        {
                            **key,
                            **layout,
                            "segment": segment,
                            "segments": len(parts),
                            "tokens": len(part_ids),
                            "target_tokens": count,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if (index + 1) % 100 == 0:
                print(f"{directory.name}: prepared {index + 1} trajectories", flush=True)
    for raw, name, dtype in [
        (raw_ids, "input_ids.npy", np.uint32),
        (raw_labels, "labels.npy", np.int32),
    ]:
        source = np.memmap(raw, mode="r", dtype=dtype)
        output = np.lib.format.open_memmap(
            directory / name, mode="w+", dtype=dtype, shape=(offsets[-1],)
        )
        for start in range(0, len(source), 8 * 1024 * 1024):
            output[start : start + 8 * 1024 * 1024] = source[start : start + 8 * 1024 * 1024]
        output.flush()
        del output, source
        raw.unlink()
    np.save(directory / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    return {
        "rows": len(offsets) - 1,
        "trajectories": len(records),
        "tasks": len({key["task"] for key, _ in records}),
        "tokens": offsets[-1],
        "target_tokens": targets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Messages JSONL or JSONL.gz")
    parser.add_argument("--validation", type=Path, help="Optional task-disjoint validation JSONL")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model", type=Path, required=True, help="Local Nemotron tokenizer directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--context-tokens", type=int, default=16384)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.max_length < 8 or args.context_tokens < 1:
        parser.error("--max-length must be at least 8 and --context-tokens must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    train = load_records(args.train)
    if args.validation:
        validation = load_records(args.validation)
    else:
        train, validation = split_records(train, args.validation_fraction, args.seed)
    check_disjoint(train, validation)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model.resolve()), trust_remote_code=True, local_files_only=True, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError("Assistant masking requires a fast tokenizer with character offsets")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": 3,
        "max_length": args.max_length,
        "context_tokens": args.context_tokens,
        "seed": args.seed,
        "model": str(args.model.resolve()),
        "tokenizer_sha256": data_gate.tokenizer_fingerprint(args.model),
        "source_sha256": {
            str(path.resolve()): data_gate.digest(path)
            for path in [args.train] + ([args.validation] if args.validation else [])
        },
        "splits": {},
    }
    for name, records in [("train", train), ("validation", validation)]:
        summary["splits"][name] = write_split(
            args.output / name, records, tokenizer, args.max_length, args.context_tokens
        )
    data_gate.publish_ready(args.output, summary)
    data_gate.verify(args.output)
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
