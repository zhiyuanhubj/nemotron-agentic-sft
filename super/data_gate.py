"""Hash and verify prepared arrays, token counts and task-disjoint splits."""

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tokenizer_fingerprint(directory):
    """Identify local tokenization assets independently of their filesystem path."""
    directory = Path(directory)
    names = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
    ]
    files = [directory / name for name in names if (directory / name).is_file()]
    files.extend(sorted(directory.glob("chat_templates/*.jinja")))
    return {str(path.relative_to(directory)): digest(path) for path in files}


def publish_ready(directory, stats):
    directory = Path(directory)
    stats["ready_for_training"] = True
    stats["files_sha256"] = {
        str(p.relative_to(directory)): digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name not in {"summary.json", "READY.json"}
    }
    summary = directory / "summary.json"
    summary.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    ready = {"ready_for_training": True, "summary_sha256": digest(summary)}
    (directory / "READY.json").write_text(json.dumps(ready, indent=2) + "\n")


def verify(directory, verify_hashes=True):
    directory = Path(directory).resolve()
    ready = json.loads((directory / "READY.json").read_text())
    assert ready["ready_for_training"] is True
    assert ready["summary_sha256"] == digest(directory / "summary.json")
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["ready_for_training"] is True
    if verify_hashes:
        for relative, sha in summary["files_sha256"].items():
            file = directory / relative
            assert file.resolve().is_relative_to(directory), relative
            assert digest(file) == sha, relative
    tasks = []
    for split, info in summary["splits"].items():
        entries = [
            json.loads(line)
            for line in (directory / split / "manifest.jsonl").read_text().splitlines()
        ]
        ids = np.load(directory / split / "input_ids.npy", mmap_mode="r")
        labels = np.load(directory / split / "labels.npy", mmap_mode="r")
        offsets = np.load(directory / split / "offsets.npy", mmap_mode="r")
        assert len(entries) == info["rows"] and offsets.shape == (len(entries) + 1,)
        assert int(offsets[0]) == 0 and np.all(np.diff(offsets) > 0)
        assert ids.ndim == labels.ndim == 1
        assert ids.dtype == np.uint32 and labels.dtype == np.int32
        assert offsets.dtype == np.int64
        assert len(ids) == len(labels) == int(offsets[-1]) == info["tokens"]
        assert all(
            int(offsets[i + 1] - offsets[i]) == row["tokens"] for i, row in enumerate(entries)
        )
        assert all(row["tokens"] <= summary["max_length"] for row in entries)
        assert sum(row["target_tokens"] for row in entries) == info["target_tokens"] > 0
        for i, row in enumerate(entries):
            start, end = map(int, offsets[i : i + 2])
            target = labels[start:end]
            assert target[0] == -100, "First token has no causal context"
            mask = target != -100
            assert int(mask.sum()) == row["target_tokens"] > 0
            assert np.all(target[mask] == ids[start:end][mask])
        tasks.append({row["task"] for row in entries})
    assert len(tasks) == 2 and not tasks[0].intersection(tasks[1]), (
        "Validation tasks must be independent"
    )
    return ready


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.prepared_dir),
            indent=2,
        )
    )
