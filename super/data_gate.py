"""Freeze and verify prepared arrays and their source manifest before training."""

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


def publish_ready(directory, stats, accepted_manifest=None):
    directory = Path(directory)
    stats["ready_for_training"] = True
    stats["files_sha256"] = {
        str(p.relative_to(directory)): digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name not in {"summary.json", "READY.json"}
    }
    if accepted_manifest:
        stats["accepted_manifest_sha256"] = digest(accepted_manifest)
    summary = directory / "summary.json"
    summary.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    ready = {"ready_for_training": True, "summary_sha256": digest(summary)}
    if accepted_manifest:
        ready["accepted_manifest_sha256"] = digest(accepted_manifest)
    (directory / "READY.json").write_text(json.dumps(ready, indent=2) + "\n")


def verify(directory, accepted_manifest=None, audit_directory=None, verify_hashes=True):
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
    if accepted_manifest:
        from integrity_holds import held_source_hashes

        accepted = json.loads(Path(accepted_manifest).read_text())
        assert accepted["review_complete"] is True and accepted["accepted"]
        assert ready["accepted_manifest_sha256"] == digest(accepted_manifest)
        assert len({x["task_id"] for x in accepted["accepted"]}) == len(
            accepted["accepted"]
        )
        held = held_source_hashes(Path(audit_directory)) if audit_directory else set()
        for row in accepted["accepted"]:
            assert row["integrity_status"] == "accepted"
            assert row["source_sha256"] not in held
            assert digest(row["source"]) == row["source_sha256"]
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
        assert len(ids) == len(labels) == int(offsets[-1]) == info["tokens"]
        assert all(
            int(offsets[i + 1] - offsets[i]) == row["tokens"]
            for i, row in enumerate(entries)
        )
        assert all(row["tokens"] <= summary["max_length"] for row in entries)
        assert sum(row["target_tokens"] for row in entries) == info["target_tokens"] > 0
        tasks.append({row["task"] for row in entries})
    assert len(tasks) == 2 and not tasks[0].intersection(tasks[1]), (
        "Validation tasks must be independent"
    )
    return ready


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("--accepted-manifest", type=Path)
    parser.add_argument("--integrity-audit", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.prepared_dir, args.accepted_manifest, args.integrity_audit),
            indent=2,
        )
    )
