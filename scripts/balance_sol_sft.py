#!/usr/bin/env python3
"""Deterministically match the prior V4 source-token mixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(".")
WEIGHTS = {
    "swe_old": 3.54,
    "swe_v2": 1.00,
    "tmax1": 2.38,
    "tmax2": 2.38,
}


def copies(row: dict) -> int:
    weight = WEIGHTS[row["source_group"]]
    whole = int(weight)
    fraction = weight - whole
    digest = hashlib.sha1(
        f'{row["source_group"]}:{row["task"]}'.encode()
    ).digest()
    return whole + (int.from_bytes(digest[:2], "big") / 65536 < fraction)


def order_key(row: dict) -> str:
    return hashlib.sha1(
        f'{row["source_group"]}:{row["task"]}:{row["_copy"]}'.encode()
    ).hexdigest()


def balance(split: str) -> None:
    rows = [json.loads(line) for line in (ROOT / f"{split}.jsonl").open()]
    expanded = []
    for row in rows:
        for index in range(copies(row)):
            copy = dict(row)
            copy["_copy"] = index
            expanded.append(copy)
    expanded.sort(key=order_key)
    with (ROOT / f"{split}_balanced.jsonl").open("w") as handle:
        for row in expanded:
            row.pop("_copy", None)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(split, len(rows), "->", len(expanded))


def main() -> None:
    import argparse
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sft_data/sol_v1")
    args = ap.parse_args()
    ROOT = Path(args.root)
    balance("train")
    balance("val")


if __name__ == "__main__":
    main()
