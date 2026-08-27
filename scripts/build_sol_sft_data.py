#!/usr/bin/env python3
"""Build strict Sol-thinking SFT rows from the same 739-task V4 source pool."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "sft_data/sol_v1"
DECODED = ROOT / "thinking_archive/decoded-solmax-thinking.jsonl"
SOURCES = {}

TEST = re.compile(r"\b(pytest|go test|cargo test|npm test|make test|tox|unittest)\b")


def source_group(job: str) -> str | None:
    for group, (prefixes, _) in SOURCES.items():
        if any(job.startswith(prefix) for prefix in prefixes):
            return group
    return None


def reward(result: dict) -> float:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    values = [value for value in rewards.values() if isinstance(value, (int, float))]
    return max(values) if values else 0.0


def decoded_by_response() -> dict[str, dict]:
    latest = {}
    with DECODED.open(errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            response = row.get("source_response_id")
            if response:
                latest[str(response)] = row
    return latest


def visible_text(response: dict) -> str:
    parts = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                text = part.get("text")
                if text:
                    parts.append(str(text))
    return "\n".join(parts).strip()


def tool_calls(response: dict) -> list[dict]:
    calls = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        calls.append(
            {
                "id": str(item.get("call_id") or item.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(item.get("name") or ""),
                    "arguments": str(item.get("arguments") or "{}"),
                },
            }
        )
    return calls


def command_from_call(call: dict) -> str:
    try:
        payload = json.loads((call.get("function") or {}).get("arguments") or "{}")
    except ValueError:
        return ""
    command = payload.get("command") if isinstance(payload, dict) else ""
    return command if isinstance(command, str) else ""


def convert_trajectory(path: Path, decoded: dict[str, dict], group: str) -> tuple[dict | None, str]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, "missing_trajectory"
    original = raw.get("messages") or []
    messages = []
    commands = []
    reasoning_tokens = []
    response_steps = 0

    for item in original:
        role = item.get("role")
        if role in ("system", "user"):
            messages.append({"role": role, "content": str(item.get("content") or "")})
            continue
        if item.get("type") == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": str(item.get("output") or ""),
                }
            )
            continue
        if item.get("object") != "response" and not str(item.get("id") or "").startswith("resp_"):
            continue

        response_steps += 1
        response_id = str(item.get("id") or "")
        usage = item.get("usage") or {}
        billed = (
            ((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)) or 0
        )
        recovered = decoded.get(response_id)
        if billed and (not recovered or recovered.get("complete") is not True):
            return None, "decode_gap"
        reasoning = str((recovered or {}).get("thinking") or "")
        recovered_tokens = int((recovered or {}).get("recovered_tokens") or 0)
        if recovered_tokens:
            reasoning_tokens.append(recovered_tokens)

        calls = tool_calls(item)
        for call in calls:
            command = command_from_call(call)
            if command:
                commands.append(command)
        assistant = {
            "role": "assistant",
            "content": visible_text(item),
            "reasoning_content": reasoning,
        }
        if calls:
            assistant["tool_calls"] = calls
        if assistant["content"] or assistant["reasoning_content"] or calls:
            messages.append(assistant)

    if response_steps < 2:
        return None, "too_few_steps"
    if response_steps > 80:
        return None, "too_many_steps"
    if len(commands) < 2:
        return None, "too_few_actions"

    normalized = [" ".join(command.split())[:160] for command in commands]
    duplicate_ratio = (len(normalized) - len(set(normalized))) / len(normalized)
    if duplicate_ratio > 0.30:
        return None, "repetitive_actions"

    single_cap = 3072 if group.startswith("tmax") else 2048
    if reasoning_tokens and max(reasoning_tokens) > single_cap:
        return None, "reasoning_step_too_long"
    if sum(reasoning_tokens) > 30000:
        return None, "reasoning_trajectory_too_long"

    ran_tests = any(TEST.search(command) for command in commands)
    if group.startswith("swe_") and not ran_tests:
        return None, "swe_without_tests"

    return (
        {
            "messages": messages,
            "steps": response_steps,
            "ran_tests": ran_tests,
            "reasoning_tokens": sum(reasoning_tokens),
            "max_reasoning_step_tokens": max(reasoning_tokens, default=0),
            "commands": len(commands),
            "duplicate_command_ratio": duplicate_ratio,
        },
        "",
    )


def stable_val(group: str, task: str) -> bool:
    digest = hashlib.sha1(f"{group}:{task}".encode()).digest()
    return int.from_bytes(digest[:2], "big") % 10 == 0


def build_sources(root: Path) -> dict:
    return {
        "swe_old": (("solmax_swe_old",), root / "sol_thinking_tasklists/swe_old_exact.txt"),
        "swe_v2": (("solmax_swe_v2",), root / "sol_thinking_tasklists/swe_v2_exact.txt"),
        "tmax1": (
            ("solmax_tmax1", "solmax_tmax_missing"),
            root / "sol_thinking_tasklists/tmax1_exact.txt",
        ),
        "tmax2": (("solmax_tmax2",), root / "sol_thinking_tasklists/tmax2_exact.txt"),
    }


def main() -> None:
    import argparse
    global ROOT, OUT, DECODED, SOURCES
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Harbor + thinking_archive parent")
    ap.add_argument("--out", default="sft_data/sol_v1")
    ap.add_argument("--decoded", default="thinking_archive/decoded-solmax-thinking.jsonl")
    args = ap.parse_args()
    ROOT = Path(args.root).resolve()
    OUT = Path(args.out).resolve() if Path(args.out).is_absolute() else (ROOT / args.out)
    DECODED = Path(args.decoded).resolve() if Path(args.decoded).is_absolute() else (ROOT / args.decoded)
    SOURCES = build_sources(ROOT)
    decoded = decoded_by_response()
    expected = {
        group: set(path.read_text().splitlines())
        for group, (_, path) in SOURCES.items()
    }
    attempts: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for job_dir in (ROOT / "results").iterdir():
        if not job_dir.is_dir():
            continue
        group = source_group(job_dir.name)
        if group is None:
            continue
        for result_path in job_dir.glob("*/result.json"):
            try:
                result = json.loads(result_path.read_text())
            except (OSError, ValueError):
                continue
            task = str(result.get("task_name") or "").split("/")[-1]
            if task not in expected[group]:
                continue
            if (result.get("exception_info") or {}).get("exception_type") or reward(result) <= 0:
                continue
            attempts[(group, task)].append(
                {
                    "started": result.get("started_at") or result.get("finished_at") or "",
                    "job": job_dir.name,
                    "trial": str(result.get("trial_name") or result_path.parent.name),
                    "trajectory": result_path.parent / "agent/mini-swe-agent.trajectory.json",
                }
            )

    rows = []
    rejected = Counter()
    source_stats = defaultdict(Counter)
    for group in SOURCES:
        for task in sorted(expected[group]):
            candidates = sorted(attempts.get((group, task), []), key=lambda row: row["started"])
            if not candidates:
                source_stats[group]["no_verified_pass"] += 1
                continue
            chosen = None
            reasons = []
            for candidate in candidates:
                converted, reason = convert_trajectory(candidate["trajectory"], decoded, group)
                if converted is not None:
                    chosen = {**candidate, **converted}
                    break
                reasons.append(reason)
            if chosen is None:
                reason = reasons[0] if reasons else "no_candidate"
                rejected[f"{group}:{reason}"] += 1
                source_stats[group]["all_passes_rejected"] += 1
                continue
            split = "val" if stable_val(group, task) else "train"
            rows.append(
                {
                    "task": task,
                    "source_group": group,
                    "source_job": chosen["job"],
                    "source_trial": chosen["trial"],
                    "teacher": "gpt-5.6-sol-max",
                    "split": split,
                    "steps": chosen["steps"],
                    "ran_tests": chosen["ran_tests"],
                    "quality": {
                        "reasoning_tokens": chosen["reasoning_tokens"],
                        "max_reasoning_step_tokens": chosen["max_reasoning_step_tokens"],
                        "commands": chosen["commands"],
                        "duplicate_command_ratio": chosen["duplicate_command_ratio"],
                    },
                    "messages": chosen["messages"],
                }
            )
            source_stats[group]["kept"] += 1
            source_stats[group][split] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    for name, selected in {
        "all": rows,
        "train": [row for row in rows if row["split"] == "train"],
        "val": [row for row in rows if row["split"] == "val"],
    }.items():
        with (OUT / f"{name}.jsonl").open("w") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source_pool_tasks": sum(len(tasks) for tasks in expected.values()),
        "kept_tasks": len(rows),
        "train_tasks": sum(row["split"] == "train" for row in rows),
        "val_tasks": sum(row["split"] == "val" for row in rows),
        "source_stats": {group: dict(stats) for group, stats in source_stats.items()},
        "rejections": dict(rejected),
        "filters": {
            "verified_pass": True,
            "fully_decoded_billed_reasoning": True,
            "steps": [2, 80],
            "min_commands": 2,
            "max_duplicate_command_ratio": 0.30,
            "max_single_reasoning_tokens_swe": 2048,
            "max_single_reasoning_tokens_tmax": 3072,
            "max_trajectory_reasoning_tokens": 30000,
            "swe_requires_tests": True,
            "one_trajectory_per_task": True,
            "task_level_stable_10pct_validation": True,
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
