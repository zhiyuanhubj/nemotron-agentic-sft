"""Portable version of the October 6 reviewed Opus native-token preparation."""

import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

import data_gate
from integrity_holds import held_source_hashes
from prepare_native import normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--integrity-audit", type=Path, required=True)
    parser.add_argument("--collection-protocol", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=123)
    parser.add_argument("--validation-tasks", type=int, default=128)
    args = parser.parse_args()
    import prepare_v41 as pipeline

    selected_manifest = json.loads(args.accepted_manifest.read_text())
    assert selected_manifest["review_complete"] is True
    selected = selected_manifest["accepted"]
    assert len(selected) == len({x["task_id"] for x in selected}) == args.expected_tasks
    benchmark = {
        x["instance_id"]: x for x in map(json.loads, pipeline.BENCHMARK.open())
    }
    assert {x["task_id"] for x in selected}.issubset(benchmark)
    assert args.integrity_audit.is_dir(), (
        "Supply the current review directory, including holds"
    )
    held = held_source_hashes(args.integrity_audit)
    protocol = json.loads(args.collection_protocol.read_text())
    collection = {}
    for group in protocol["groups"]:
        spec = json.loads(Path(group["manifest"]).read_text())
        collection[spec["run_id"]] = {
            "kind": group.get("kind", "fresh_primary"),
            "own_patch_repair": spec.get("own_patch_repair", False),
        }
    out = pipeline.OUT
    assert not out.exists(), "Use a fresh PREPARED_DIR to avoid stale data"
    out.mkdir(parents=True, mode=0o700)
    records, proofs, seen = [], [], set()
    thinking_chars = 0
    roles = collections.Counter()
    for entry in selected:
        assert (
            entry["integrity_status"] == "accepted"
            and entry["source_sha256"] not in held
        )
        source = Path(entry["source"])
        assert data_gate.digest(source) == entry["source_sha256"]
        result = json.loads(source.read_text())
        raw = (
            (result.get("meta") or {}).get("benchmark", {}).get("eval_raw_data")
            or (result.get("extra") or {}).get("eval_raw_data")
            or {}
        )
        assert (
            result["metrics"]["correct"] is True
            and raw["completed"] is True
            and raw["resolved"] is True
        )
        statuses = {test["name"]: test["status"] for test in raw.get("tests", [])}
        required = raw.get("fail_to_pass", []) + raw.get("pass_to_pass", [])
        assert required and all(statuses.get(test) == "PASSED" for test in required)
        assert (
            json.loads((source.parent.parent / "task.json").read_text())["task_id"]
            == entry["task_id"]
        )
        messages = copy.deepcopy(
            result["artifacts"]["mini_swe_agent_raw_trajectory"]["messages"]
        )
        actual = 0
        for message in messages:
            if message["role"] == "assistant":
                if not message.get("reasoning_content"):
                    message["reasoning_content"] = "\n".join(
                        block.get("thinking", "")
                        for block in message.get("thinking_blocks", [])
                        if isinstance(block, dict)
                    )
                actual += len((message.get("reasoning_content") or "").strip())
        assert actual > 0
        thinking_chars += actual
        normalized = normalize({"trajectory": {"messages": messages}})
        originals = [message for message in messages if message["role"] != "exit"]
        assert [message["role"] for message in normalized] == [
            message["role"] for message in originals
        ]
        for old, new in zip(originals, normalized):
            assert new["content"] == (old.get("content") or "")
            assert new.get("reasoning_content") == old.get("reasoning_content")
            assert new.get("tool_call_id") == old.get("tool_call_id")
            assert len(new.get("tool_calls") or []) == len(old.get("tool_calls") or [])
            roles.update([new["role"]])
        sha = hashlib.sha256(
            json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        assert sha not in seen
        seen.add(sha)
        condition = collection[source.parts[-5]]
        key = {
            "task": entry["task_id"],
            "source": str(source),
            "source_sha256": entry["source_sha256"],
            "attempt": entry["attempt"],
            "language": benchmark[entry["task_id"]]["repo_language"],
            "group": "fresh_opus48_official_pass",
            "collection_kind": condition["kind"],
            "own_failed_patch_and_binary_feedback_supplied": condition[
                "own_patch_repair"
            ],
            "integrity_status": "accepted",
            "sha256": sha,
        }
        records.append((key, normalized))
        proofs.append(
            {
                "task": entry["task_id"],
                "source_sha256": entry["source_sha256"],
                "actual_thinking_chars": actual,
                "required_native_tests": len(required),
            }
        )
    training_tasks = {key["task"] for key, _ in records}
    validation = pipeline.load_validation(training_tasks, args.validation_tasks)
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["NEMOTRON_MODEL"], trust_remote_code=True, local_files_only=True
    )
    summary = {
        "max_length": pipeline.MAX_LENGTH,
        "native_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
        "splits": {
            "augmented": pipeline.write_split("augmented", records, tokenizer),
            "monitor": pipeline.write_split("monitor", validation, tokenizer),
        },
        "validation": {
            "tasks": args.validation_tasks,
            "training_overlap": 0,
            "type": "Independent task teacher loss; not agent pass rate",
            "reference_source": str(pipeline.REFERENCE),
            "reference_sha256": data_gate.digest(pipeline.REFERENCE),
        },
        "data_recheck": {
            "training_tasks": len(records),
            "thinking_chars": thinking_chars,
            "exact_normalized_duplicates": 0,
            "roles": dict(roles),
        },
        "policy": "One reviewed trajectory per task; native thinking/actions/EOS targets; "
        "rolling windows supervise each original target once; no task length exclusion.",
    }
    (out / "source_recheck.json").write_text(json.dumps(proofs, indent=2) + "\n")
    data_gate.publish_ready(out, summary, args.accepted_manifest)
    data_gate.verify(out, args.accepted_manifest, args.integrity_audit)
    print(json.dumps(summary["data_recheck"], indent=2))


if __name__ == "__main__":
    main()
