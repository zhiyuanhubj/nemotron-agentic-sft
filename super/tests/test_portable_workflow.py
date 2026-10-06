"""Exercise the public JSONL -> prepared arrays -> configurable launcher workflow."""

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import data_gate
import prepare_data
import run_training
from agent_sft_data import PretokenizedDataset, collate_fn
from prepare_native import normalize
from topology import validate_topology


class CharacterTokenizer:
    """Synthetic template for CPU workflow tests; real-template coverage is separate."""

    is_fast = True

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = ""
        for message in messages:
            body = message.get("content", "")
            if message["role"] == "assistant":
                reasoning = message.get("reasoning_content", "")
                body = f"<think>\n{reasoning}</think>" + body
                if message.get("tool_calls"):
                    body += json.dumps(message["tool_calls"])
            text += f"<|im_start|>{message['role']}\n{body}<|im_end|>"
        return [ord(char) for char in text] if tokenize else text

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class PortableWorkflowTests(unittest.TestCase):
    def run_cli(self, script, *args, success=True):
        result = subprocess.run(
            [sys.executable, str(CODE / script), *map(str, args)],
            text=True,
            capture_output=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_public_workflow_and_multiple_topologies(self):
        with tempfile.TemporaryDirectory(prefix="sft test # ") as temporary:
            root = Path(temporary)
            model, prepared = root / "model", root / "prepared"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"text_config": {"n_routed_experts": 8}}))
            dataset = root / "messages.jsonl.gz"
            rows = [
                {
                    "task_id": f"task-{i}",
                    "messages": [
                        {"role": "user", "content": f"Question {i}"},
                        {"role": "assistant", "content": f"Answer {i}" + "x" * 600},
                    ],
                }
                for i in range(3)
            ]
            with gzip.open(dataset, "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            fake_transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(
                    from_pretrained=lambda *args, **kwargs: CharacterTokenizer()
                )
            )
            with (
                patch.dict(sys.modules, {"transformers": fake_transformers}),
                patch.object(
                    sys,
                    "argv",
                    [
                        "prepare_data.py",
                        "--train",
                        str(dataset),
                        "--model",
                        str(model),
                        "--output",
                        str(prepared),
                        "--max-length",
                        "256",
                    ],
                ),
            ):
                prepare_data.main()
            data_gate.verify(prepared)
            summary = json.loads((prepared / "summary.json").read_text())
            self.assertEqual(summary["splits"]["train"]["tasks"], 2)
            self.assertEqual(summary["splits"]["validation"]["tasks"], 1)
            dataset = PretokenizedDataset(prepared / "train")
            self.assertGreater(len(dataset), 2)
            # Targets survive both rolling-window segmentation and the causal shift.
            observed = sum(
                int((collate_fn([dataset[i]])["labels"] != -100).sum()) for i in range(len(dataset))
            )
            self.assertEqual(observed, summary["splits"]["train"]["target_tokens"])
            for nodes, workers, cp, ep in [(1, 1, 1, 1), (1, 4, 1, 1), (2, 2, 2, 4)]:
                run = root / f"run-{nodes}-{workers}"
                self.run_cli(
                    "configure.py",
                    "--model",
                    model,
                    "--prepared-dir",
                    prepared,
                    "--run-dir",
                    run,
                    "--nnodes",
                    nodes,
                    "--nproc-per-node",
                    workers,
                    "--cp-size",
                    cp,
                    "--ep-size",
                    ep,
                )
                config = yaml.safe_load((run / "train.yaml").read_text())
                self.assertEqual(config["dataset"]["path"], str(prepared / "train"))
                self.assertEqual(config["checkpoint"]["checkpoint_dir"], str(run / "checkpoints"))
                self.assertEqual(config["dataloader"]["collate_fn"]["max_length"], 256)
                self.assertNotIn("wandb", config)
                # Exercise the runtime preflight without loading a GPU model.
                calls = []
                fake_hooks = {
                    name: SimpleNamespace(install=lambda *args, name=name: calls.append(name))
                    for name in ["checkpoint_state", "early_stopping", "token_budget"]
                }
                fake_hooks["nemo_automodel.cli.app"] = SimpleNamespace(main=lambda: "train")
                with (
                    patch.dict(sys.modules, fake_hooks),
                    patch.dict(
                        "os.environ",
                        {
                            "WORLD_SIZE": str(nodes * workers),
                            "LOCAL_WORLD_SIZE": str(workers),
                            "RANK": str((nodes - 1) * workers),
                            "LOCAL_RANK": "0",
                        },
                    ),
                    patch.object(sys, "argv", ["run_training.py", str(run / "train.yaml")]),
                ):
                    self.assertEqual(run_training.main(), "train")
                self.assertEqual(calls, ["checkpoint_state", "early_stopping", "token_budget"])
                launch = self.run_cli(
                    "launch.py",
                    "--run-dir",
                    run,
                    "--dry-run",
                    "--node-rank",
                    nodes - 1,
                    "--master-addr",
                    "host-0",
                )
                self.assertIn(f"--nnodes={nodes}", launch.stdout)
                self.assertIn(f"--nproc-per-node={workers}", launch.stdout)
                self.assertEqual("--standalone" in launch.stdout, nodes == 1)
                if nodes > 1:
                    self.run_cli("launch.py", "--run-dir", run, "--dry-run", success=False)
                # Editing a generated config must not silently bypass validation.
                (run / "train.yaml").write_text((run / "train.yaml").read_text() + "\n")
                self.run_cli("launch.py", "--run-dir", run, "--dry-run", success=False)
            # An identical tokenizer at a new path is valid for portable prepared data.
            relocated = root / "relocated-model"
            relocated.mkdir()
            (relocated / "config.json").write_bytes((model / "config.json").read_bytes())
            self.run_cli(
                "configure.py",
                "--model",
                relocated,
                "--prepared-dir",
                prepared,
                "--run-dir",
                root / "relocated-run",
            )
            (relocated / "tokenizer_config.json").write_text('{"chat_template": "changed"}')
            result = self.run_cli(
                "configure.py",
                "--model",
                relocated,
                "--prepared-dir",
                prepared,
                "--run-dir",
                root / "mismatched-run",
                success=False,
            )
            self.assertIn("tokenization assets differ", result.stderr)

    def test_split_groups_repeated_tasks_and_rejects_leakage(self):
        records = [
            ({"task": task, "sha256": str(i)}, [])
            for i, task in enumerate(["a", "a", "b", "c", "c"])
        ]
        train, validation = prepare_data.split_records(records, 0.3, 42)
        self.assertEqual((train, validation), prepare_data.split_records(records, 0.3, 42))
        prepare_data.check_disjoint(train, validation)
        with self.assertRaises(ValueError):
            prepare_data.check_disjoint(train, train[:1])
        with self.assertRaises(ValueError):
            prepare_data.check_disjoint(
                [({"task": "a", "sha256": "same"}, [])], [({"task": "b", "sha256": "same"}, [])]
            )

    def test_plain_input_duplicate_and_tool_argument_validation(self):
        examples = CODE.parent / "examples"
        train = prepare_data.load_records(examples / "train.jsonl")
        validation = prepare_data.load_records(examples / "validation.jsonl")
        prepare_data.check_disjoint(train, validation)
        row = {
            "messages": [
                {"role": "user", "content": "Run a command"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "ls"}',
                            }
                        }
                    ],
                },
            ]
        }
        self.assertIsInstance(normalize(row)[1]["tool_calls"][0]["function"]["arguments"], dict)
        # Normalization must not mutate the caller's original records.
        self.assertIsInstance(row["messages"][1]["tool_calls"][0]["function"]["arguments"], str)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            path.write_text(json.dumps({"task_id": "a", **row}) + "\n")
            prepare_data.load_records(path)
            path.write_text(path.read_text() * 2)
            with self.assertRaisesRegex(ValueError, "duplicate conversation"):
                prepare_data.load_records(path)

    def test_invalid_parallelism_and_batch_sizes(self):
        config = yaml.safe_load((CODE / "configs/full_sft.yaml").read_text())
        config["distributed"]["dp_size"] = None
        for cp, ep, global_batch in [(3, 1, 8), (1, 3, 8), (1, 1, 3), (0, 1, 8)]:
            config["distributed"].update(cp_size=cp, ep_size=ep)
            config["step_scheduler"]["global_batch_size"] = global_batch
            with self.assertRaises(ValueError):
                validate_topology(config, 4)


if __name__ == "__main__":
    unittest.main()
