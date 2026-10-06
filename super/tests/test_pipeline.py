"""Exercise production token boundaries, rolling targets and data freeze gates."""

import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("NEMOTRON_MODEL", "unused-by-synthetic-tests")
os.environ.setdefault("REFERENCE_JSONL", "unused")
os.environ.setdefault("BENCHMARK_JSONL", "unused")
import data_gate
from agent_sft_data import PretokenizedDataset, collate_fn
from early_stopping import OverfitMonitor
from prepare_v41 import windows
from verify_checkpoint import verify_complete


class PipelineTests(unittest.TestCase):
    def test_rolling_targets_once_with_oversized_round(self):
        import prepare_v41 as pipeline

        original = pipeline.MAX_LENGTH
        pipeline.MAX_LENGTH = 24
        try:
            ids = np.arange(95, dtype=np.int32)
            labels = ids.copy()
            labels[:5] = -100
            labels[::7] = -100
            parts = windows(ids, labels, [5, 12, 21, 70, 90])
            observed = np.concatenate([lab[lab != -100] for _, lab, _ in parts])
            np.testing.assert_array_equal(observed, labels[labels != -100])
            self.assertTrue(all(len(tokens) <= 24 for tokens, _, _ in parts))
            self.assertTrue(
                any(layout["partial_round_boundary"] for _, _, layout in parts)
            )
            for tokens, _, _ in parts:
                np.testing.assert_array_equal(tokens[:5], ids[:5])
        finally:
            pipeline.MAX_LENGTH = original

    def test_collation_shifts_once_and_masks_padding(self):
        examples = [
            {
                "input_ids": torch.tensor([10, 11, 12, 13]),
                "labels": torch.tensor([-100, -100, 12, 13]),
            },
            {
                "input_ids": torch.tensor([20, 21, 22]),
                "labels": torch.tensor([-100, 21, 22]),
            },
        ]
        batch = collate_fn(examples, SimpleNamespace(pad_token_id=99))
        self.assertEqual(batch["input_ids"].tolist(), [[10, 11, 12], [20, 21, 22]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 12, 13], [21, 22, -100]])

    def test_native_template_targets_and_tool_arguments(self):
        model = os.environ.get("TEST_MODEL")
        if not model:
            self.skipTest("Set TEST_MODEL to the local Nemotron tokenizer")
        from transformers import AutoTokenizer
        from prepare_native import normalize, tokenize

        tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True, local_files_only=True
        )
        messages = normalize(
            {
                "trajectory": {
                    "messages": [
                        {"role": "system", "content": "SYSTEM_SENTINEL"},
                        {"role": "user", "content": "USER_SENTINEL"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Inspect the repository first.",
                            "tool_calls": [
                                {
                                    "id": "call1",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command":"ls"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call1",
                            "content": "TOOL_SENTINEL",
                        },
                        {
                            "role": "assistant",
                            "content": "Done.",
                            "reasoning_content": "",
                        },
                    ]
                }
            }
        )
        self.assertIsInstance(
            messages[2]["tool_calls"][0]["function"]["arguments"], dict
        )
        ids, labels = tokenize(messages, tokenizer)
        targets = tokenizer.decode(ids[labels != -100].tolist())
        for sentinel in [
            "SYSTEM_SENTINEL",
            "USER_SENTINEL",
            "TOOL_SENTINEL",
            "<|im_start|>",
            "<think>",
        ]:
            self.assertNotIn(sentinel, targets)
        for expected in ["Inspect", "Done.", "ls", "<|im_end|>"]:
            self.assertIn(expected, targets)

    def test_freeze_gate_detects_array_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = {}
            for split, task in [("train", "a"), ("validation", "b")]:
                directory = root / split
                directory.mkdir()
                np.save(
                    directory / "input_ids.npy", np.array([1, 2, 3], dtype=np.uint32)
                )
                np.save(
                    directory / "labels.npy", np.array([-100, 2, 3], dtype=np.int32)
                )
                np.save(directory / "offsets.npy", np.array([0, 3], dtype=np.int64))
                (directory / "manifest.jsonl").write_text(
                    json.dumps({"task": task, "tokens": 3, "target_tokens": 2}) + "\n"
                )
                splits[split] = {"rows": 1, "tokens": 3, "target_tokens": 2}
            data_gate.publish_ready(root, {"max_length": 128, "splits": splits})
            data_gate.verify(root)
            dataset = PretokenizedDataset(root / "train")
            self.assertEqual(dataset[0]["input_ids"].tolist(), [1, 2, 3])
            with (root / "train/input_ids.npy").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(AssertionError):
                data_gate.verify(root)

    def test_checkpoint_rejects_missing_rank_state(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "epoch_0_step_2"
            for component in ["model", "optim", "rng", "dataloader"]:
                (path / component).mkdir(parents=True, exist_ok=True)
            for rank in range(64):
                save_file(
                    {"weight": torch.tensor([rank], dtype=torch.float32)},
                    path / "model" / f"{rank}.safetensors",
                )
                for component, suffix in [
                    ("optim", "distcp"),
                    ("rng", "pt"),
                    ("dataloader", "pt"),
                ]:
                    (path / component / f"{rank}.{suffix}").write_bytes(b"test")
            metadata = SimpleNamespace(
                storage_data={
                    "key": SimpleNamespace(relative_path="0.distcp", offset=0, length=4)
                }
            )
            (path / "optim/.metadata").write_bytes(pickle.dumps(metadata))
            for name in ["config.yaml", "losses.json", "step_scheduler.pt"]:
                (path / name).write_bytes(b"test")
            manifest = {
                str(file.relative_to(path)): {
                    "bytes": file.stat().st_size,
                    "sha256": data_gate.digest(file),
                }
                for file in path.rglob("*")
                if file.is_file()
            }
            (path / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "world_size": 64,
                        "nodes": 8,
                        "node_manifests": {str(i): manifest for i in range(8)},
                    }
                )
            )
            verify_complete(path)
            (path / "rng/63.pt").unlink()
            with self.assertRaises((AssertionError, FileNotFoundError)):
                verify_complete(path)

    def test_early_stop_requires_sustained_overfit(self):
        monitor = OverfitMonitor(min_steps=2, consecutive_evals=2)
        monitor.observe_training(1.0)
        self.assertFalse(monitor.observe_validation(0, 1.0)["stop"])
        for _ in range(10):
            monitor.observe_training(0.8)
        self.assertFalse(monitor.observe_validation(1, 1.2)["stop"])
        self.assertTrue(monitor.observe_validation(2, 1.2)["stop"])


if __name__ == "__main__":
    unittest.main()
