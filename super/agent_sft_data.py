"""Memory-mapped native-template tokens with exactly one causal label shift."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PretokenizedDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        self.ids = np.load(path / "input_ids.npy", mmap_mode="r")
        self.labels = np.load(path / "labels.npy", mmap_mode="r")
        self.offsets = np.load(path / "offsets.npy", mmap_mode="r")
        assert len(self.ids) == len(self.labels) == self.offsets[-1]

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        start, end = self.offsets[i : i + 2]
        return {
            "input_ids": torch.from_numpy(self.ids[start:end].astype(np.int64)),
            "labels": torch.from_numpy(self.labels[start:end].astype(np.int64)),
        }


@dataclass(frozen=True)
class PretokenizedDatasetConfig:
    path: str

    def build(self):
        return PretokenizedDataset(self.path)


def collate_fn(examples, processor=None, max_length=131072):
    length = max(len(x["input_ids"]) for x in examples)
    assert length <= max_length, "Preprocessing must reject overlength trajectories"
    # The recipe pads to the context-parallel multiple after this collator.
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = pad_id if pad_id is not None else 0
    ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, ex in enumerate(examples):
        n = len(ex["input_ids"])
        ids[i, :n] = ex["input_ids"]
        labels[i, :n] = ex["labels"]
        attention[i, :n] = 1
    return {
        "input_ids": ids[:, :-1],
        "labels": labels[:, 1:],
        "attention_mask": attention[:, :-1],
    }
