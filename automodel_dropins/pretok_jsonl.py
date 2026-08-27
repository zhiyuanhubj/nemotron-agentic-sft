"""Read rows that already carry `input_ids` and `labels`.

The point of pretokenising is that nothing downstream re-derives the mask, so
this class deliberately does no formatting, no template application, and no
role inspection -- it hands the arrays through untouched. Anything that needs
deciding was decided in scripts/pretokenize.py, where the segment boundaries
are asserted against the chat template.

Copy this file into the Automodel checkout:

    cp automodel_dropins/pretok_jsonl.py \\
      $AUTOMODEL/nemo_automodel/components/datasets/llm/pretok_jsonl.py
"""

import json

from torch.utils.data import Dataset


class PretokenizedJsonl(Dataset):
    def __init__(self, path_or_dataset_id: str, split: str | None = None,
                 max_len: int | None = None, **kw):
        self.max_len = max_len
        self.rows = []
        with open(path_or_dataset_id) as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"no rows in {path_or_dataset_id}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids, labels = r["input_ids"], r["labels"]
        if self.max_len is not None:
            ids, labels = ids[: self.max_len], labels[: self.max_len]
        # The recipe consumes pre-shifted pairs (ChatDataset's `unshifted` flag
        # defaults to False and it shifts before returning). Handing over the
        # aligned arrays instead had the model predicting token i from token i,
        # which is why loss sat at ~24 and barely moved between the 4K and 16K
        # probes despite a 9x difference in trained tokens.
        return {"input_ids": ids[:-1], "labels": labels[1:]}
