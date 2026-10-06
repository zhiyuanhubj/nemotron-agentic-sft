"""Real Gloo DP2/CP2 accounting check without model weights or GPU allocation."""

import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from token_budget import consume


def rank_test(rank, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4)
    dp_groups = [dist.new_group(x) for x in [[0, 2], [1, 3]]]
    cp_groups = [dist.new_group(x) for x in [[0, 1], [2, 3]]]
    group = dp_groups[rank % 2]
    cp = cp_groups[rank // 2]

    class Recipe:
        def __init__(self):
            self.step_scheduler = SimpleNamespace(step=0, max_steps=100, sigterm_flag=False)

        def _dp_allreduce(self, x):
            dist.all_reduce(x, group=group)
            return x

        def _get_dp_group(self):
            return group

        def _get_dp_group_size(self):
            return 2

    recipe = Recipe()
    counter = {"supervised_tokens": 0, "optimizer_steps": 0, "input_tokens": 0}

    def original(r, batches, max_grad_norm):
        local = torch.tensor(sum(int((x["labels"] != -100).sum()) for x in batches))
        dist.all_reduce(local, group=group)
        inputs = torch.tensor(sum(x["input_ids"].numel() for x in batches))
        dist.all_reduce(inputs, group=group)
        return SimpleNamespace(
            step=r.step_scheduler.step,
            metrics={
                "num_label_tokens": int(local),
                "num_tokens_per_step": int(inputs),
            },
        )

    for step in range(2):
        recipe.step_scheduler.step = step
        count = 3 if rank // 2 == 0 else 5
        storage = torch.full((2, 4), -100)
        storage[:, 1:] = torch.tensor([[0, 1, 2], [3, 4, 5]])
        storage[:, 1:][torch.tensor([[0, 1, 2], [3, 4, 5]]) >= count] = -100
        labels = storage[:, 1:]
        assert not labels.is_contiguous()
        inputs = torch.arange(6).reshape(2, 3)
        before_inputs = inputs.clone()
        before_labels = labels.clone()
        data = consume(recipe, [{"labels": labels, "input_ids": inputs}], original, 10, counter)
        assert torch.equal(inputs, before_inputs) and torch.equal(labels, before_labels), (
            "Input dataset tensors mutated"
        )
        assert data.metrics["num_label_tokens"] == (8 if step == 0 else 2), (
            "CP replicas double-counted or final budget exceeded"
        )
        assert data.metrics["budget/final_step_labels_clipped"] == int(step == 1)
        values = [None, None]
        dist.all_gather_object(values, data.metrics, group=cp)
        assert values[0] == values[1], "CP replicas consumed unequal budgets"
    assert counter == {
        "supervised_tokens": 10,
        "optimizer_steps": 2,
        "input_tokens": 24,
    }
    assert recipe.step_scheduler.max_steps == 2 and recipe.step_scheduler.sigterm_flag is False
    # Neither an extra step nor a silently zeroed optimizer update is allowed.
    try:
        consume(
            recipe,
            [{"labels": torch.tensor([[1]]), "input_ids": torch.tensor([[1]])}],
            original,
            10,
            counter,
        )
    except AssertionError as e:
        assert "after token budget" in str(e)
    else:
        raise AssertionError("Token budget allowed an extra optimizer update")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(rank_test, args=(port,), nprocs=4, join=True)
    result = {
        "status": "passed",
        "world_size": 4,
        "dp_size": 2,
        "cp_size": 2,
        "verified": [
            "CP replicas are not double counted",
            "final DP split clips exact remaining targets, including zero targets on one DP worker",
            "original input and labels remain immutable",
            "ordinary final validation/checkpoint signaled by max_steps, no SIGTERM",
            "extra optimizer update refused",
        ],
        "scope": "Distributed token accounting only; hook remains disabled until corpus budget and full training config are frozen.",
    }
    print(json.dumps(result))
