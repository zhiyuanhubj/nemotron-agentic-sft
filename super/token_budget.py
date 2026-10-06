"""Exact global assistant-token budget; preserve CP replication and final save.

This hook is dormant unless token_budget.enabled is set in the frozen config.
It counts actual optimizer inputs, including sampler padding or repeated rows.
"""

from pathlib import Path
from datetime import datetime, timezone
import json, os


def trim_labels(batches, quota):
    """Keep the earliest quota nonignored labels, without changing inputs."""
    import torch

    remaining = int(quota)
    out = []
    for batch in batches:
        labels = batch["labels"]
        valid = torch.nonzero(labels.reshape(-1) != -100, as_tuple=False).reshape(-1)
        take = min(remaining, len(valid))
        remaining -= take
        if take < len(valid):
            batch = dict(batch)
            cloned = labels.clone(memory_format=torch.contiguous_format)
            flat = cloned.reshape(-1)
            flat[valid[take:]] = -100
            batch["labels"] = cloned
        out.append(batch)
    assert remaining == 0, "Local target quota exceeds available labels"
    return out


def consume(recipe, batches, original, limit, counter, max_grad_norm=None):
    import torch
    import torch.distributed as dist

    local = sum(int((b["labels"] != -100).sum().item()) for b in batches)
    global_count = int(
        recipe._dp_allreduce(torch.tensor(local, dtype=torch.long)).item()
    )
    remaining = int(limit) - counter["supervised_tokens"]
    assert remaining > 0, "Optimizer called after token budget completed"
    clipped = global_count > remaining
    if clipped:
        group = recipe._get_dp_group()
        size = recipe._get_dp_group_size()
        if size == 1:
            counts = [local]
            rank = 0
        else:
            device = (
                torch.cuda.current_device()
                if dist.get_backend(group) == "nccl"
                else "cpu"
            )
            value = torch.tensor([local], dtype=torch.long, device=device)
            values = [torch.zeros_like(value) for _ in range(size)]
            dist.all_gather(values, value, group=group)
            counts = [int(x.item()) for x in values]
            rank = dist.get_rank(group)
        assert sum(counts) == global_count
        quota = max(0, min(local, remaining - sum(counts[:rank])))
        batches = trim_labels(batches, quota)
    data = original(recipe, batches, max_grad_norm)
    actual = int(data.metrics["num_label_tokens"])
    assert actual == min(global_count, remaining)
    counter["supervised_tokens"] += actual
    counter["optimizer_steps"] += 1
    counter["input_tokens"] += int(data.metrics["num_tokens_per_step"])
    data.metrics["budget/supervised_tokens_total"] = counter["supervised_tokens"]
    data.metrics["budget/input_tokens_total"] = counter["input_tokens"]
    data.metrics["budget/limit"] = limit
    data.metrics["budget/final_step_labels_clipped"] = int(clipped)
    if counter["supervised_tokens"] == limit:
        # Setting max_steps makes the existing scheduler validate and save this
        # final step normally. It does not signal SIGTERM or skip validation.
        recipe.step_scheduler.max_steps = min(
            recipe.step_scheduler.max_steps, recipe.step_scheduler.step + 1
        )
    return data


def install(config):
    options = config.get("token_budget", {})
    if not options.get("enabled"):
        return
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM
    import nvme_checkpoint as nv

    mode = options.get("mode", "exact")
    assert mode in {"exact", "track_only"}
    limit = (1 << 60) if mode == "track_only" else int(options["supervised_tokens"])
    assert limit > 0
    counter = {"supervised_tokens": 0, "optimizer_steps": 0, "input_tokens": 0}
    restore = config.get("checkpoint", {}).get("restore_from")
    if restore:
        saved = json.loads((Path(restore) / "token_budget.json").read_text())
        assert saved["limit"] == limit
        counter.update(saved["counter"])
    original = FinetuneRecipeForVLM._run_train_optim_step
    state = Path(os.environ["RUN_DIR"]) / "state/token_budget.json"
    state.parent.mkdir(exist_ok=True)

    def step(recipe, batches, max_grad_norm=None):
        data = consume(recipe, batches, original, limit, counter, max_grad_norm)
        if recipe.dist_env.is_main:
            record = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "limit": limit,
                "counter": counter,
                "reached": counter["supervised_tokens"] == limit,
                "step": data.step,
                "early_stopped_before_budget": bool(recipe.step_scheduler.sigterm_flag),
            }
            tmp = state.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2) + "\n")
            tmp.replace(state)
        return data

    prior_backup = nv.backup_checkpoint

    def backup(path):
        def persist():
            assert state.is_file()
            (path / "token_budget.json").write_text(state.read_text())

        nv.collective_step(persist, int(os.environ.get("LOCAL_RANK", "0")) == 0)
        return prior_backup(path)

    FinetuneRecipeForVLM._run_train_optim_step = step
    nv.backup_checkpoint = backup
