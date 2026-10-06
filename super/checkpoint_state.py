"""Persist optional hook state alongside synchronous Automodel checkpoints."""

import json
from pathlib import Path

WRITERS = {}


def register(name, snapshot):
    WRITERS[name] = snapshot


def install():
    import torch.distributed as dist
    from nemo_automodel.recipes.base_recipe import BaseRecipe

    original = BaseRecipe.save_checkpoint

    def save(recipe, epoch, step, *args, **kwargs):
        original(recipe, epoch, step, *args, **kwargs)
        if not recipe.checkpointer.config.enabled:
            return
        path = Path(recipe.checkpointer.config.checkpoint_dir) / f"epoch_{epoch}_step_{step}"

        def persist():
            for name, snapshot in WRITERS.items():
                temporary = path / f"{name}.tmp"
                temporary.write_text(json.dumps(snapshot(), indent=2) + "\n")
                temporary.replace(path / f"{name}.json")

        # Only the global main rank owns monitor state. The optional node-local
        # backup hook runs after this wrapper and collects this rank's metadata.
        error = None
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                persist()
            except Exception as exc:
                error = exc
        if dist.is_initialized():
            import torch

            device = torch.cuda.current_device() if dist.get_backend() == "nccl" else "cpu"
            failed = torch.tensor(int(error is not None), device=device)
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            if failed.item():
                raise RuntimeError("Unable to persist checkpoint hook state") from error
        elif error is not None:
            raise error

    BaseRecipe.save_checkpoint = save
