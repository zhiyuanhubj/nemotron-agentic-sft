"""Run-scoped local NVMe checkpointing with verified atomic FSx publication.

Every node retains the union of the latest two and best three local shard sets.
FSx retains one best complete global checkpoint; its predecessor is deleted only
after a newly better set has been verified.
An incomplete copy is never advertised as resumable. No Hub uploads are made.
"""

import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

RUN = Path(os.environ["RUN_DIR"]).resolve()
LOCAL = Path(os.environ["LOCAL_CKPT_DIR"]).resolve()
BACKUP = RUN / "checkpoints"


def collective_step(fn, enabled=True):
    error = None
    if enabled:
        try:
            fn()
        except Exception as exc:
            error = exc
            print(f"CHECKPOINT_FAILURE rank={dist.get_rank()}: {exc!r}", flush=True)
    device = torch.cuda.current_device() if dist.get_backend() == "nccl" else "cpu"
    failed = torch.tensor(int(error is not None), device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise RuntimeError(
            "Checkpoint backup failed; old complete backup and local shards retained"
        ) from error


def checkpoint_order(path):
    match = re.fullmatch(r"epoch_(\d+)_step_(\d+)", path.name)
    return tuple(map(int, match.groups())) if match else (-1, -1)


def validation_score(path):
    for filename in ("score.json", "losses.json"):
        file = path / filename
        if file.is_file():
            score = json.loads(file.read_text()).get("val_loss")
            if score is not None and math.isfinite(float(score)):
                return float(score)
    return None


def retention_selection(root):
    candidates = sorted(
        [
            p
            for p in root.glob("epoch_*_step_*")
            if p.is_dir() and not (p / ".incomplete").exists()
        ],
        key=checkpoint_order,
    )
    best = sorted(
        [p for p in candidates if validation_score(p) is not None],
        key=lambda p: (validation_score(p), checkpoint_order(p)),
    )[:3]
    keep = set(candidates[-2:]) | set(best)
    return candidates, best, keep


def prune_local():
    candidates, best, keep = retention_selection(LOCAL)
    for old in candidates:
        if old not in keep:
            shutil.rmtree(old)
    for index in range(3):
        pointer = LOCAL / f"BEST_{index + 1}"
        pointer.unlink(missing_ok=True)
        if index < len(best):
            pointer.symlink_to(best[index].name)
    pointer = LOCAL / "LOWEST_VAL"
    pointer.unlink(missing_ok=True)
    if best:
        pointer.symlink_to(best[0].name)
    (LOCAL / "RETENTION.json").write_text(
        json.dumps(
            {
                "latest": [p.name for p in candidates[-2:]],
                "best": [p.name for p in best],
                "retained": sorted(p.name for p in keep),
                "max_union": 5,
            },
            indent=2,
        )
    )


def copy_node(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude=.incomplete",
            str(source) + "/",
            str(destination) + "/",
        ],
        check=True,
    )
    manifest = {}
    for file in source.rglob("*"):
        if not file.is_file() or file.name == ".incomplete":
            continue
        rel = str(file.relative_to(source))
        copied = destination / rel
        assert copied.is_file() and copied.stat().st_size == file.stat().st_size, rel
        # rsync verifies transferred payloads. Hash small metadata independently.
        info = {"bytes": file.stat().st_size}
        if info["bytes"] < 16 * 1024 * 1024:
            info["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            assert info["sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest(), (
                rel
            )
        manifest[rel] = info
    (destination / "NODE_MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True)
    )


def assemble_and_verify(incoming, name, world_size, nodes):
    assembled = incoming / "assembled"
    assembled.mkdir()
    manifests = {}
    for node in range(nodes):
        node_dir = incoming / f"node_{node}"
        manifests[str(node)] = json.loads((node_dir / "NODE_MANIFEST.json").read_text())
        for rel, info in manifests[str(node)].items():
            src, dst = node_dir / rel, assembled / rel
            assert src.stat().st_size == info["bytes"], rel
            if dst.exists():
                assert dst.stat().st_size == src.stat().st_size, (
                    f"Conflicting node metadata: {rel}"
                )
                if info.get("sha256"):
                    assert (
                        hashlib.sha256(dst.read_bytes()).hexdigest() == info["sha256"]
                    ), rel
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
    from safetensors import safe_open

    shards = list((assembled / "model").glob("*.safetensors"))
    assert len(shards) >= world_size, (
        f"Missing model shards: {len(shards)}/{world_size}"
    )
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            assert list(f.keys()), shard.name
    optim_dir = assembled / "optim"
    with (optim_dir / ".metadata").open("rb") as f:
        metadata = pickle.load(f)  # Locally generated PyTorch DCP metadata only.
    for storage in metadata.storage_data.values():
        file = optim_dir / storage.relative_path
        assert (
            file.is_file() and file.stat().st_size >= storage.offset + storage.length
        ), file
    assert len(list(optim_dir.glob("*.distcp"))) >= world_size
    for component in ("rng", "dataloader"):
        files = list((assembled / component).glob("*.pt"))
        assert len(files) == world_size, (
            f"Missing {component}: {len(files)}/{world_size}"
        )
    for required in ("config.yaml", "losses.json", "step_scheduler.pt"):
        assert (assembled / required).is_file(), required
    complete = {
        "name": name,
        "world_size": world_size,
        "nodes": nodes,
        "verified_at": time.time(),
        "node_manifests": manifests,
    }
    (assembled / "COMPLETE.json").write_text(json.dumps(complete, sort_keys=True))
    target = BACKUP / name
    assert not target.exists()
    assembled.rename(target)
    pointer = BACKUP / "LATEST.tmp"
    pointer.unlink(missing_ok=True)
    pointer.symlink_to(name)
    os.replace(pointer, BACKUP / "LATEST")
    best_pointer = BACKUP / "BEST.tmp"
    best_pointer.unlink(missing_ok=True)
    best_pointer.symlink_to(name)
    os.replace(best_pointer, BACKUP / "BEST")
    shutil.rmtree(incoming)
    # Before retiring any predecessor, recheck this complete new full-state
    # backup and all eight nodes' retained original checkpoint shard sets.
    from verify_checkpoint import verify_complete

    verify_complete(target)
    # Only this run's previously verified backups are eligible for deletion.
    for previous in BACKUP.glob("epoch_*_step_*"):
        if (
            previous != target
            and previous.is_dir()
            and (previous / "COMPLETE.json").is_file()
        ):
            shutil.rmtree(previous)
    print(
        f"CHECKPOINT_BACKUP_COMPLETE path={target} model_shards={len(shards)} world_size={world_size} "
        f"val_loss={validation_score(target)} fsx_policy=best1",
        flush=True,
    )


def backup_checkpoint(path):
    rank, world = dist.get_rank(), dist.get_world_size()
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    node = rank // local_world
    coordinator = int(os.environ["LOCAL_RANK"]) == 0
    incoming = BACKUP / (path.name + ".partial")

    scores = [validation_score(path) if rank == 0 else None]
    dist.broadcast_object_list(scores, src=0)
    score = scores[0]

    def record_scores():
        # Bootstrap old local checkpoints made before this policy. Their full
        # FSx backup contains rank-zero losses even on nodes without losses.json.
        for previous in LOCAL.glob("epoch_*_step_*"):
            if not previous.is_dir() or (previous / "score.json").exists():
                continue
            previous_score = validation_score(previous)
            if previous_score is None:
                previous_score = validation_score(BACKUP / previous.name)
            if previous_score is not None:
                (previous / "score.json").write_text(
                    json.dumps({"val_loss": previous_score})
                )
        (path / "score.json").write_text(json.dumps({"val_loss": score}))

    collective_step(record_scores, coordinator)
    should_copy = [None]
    if rank == 0:
        pointer = BACKUP / "LATEST"
        old = pointer.resolve() if pointer.exists() else None
        old_score = validation_score(old) if old is not None else None
        should_copy[0] = old is None or (
            score is not None and (old_score is None or score < old_score)
        )
        if not should_copy[0]:
            print(
                f"CHECKPOINT_NVME_SAVED path={path} val_loss={score}; FSX_BEST_UNCHANGED "
                f"path={old} val_loss={old_score}",
                flush=True,
            )
    dist.broadcast_object_list(should_copy, src=0)
    if not should_copy[0]:
        collective_step(prune_local, coordinator)
        return

    def initialize():
        BACKUP.mkdir(exist_ok=True)
        # Conservative preflight based on the actual total sizes reported below.
        if incoming.exists():
            shutil.rmtree(incoming)
        incoming.mkdir()

    collective_step(initialize, rank == 0)
    device = torch.cuda.current_device() if dist.get_backend() == "nccl" else "cpu"
    local_bytes = (
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if coordinator
        else 0
    )
    total = torch.tensor(local_bytes, device=device, dtype=torch.int64)
    dist.all_reduce(total)

    def check_space():
        assert shutil.disk_usage(BACKUP).free > total.item() * 1.05 + 20 * 1024**3, (
            "Insufficient FSx space for atomic backup"
        )

    collective_step(check_space, rank == 0)
    collective_step(lambda: copy_node(path, incoming / f"node_{node}"), coordinator)
    collective_step(
        lambda: assemble_and_verify(incoming, path.name, world, world // local_world),
        rank == 0,
    )

    collective_step(prune_local, coordinator)


def install():
    from nemo_automodel.components.checkpoint import checkpointing
    from nemo_automodel.components.checkpoint.lifecycle import CheckpointLifecycle
    from nemo_automodel.recipes.base_recipe import BaseRecipe

    original_dirs = checkpointing._ensure_shared_dirs
    original_coordinator = CheckpointLifecycle._is_coordinator
    original_prune = CheckpointLifecycle._prune_old_checkpoints
    original_best = CheckpointLifecycle._update_best_checkpoint
    original_save = BaseRecipe.save_checkpoint

    def local_config(lifecycle):
        return Path(lifecycle.config.checkpoint_dir) == LOCAL

    def ensure_dirs(*dirs, process_group=None):
        if all(d is None or Path(d).is_relative_to(LOCAL) for d in dirs):
            return checkpointing._ensure_dirs(*dirs, process_group=process_group)
        return original_dirs(*dirs, process_group=process_group)

    def is_coordinator(lifecycle):
        if local_config(lifecycle):
            return int(os.environ.get("LOCAL_RANK", "0")) == 0
        return original_coordinator(lifecycle)

    def prune(lifecycle):
        if not local_config(lifecycle):
            return original_prune(lifecycle)
        # Defer local retention until the complete global FSx backup is verified.

    def update_best(lifecycle, *args, **kwargs):
        if not local_config(lifecycle):
            return original_best(lifecycle, *args, **kwargs)
        # The run-specific policy maintains three best checkpoints on every node.

    def save(recipe, epoch, step, *args, **kwargs):
        original_save(recipe, epoch, step, *args, **kwargs)
        if recipe.checkpointer.config.enabled and local_config(
            recipe.checkpointer.lifecycle
        ):
            assert not recipe.checkpointer.config.is_async
            backup_checkpoint(LOCAL / f"epoch_{epoch}_step_{step}")

    checkpointing._ensure_shared_dirs = ensure_dirs
    CheckpointLifecycle._is_coordinator = is_coordinator
    CheckpointLifecycle._prune_old_checkpoints = prune
    CheckpointLifecycle._update_best_checkpoint = update_best
    BaseRecipe.save_checkpoint = save
