"""Generate a resolved training configuration for the user's hardware and paths."""

import argparse
import json
import os
import re
from pathlib import Path
from string import Template

import data_gate
import yaml
from topology import validate_topology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/full_sft.yaml"
    )
    parser.add_argument("--run-name", default="sft")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--cp-size", type=int)
    parser.add_argument("--ep-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--local-batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoint-dir", type=Path, help="Defaults to RUN_DIR/checkpoints")
    parser.add_argument(
        "--nvme-root", type=Path, help="Optional node-local staging and checkpoint root"
    )
    parser.add_argument("--restore-from", type=Path, help="Full-state checkpoint from the same run")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name must contain only letters, digits, underscores or hyphens")
    if not args.model.is_dir():
        parser.error("--model must be a local model directory")
    if args.nnodes < 1 or args.nproc_per_node < 1:
        parser.error("Node and worker counts must be positive")
    run, prepared, model = args.run_dir.resolve(), args.prepared_dir.resolve(), args.model.resolve()
    if (run / "train.yaml").exists():
        parser.error("Archive the existing train.yaml before generating a new configuration")
    if not args.restore_from and (run / "state/READY").exists():
        parser.error("An existing run requires --restore-from; use a fresh run directory otherwise")
    ready = data_gate.verify(prepared)
    summary = json.loads((prepared / "summary.json").read_text())
    fingerprint = data_gate.tokenizer_fingerprint(model)
    if not fingerprint or summary.get("tokenizer_sha256") != fingerprint:
        parser.error("Model tokenization assets differ from those used to prepare the dataset")
    staging = args.nvme_root.resolve() if args.nvme_root else None
    checkpoint_dir = (args.checkpoint_dir or run / "checkpoints").resolve()
    local_data = staging / "data" / args.run_name if staging else prepared
    values = {
        "LOCAL_MODEL": str(staging / "model" if staging else model),
        "LOCAL_DATA": str(local_data),
        "LOCAL_CKPT_DIR": str(
            staging / "checkpoints" / args.run_name if staging else checkpoint_dir
        ),
        "LOCAL_WANDB": str(run / "wandb"),
        "RUN_NAME": args.run_name,
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
    }

    # Load YAML first so paths containing ': ' or '#' remain ordinary strings.
    def resolve(value):
        if isinstance(value, str):
            return Template(value).substitute(values)
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    config = resolve(yaml.safe_load(args.config.read_text()))
    config["distributed"]["dp_size"] = None
    for name in ["cp_size", "ep_size"]:
        if getattr(args, name) is not None:
            config["distributed"][name] = getattr(args, name)
    if config["distributed"]["ep_size"] == 1:
        config["distributed"].pop("moe", None)
    else:
        config["distributed"].setdefault(
            "moe",
            {
                "reshard_after_forward": True,
                "wrap_outer_model": True,
            },
        )
    for name in ["global_batch_size", "local_batch_size"]:
        if getattr(args, name) is not None:
            config["step_scheduler"][name] = getattr(args, name)
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        config["step_scheduler"]["num_epochs"] = args.epochs
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            parser.error("--learning-rate must be positive")
        config["optimizer"]["lr"] = args.learning_rate
    max_length = args.max_length if args.max_length is not None else summary["max_length"]
    if max_length < summary["max_length"]:
        parser.error(
            "Reprocess the data to reduce --max-length; training does not truncate targets"
        )
    for name in ["dataloader", "validation_dataloader"]:
        config[name]["collate_fn"]["max_length"] = max_length
    if not args.wandb:
        config.pop("wandb", None)
    if config["checkpoint"].get("is_async"):
        parser.error("Training-state hooks require synchronous checkpoint saving")
    config["run_dir"] = str(run)
    config["checkpoint_storage"] = "node_local" if staging else "shared"
    topology = {"nnodes": args.nnodes, "nproc_per_node": args.nproc_per_node}
    parallelism = {name: config["distributed"][name] for name in ["cp_size", "ep_size"]}
    resume_settings = {
        "parallelism": parallelism,
        "global_batch_size": config["step_scheduler"]["global_batch_size"],
        "local_batch_size": config["step_scheduler"]["local_batch_size"],
        "max_length": max_length,
        "checkpoint_storage": config["checkpoint_storage"],
        "early_stopping": config.get("early_stopping"),
        "token_budget": config.get("token_budget"),
    }
    config["launch_topology"] = topology
    try:
        validate_topology(config, args.nnodes * args.nproc_per_node, model)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.restore_from and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        parser.error("Checkpoint directory is not empty; use --restore-from or a fresh directory")
    if args.restore_from:
        restore = args.restore_from.resolve()
        if (
            restore.parent != checkpoint_dir
            or not restore.is_dir()
            or (restore / ".incomplete").exists()
        ):
            parser.error(
                "--restore-from must be a completed checkpoint in this run checkpoint directory"
            )
        previous = json.loads((run / "state/READY").read_text())
        if previous["topology"] != topology or previous["ready"] != ready:
            parser.error("Resume requires the original topology and prepared data")
        if previous["parallelism"] != parallelism:
            parser.error("Resume requires the original context and expert parallelism")
        if previous["resume_settings"] != resume_settings:
            parser.error("Resume requires unchanged batch, sequence, storage and hook settings")
        if previous["model_source"] != str(model):
            parser.error("Resume requires the original base model directory")
        if staging:
            from verify_checkpoint import verify_complete

            verify_complete(restore)
        config["checkpoint"]["restore_from"] = str(restore)
    for subdir in ["state", "logs"]:
        (run / subdir).mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config["backup_checkpoint_dir"] = str(checkpoint_dir)
    (run / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    plan = {
        "model_source": str(model),
        "prepared_source": str(prepared),
        "local_model": values["LOCAL_MODEL"],
        "local_data": str(local_data),
        "local_ckpt_dir": values["LOCAL_CKPT_DIR"],
        "topology": topology,
        "parallelism": parallelism,
        "resume_settings": resume_settings,
        "ready": ready,
        "config_sha256": data_gate.digest(run / "train.yaml"),
    }
    (run / "state/READY").write_text(json.dumps(plan, indent=2) + "\n")
    print(run / "train.yaml")


if __name__ == "__main__":
    main()
