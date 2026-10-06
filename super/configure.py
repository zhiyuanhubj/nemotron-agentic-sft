"""Resolve a production config into an isolated run directory without launching."""

import argparse
import json
import os
from pathlib import Path
import re
from string import Template

import yaml

import data_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/full_sft.yaml"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--nvme-root", type=Path, default=Path("/opt/dlami/nvme/nemotron35super")
    )
    parser.add_argument("--accepted-manifest", type=Path)
    parser.add_argument("--integrity-audit", type=Path)
    parser.add_argument("--restore-from", type=Path)
    args = parser.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name)
    assert args.model.is_dir()
    run = args.run_dir.resolve()
    assert not (run / "train.yaml").exists(), "Use a fresh run directory"
    nvme = args.nvme_root.resolve()
    local_data = nvme / "data" / args.run_name
    values = {
        "LOCAL_MODEL": str(nvme / "model"),
        "LOCAL_DATA": str(local_data),
        "LOCAL_CKPT_DIR": str(nvme / "checkpoints" / args.run_name),
        "LOCAL_WANDB": str(nvme / "wandb" / args.run_name),
        "RUN_NAME": args.run_name,
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
        "ACCEPTED_MANIFEST": str(args.accepted_manifest.resolve())
        if args.accepted_manifest
        else "",
        "INTEGRITY_AUDIT": str(args.integrity_audit.resolve())
        if args.integrity_audit
        else "",
    }
    config = yaml.safe_load(Template(args.config.read_text()).substitute(values))
    if config.get("accepted_manifest") is None and "accepted_manifest" in config:
        raise ValueError(
            "Reviewed recipe requires --accepted-manifest and --integrity-audit"
        )
    if config.get("accepted_manifest") and not args.integrity_audit:
        raise ValueError("Supply --integrity-audit for current source holds")
    ready = data_gate.verify(
        args.prepared_dir, args.accepted_manifest, args.integrity_audit
    )
    if not os.environ.get("WANDB_ENTITY"):
        config.pop("wandb", None)
    config["run_dir"] = str(run)
    if args.restore_from:
        from verify_checkpoint import verify_complete

        restore = args.restore_from.resolve()
        assert restore.parent == run / "checkpoints", (
            "Resume only this run full-state backup"
        )
        verify_complete(restore)
        config["checkpoint"]["restore_from"] = str(restore)
    for subdir in ["state", "logs", "checkpoints"]:
        (run / subdir).mkdir(parents=True, exist_ok=True)
    (run / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    plan = {
        "model_source": str(args.model.resolve()),
        "prepared_source": str(args.prepared_dir.resolve()),
        "local_model": values["LOCAL_MODEL"],
        "local_data": str(local_data),
        "local_ckpt_dir": values["LOCAL_CKPT_DIR"],
        "ready": ready,
        "config_sha256": data_gate.digest(run / "train.yaml"),
    }
    (run / "state/READY").write_text(json.dumps(plan, indent=2) + "\n")
    print(run / "train.yaml")


if __name__ == "__main__":
    main()
