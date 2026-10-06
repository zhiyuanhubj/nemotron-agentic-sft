"""Validate prepared data and topology, then run full-parameter SFT."""

import hashlib
import json
import os
import sys
from pathlib import Path

import data_gate
import yaml
from topology import validate_topology


def main():
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Start this entry point through launch.py or torchrun")
    config_path = Path(sys.argv[1]).resolve()
    config = yaml.safe_load(config_path.read_text())
    run = Path(config["run_dir"]).resolve()
    os.environ["RUN_DIR"] = str(run)
    os.environ["LOCAL_CKPT_DIR"] = config["checkpoint"]["checkpoint_dir"]
    os.environ["BACKUP_CKPT_DIR"] = config["backup_checkpoint_dir"]
    assert "restore_from" not in config
    plan = json.loads((run / "state/READY").read_text())
    assert plan["config_sha256"] == data_gate.digest(config_path)
    # Each node leader hashes its local staged data once; workers verify metadata.
    data_gate.verify(
        Path(config["prepared_data_gate"]).parent,
        verify_hashes=int(os.environ.get("LOCAL_RANK", "0")) == 0,
    )
    assert json.loads(Path(config["prepared_data_gate"]).read_text()) == plan["ready"]
    summary = json.loads((Path(config["prepared_data_gate"]).parent / "summary.json").read_text())
    assert summary["tokenizer_sha256"] == data_gate.tokenizer_fingerprint(plan["local_model"]), (
        "Training model tokenization assets do not match the prepared data"
    )
    restore = config["checkpoint"].get("restore_from")
    if not restore and config["checkpoint_storage"] == "node_local":
        local_checkpoint = Path(config["checkpoint"]["checkpoint_dir"])
        if local_checkpoint.exists() and any(local_checkpoint.iterdir()):
            raise RuntimeError("Fresh training requires an empty node-local checkpoint directory")
    if restore and config["checkpoint_storage"] == "node_local":
        from verify_checkpoint import verify_complete

        assert Path(restore).resolve().parent == Path(config["backup_checkpoint_dir"])
        verify_complete(Path(restore).resolve())
    if config.get("wandb"):
        if os.environ.get("WANDB_KEY_FILE"):
            os.environ["WANDB_API_KEY"] = Path(os.environ["WANDB_KEY_FILE"]).read_text().strip()
        Path(config["wandb"]["dir"]).mkdir(parents=True, exist_ok=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"
    topology = plan["topology"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    assert world == topology["nnodes"] * topology["nproc_per_node"]
    assert local_world == topology["nproc_per_node"]
    validate_topology(config, world, plan["local_model"])
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        node_rank = int(os.environ.get("RANK", "0")) // int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        record = {
            "node_rank": node_rank,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
        (run / f"state/rank{node_rank}_started.json").write_text(json.dumps(record) + "\n")
    from checkpoint_state import install as install_checkpoint_state
    from early_stopping import install as install_early_stopping
    from token_budget import install as install_token_budget

    install_checkpoint_state()
    if config["checkpoint_storage"] == "node_local":
        from nvme_checkpoint import install

        install()
    install_early_stopping(config)
    install_token_budget(config)
    from nemo_automodel.cli.app import main as train

    return train()


if __name__ == "__main__":
    raise SystemExit(main())
