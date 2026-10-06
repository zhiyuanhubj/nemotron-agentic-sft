"""Check staged production data, then install run-scoped recipe hooks."""

import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

import data_gate


def main():
    config_path = Path(sys.argv[1]).resolve()
    config = yaml.safe_load(config_path.read_text())
    run = Path(config["run_dir"]).resolve()
    os.environ["RUN_DIR"] = str(run)
    os.environ["LOCAL_CKPT_DIR"] = config["checkpoint"]["checkpoint_dir"]
    assert "restore_from" not in config
    plan = json.loads((run / "state/READY").read_text())
    assert plan["config_sha256"] == data_gate.digest(config_path)
    # Each node leader hashes its local staged data once; workers verify metadata.
    data_gate.verify(
        Path(config["prepared_data_gate"]).parent,
        config.get("accepted_manifest"),
        config.get("integrity_audit"),
        verify_hashes=int(os.environ.get("LOCAL_RANK", "0")) == 0,
    )
    assert json.loads(Path(config["prepared_data_gate"]).read_text()) == plan["ready"]
    restore = config["checkpoint"].get("restore_from")
    if restore:
        from verify_checkpoint import verify_complete

        assert Path(restore).resolve().parent == run / "checkpoints"
        verify_complete(Path(restore).resolve())
    if config.get("wandb"):
        if os.environ.get("WANDB_KEY_FILE"):
            os.environ["WANDB_API_KEY"] = (
                Path(os.environ["WANDB_KEY_FILE"]).read_text().strip()
            )
        assert os.environ.get("WANDB_API_KEY"), "Set WANDB_API_KEY or WANDB_KEY_FILE"
        Path(config["wandb"]["dir"]).mkdir(parents=True, exist_ok=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        node_rank = int(os.environ.get("RANK", "0")) // int(
            os.environ.get("LOCAL_WORLD_SIZE", "8")
        )
        assert (
            int(os.environ["WORLD_SIZE"]) == 64
            and int(os.environ["LOCAL_WORLD_SIZE"]) == 8
        )
        record = {
            "node_rank": node_rank,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
        (run / f"state/rank{node_rank}_started.json").write_text(
            json.dumps(record) + "\n"
        )
    from nvme_checkpoint import install
    from early_stopping import install as install_early_stopping
    from token_budget import install as install_token_budget

    install()
    install_early_stopping(config)
    install_token_budget(config)
    from nemo_automodel.cli.app import main as train

    return train()


if __name__ == "__main__":
    raise SystemExit(main())
