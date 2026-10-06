"""Launch one torchrun process per host using an explicit, configurable topology."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import data_gate
import yaml
from topology import validate_topology


def build_command(config_path, topology, node_rank, master_addr, master_port):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={topology['nnodes']}",
        f"--nproc-per-node={topology['nproc_per_node']}",
    ]
    if topology["nnodes"] == 1:
        command.append("--standalone")
    else:
        if not master_addr:
            raise ValueError("Multi-node training requires --master-addr or MASTER_ADDR")
        command.extend(
            [
                f"--node-rank={node_rank}",
                f"--master-addr={master_addr}",
                f"--master-port={master_port}",
            ]
        )
    return command + [str(Path(__file__).with_name("run_training.py")), str(config_path)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--automodel", type=Path, help="Optional source checkout; otherwise use installed package"
    )
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR"))
    parser.add_argument(
        "--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500"))
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the launch command"
    )
    args = parser.parse_args()
    config_path = args.run_dir.resolve() / "train.yaml"
    config = yaml.safe_load(config_path.read_text())
    plan = json.loads((args.run_dir / "state/READY").read_text())
    if plan["config_sha256"] != data_gate.digest(config_path):
        parser.error("Configuration changed after generation; regenerate it with configure.py")
    topology = plan["topology"]
    if not 0 <= args.node_rank < topology["nnodes"]:
        parser.error("--node-rank must be between zero and nnodes - 1")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be between 1 and 65535")
    validate_topology(config, topology["nnodes"] * topology["nproc_per_node"], plan["model_source"])
    env = os.environ.copy()
    paths = [str(Path(__file__).resolve().parent)]
    if args.automodel:
        if not (args.automodel / "nemo_automodel").is_dir():
            parser.error("--automodel must point to an Automodel source checkout")
        paths.append(str(args.automodel.resolve()))
    env["PYTHONPATH"] = os.pathsep.join(
        paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    try:
        command = build_command(
            config_path, topology, args.node_rank, args.master_addr, args.master_port
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
