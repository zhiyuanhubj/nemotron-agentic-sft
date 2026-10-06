"""Stage and launch on eight empty held Slurm nodes, then watch this run."""

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

import yaml

import data_gate

CODE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--automodel", type=Path, required=True)
    parser.add_argument(
        "--jobs", required=True, help="Comma-separated existing one-node allocation IDs"
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Shared training Python; defaults to AUTOMODEL/.venv/bin/python",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until eight supplied allocations have empty GPUs",
    )
    parser.add_argument("--port", type=int, default=29676)
    parser.add_argument("--subnet", default="10.1.")
    parser.add_argument("--cpus", type=int, default=96)
    parser.add_argument("--memory", default="1500G")
    parser.add_argument("--min-free-bytes", type=int, default=1099511627776)
    args = parser.parse_args()
    jobs = args.jobs.split(",")
    assert (
        len(jobs) >= 8
        and len(set(jobs)) == len(jobs)
        and all(j.isdigit() for j in jobs)
    )
    run, repo = args.run_dir.resolve(), args.automodel.resolve()
    python = (args.python or repo / ".venv/bin/python").resolve()
    assert python.is_file()
    config = yaml.safe_load((run / "train.yaml").read_text())
    plan = json.loads((run / "state/READY").read_text())
    assert plan["config_sha256"] == data_gate.digest(run / "train.yaml")
    data_gate.verify(
        plan["prepared_source"],
        config.get("accepted_manifest"),
        config.get("integrity_audit"),
    )
    lock = (run / "state/launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (run / "state/status.json").exists(), (
        "Existing launch state: inspect before recovery"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    q = shlex.quote
    nvme = Path(plan["local_model"]).parent

    def state(status, **fields):
        record = {
            "status": status,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **fields,
        }
        temporary = run / "state/status.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(run / "state/status.json")
        print(json.dumps(record), flush=True)

    def remote(pair, script, timeout=60):
        job, node = pair
        return subprocess.run(
            [
                "srun",
                f"--jobid={job}",
                "--overlap",
                "-N1",
                "-n1",
                "-w",
                node,
                "bash",
                "-lc",
                script,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout

    def inspect(pair):
        try:
            output = remote(
                pair,
                "\n".join(
                    [
                        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                        f"df --output=avail -B1 {q(str(nvme.parent))} | tail -1",
                        "if docker ps --format '{{.Image}}' 2>/dev/null | grep -Eq 'vllm|sglang'; then echo ACTIVE_INFERENCE; fi",
                    ]
                ),
            )
            numbers = [
                int(x.strip()) for x in output.splitlines() if x.strip().isdigit()
            ]
            if (
                len(numbers) == 9
                and max(numbers[:8]) < 100
                and numbers[8] > args.min_free_bytes
                and "ACTIVE_INFERENCE" not in output
            ):
                return pair
        except (subprocess.SubprocessError, OSError):
            pass
        return None

    def stage(pair):
        checkpoint = q(plan["local_ckpt_dir"])
        script = [
            "set -euo pipefail",
            "umask 077",
            f"mkdir -p {q(plan['local_model'])} {q(plan['local_data'])} {checkpoint}",
            f'test -z "$(find {checkpoint} -mindepth 1 -maxdepth 1 -print -quit)"',
            f"rsync -a --checksum {q(plan['model_source'] + '/')} {q(plan['local_model'] + '/')}",
            f"rsync -a --checksum {q(plan['prepared_source'] + '/')} {q(plan['local_data'] + '/')}",
            f"test -f {q(plan['local_model'] + '/model.safetensors.index.json')}",
            f"PYTHONPATH={q(str(CODE))} {q(str(python))} {q(str(CODE / 'data_gate.py'))} {q(plan['local_data'])}",
        ]
        remote(pair, "\n".join(script), timeout=3600)
        print(f"STAGED {pair}", flush=True)

    while True:
        output = subprocess.check_output(
            ["squeue", "-h", "-j", ",".join(jobs), "-o", "%i|%T|%N|%L"],
            env=env,
            text=True,
        )
        candidates = []
        for line in output.splitlines():
            job, status, node, remaining = line.split("|")
            if (
                job not in jobs
                or status != "RUNNING"
                or not node
                or node == "(null)"
                or "," in node
                or "[" in node
            ):
                continue
            days, clock = (
                remaining.split("-", 1) if "-" in remaining else ("0", remaining)
            )
            seconds = (
                float("inf")
                if clock in {"UNLIMITED", "N/A"}
                else int(days) * 86400
                + sum(
                    int(value) * 60**i
                    for i, value in enumerate(reversed(clock.split(":")))
                )
            )
            candidates.append((seconds, (job, node)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            observed = list(pool.map(inspect, [pair for _, pair in candidates]))
        available, seen_nodes = [], set()
        for pair in observed:
            if pair and pair[1] not in seen_nodes:
                available.append(pair)
                seen_nodes.add(pair[1])
        if len(available) >= 8:
            selected = available[:8]
            state("staging", nodes=selected)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(stage, selected))
            assert all(inspect(pair) for pair in selected), (
                "GPU usage changed during staging"
            )
            break
        if not args.wait:
            raise RuntimeError(
                f"Need eight empty nodes; currently available: {available}"
            )
        state("waiting_for_8_idle_nodes", idle_nodes=available)
        time.sleep(60)
    master = remote(
        selected[0],
        f"ip -o -4 addr show | awk '$4 ~ /^{re.escape(args.subnet)}/ {{split($4,a,\"/\");print a[1];exit}}'",
    ).strip()
    assert master, "Master node has no IP on requested subnet"
    workers, logs = [], []
    try:
        for rank, (job, node) in enumerate(selected):
            command = [
                "set -euo pipefail",
                f"cd {q(str(repo))}",
                f"NIC=$(ip -o -4 addr show | awk '$4 ~ /^{re.escape(args.subnet)}/ {{print $2;exit}}')",
                'test -n "$NIC"',
                "export NCCL_SOCKET_IFNAME=$NIC GLOO_SOCKET_IFNAME=$NIC",
                "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false",
                "export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
                "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=8",
                "export FI_EFA_USE_HUGE_PAGE=0 NCCL_DEBUG=INFO",
                f'export PYTHONPATH={q(str(CODE) + ":" + str(repo))}:"${{PYTHONPATH:-}}"',
                f"export TRITON_CACHE_DIR={q(str(nvme / 'cache/triton'))}",
                f"export TORCHINDUCTOR_CACHE_DIR={q(str(nvme / 'cache/torchinductor'))}",
                'mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"',
                f"exec {q(str(python))} -m torch.distributed.run --nnodes=8 --nproc-per-node=8 "
                f"--node-rank={rank} --master-addr={q(master)} --master-port={args.port} "
                f"{q(str(CODE / 'run_training.py'))} {q(str(run / 'train.yaml'))}",
            ]
            log = (run / f"logs/rank{rank}.log").open("a")
            logs.append(log)
            workers.append(
                subprocess.Popen(
                    [
                        "srun",
                        f"--jobid={job}",
                        "--overlap",
                        "-N1",
                        "-n1",
                        "-w",
                        node,
                        f"--cpus-per-task={args.cpus}",
                        f"--mem={args.memory}",
                        "bash",
                        "-lc",
                        "\n".join(command),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            )
        state(
            "launched",
            nodes=selected,
            master=master,
            srun_pids=[p.pid for p in workers],
        )
        while True:
            codes = [p.poll() for p in workers]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"Training rank failed: {codes}")
            if all(code == 0 for code in codes):
                state("completed", nodes=selected, exit_codes=codes)
                break
            time.sleep(15)
    except BaseException:
        for process in workers:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        state("failed", nodes=selected, exit_codes=[p.poll() for p in workers])
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
