#!/usr/bin/env python3
"""Turn verified teacher runs into SFT trajectories.

Selection here is for SFT, not RL, and the difference matters. RL wants tasks
the policy sometimes solves, because a task that is always solved produces no
advantage signal. SFT wants correct demonstrations, so a task the teacher solved
all eight times is the *best* source, not a discard -- it yields eight clean
samples of the same problem being done right.

What decides difficulty is the student, not the teacher. A task both models
solve teaches nothing; the useful frontier is where the teacher succeeds and the
student fails. Pass --student-results once a student baseline exists and those
tasks are kept; without it every solved task is kept and the split is left to
whoever trains.

Quality filtering exists because a verified pass is not the same as a good
demonstration. A run that rewrites the same file eleven times before stumbling
onto the fix reaches reward 1 and teaches thrashing. The heuristics below are
deliberately blunt -- they cut the obviously bad tail, not the merely long.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import socket
import statistics
import sys
from collections import defaultdict

WRITE = re.compile(r"(cat\s*>|tee\s|apply_patch|sed -i|>\s*/\w|<<\s*.?EOF)")
EXPLORE = re.compile(
    r"^\s*(ls|cat|head|tail|grep|rg|find|file|wc|tree|git (log|diff|show|status))\b"
)
TEST = re.compile(
    r"\b(pytest|npm (run )?test|cargo test|go test|make test|tox|unittest)\b"
)


def task_name(trial_dir: pathlib.Path) -> str:
    return re.sub(r"__[A-Za-z0-9]+$", "", trial_dir.name)


def load_trial(t: pathlib.Path) -> dict | None:
    rj = t / "result.json"
    if not rj.exists():
        return None
    try:
        r = json.loads(rj.read_text())
    except Exception:
        return None
    traj = t / "agent" / "mini-swe-agent.trajectory.json"
    if not traj.exists():
        return None
    try:
        msgs = (json.loads(traj.read_text()) or {}).get("messages") or []
    except Exception:
        return None
    if not msgs:
        return None

    cmds = []
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            try:
                c = json.loads((tc.get("function") or {}).get("arguments") or "{}")
                if c.get("command"):
                    cmds.append(c["command"])
            except Exception:
                pass
        for a in (m.get("extra") or {}).get("actions") or []:
            c = a.get("command") if isinstance(a, dict) else a
            if isinstance(c, str) and c:
                cmds.append(c)

    return {
        "dir": t,
        "task": task_name(t),
        "passed": bool(
            ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        ),
        "exception": (r.get("exception_info") or {}).get("exception_type"),
        "messages": msgs,
        "steps": sum(1 for m in msgs if m.get("role") == "assistant"),
        "cmds": cmds,
        "writes": sum(1 for c in cmds if WRITE.search(c)),
        "explores": sum(1 for c in cmds if EXPLORE.match(c)),
        "ran_tests": any(TEST.search(c) for c in cmds),
    }


def quality_reject(tr: dict, max_steps: int, max_wr: float) -> str | None:
    """Reasons a verified-correct run still makes a poor demonstration."""
    if tr["steps"] > max_steps:
        return f"too long ({tr['steps']} steps)"
    if tr["explores"] and tr["writes"] / tr["explores"] > max_wr:
        # Rewriting far more often than reading is the signature of a model
        # that got stuck and brute-forced its way out.
        return f"thrashing (write/explore {tr['writes'] / tr['explores']:.1f})"
    if tr["steps"] < 2:
        return "trivial (under 2 steps)"
    # Re-running the same inspection command while working through a file is
    # normal; 50% rejected two thirds of otherwise clean runs. Only flag a run
    # where most of the transcript is literal repetition.
    dupes = len(tr["cmds"]) - len({re.sub(r"\s+", " ", c)[:80] for c in tr["cmds"]})
    if tr["cmds"] and dupes / len(tr["cmds"]) > 0.7:
        return f"repetitive ({dupes}/{len(tr['cmds'])} duplicate commands)"
    return None


def keep_anyway(task: str, frac: float) -> bool:
    """Whether to keep a task the student can already solve.

    Dropping every one of them leaves a training set made entirely of problems
    the student fails, which is a distribution it never sees at convergence --
    a slice of already-solvable work keeps the easy end of the range present so
    the model is not only ever corrected. Selection is by hash of the task name
    rather than at random so repeated extractions agree with each other.
    """
    if frac <= 0:
        return False
    digest = hashlib.sha1(task.encode()).digest()
    return (digest[0] + digest[1] * 256) % 1000 < frac * 1000


def refuse_on_login_node() -> None:
    """Extraction holds every kept trajectory in memory before writing.

    The login node caps each user at 10GB (memory.max on the user slice), and a
    V2 run over 5500 trials blows past that: memory.events showed 483964 'high'
    reclaim events, and the slice-wide pressure took down the proxy, the
    reporting loop, and every harness alongside the extraction -- three times,
    each looking like the login node had frozen. Run it under srun instead.
    """
    if os.environ.get("SLURM_JOB_ID"):
        return
    if os.environ.get("ALLOW_LOGIN_NODE_EXTRACT"):
        return
    sys.exit(
        f"refusing to run on {socket.gethostname()}: this needs more memory than\n"
        "the 10GB login-node cgroup allows, and overrunning it kills the proxy\n"
        "and every harness with it. Submit it to a compute node, e.g.\n\n"
        "  jid=$(squeue -u $USER -h -t R -o '%i %N' | head -1)\n"
        "  srun --jobid=${jid%% *} --overlap -N1 -n1 -w ${jid##* } \\\n"
        "       --cpus-per-task=16 --mem=64G python3 extract_sft_data.py ...\n\n"
        "Set ALLOW_LOGIN_NODE_EXTRACT=1 to override for a small --results-glob."
    )


def main() -> None:
    refuse_on_login_node()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Harbor results parent directory")
    ap.add_argument("--results-glob", default="results/swerebench_v4_s*")
    ap.add_argument("--out", default="sft_data/swerebench_v4.jsonl")
    ap.add_argument("--student-results", default="", help="glob of student runs")
    ap.add_argument(
        "--keep-solved-frac",
        type=float,
        default=0.15,
        help="fraction of student-solved tasks to keep anyway",
    )
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--max-write-explore", type=float, default=6.0)
    ap.add_argument("--per-task", type=int, default=2, help="best N runs per task")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    trials = []
    for d in sorted(root.glob(args.results_glob)):
        for t in d.iterdir():
            if t.is_dir():
                tr = load_trial(t)
                if tr:
                    trials.append(tr)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for tr in trials:
        by_task[tr["task"]].append(tr)

    student_solved: set[str] = set()
    if args.student_results:
        for d in sorted(root.glob(args.student_results)):
            for t in d.iterdir():
                if not t.is_dir():
                    continue
                tr = load_trial(t)
                if tr and tr["passed"]:
                    student_solved.add(tr["task"])

    kept, stats = [], defaultdict(int)
    for task, runs in sorted(by_task.items()):
        good = [r for r in runs if r["passed"]]
        if not good:
            stats["task: teacher never solved"] += 1
            continue
        if task in student_solved and not keep_anyway(task, args.keep_solved_frac):
            stats["task: student already solves"] += 1
            continue
        if task in student_solved:
            stats["task: student solves, kept as anchor"] += 1

        scored = []
        for r in good:
            why = quality_reject(r, args.max_steps, args.max_write_explore)
            if why:
                stats[f"run rejected: {why.split(' (')[0]}"] += 1
                continue
            # Prefer runs that checked their own work, then shorter ones.
            scored.append(((0 if r["ran_tests"] else 1, r["steps"]), r))
        if not scored:
            stats["task: all passing runs low quality"] += 1
            continue

        scored.sort(key=lambda x: x[0])
        for _, r in scored[: args.per_task]:
            kept.append(r)
        stats["task: kept"] += 1

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in kept:
            f.write(
                json.dumps(
                    {
                        "task": r["task"],
                        "source": str(r["dir"].parent.name),
                        "steps": r["steps"],
                        "ran_tests": r["ran_tests"],
                        "messages": r["messages"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"trials scanned      {len(trials)}")
    print(f"tasks seen          {len(by_task)}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<38}{v}")
    print(f"trajectories written {len(kept)} -> {out}")
    if kept:
        s = [r["steps"] for r in kept]
        print(
            f"  steps median {statistics.median(s):.0f}, "
            f"ran own tests {sum(r['ran_tests'] for r in kept)}/{len(kept)}"
        )


if __name__ == "__main__":
    main()
