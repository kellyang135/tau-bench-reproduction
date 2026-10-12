#!/usr/bin/env python3
"""Recompute tau-bench pass^k from published rewards, without API calls.

This is an audit of archived results, not a fresh model evaluation or regrading.
Only the Python standard library is required. See replay.py for tool/state replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "7362a1708d720a4d697d042f21fd8637ea144c3b"
UPSTREAM_URL = "https://github.com/sierra-research/tau-bench"
ARCHIVE_PATH = Path("historical_trajectories/gpt-4o-airline.json")
SUCCESS_TOLERANCE = 1e-6


def is_success(reward: float) -> bool:
    """Use the success tolerance in upstream run.py."""
    return 1 - SUCCESS_TOLERANCE <= reward <= 1 + SUCCESS_TOLERANCE


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list) or not records:
        raise ValueError("trajectory file must contain a nonempty JSON list")
    return records


def validate_records(
    records: list[dict[str, Any]], expected_tasks: int, expected_trials: int
) -> None:
    """Require a complete rectangular task/trial grid; never drop error runs."""
    if expected_tasks < 1 or expected_trials < 1:
        raise ValueError("expected task and trial counts must be positive")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list")
    seen: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index}: expected an object")
        for name, limit in (("task_id", expected_tasks), ("trial", expected_trials)):
            value = record.get(name)
            if type(value) is not int or not 0 <= value < limit:
                raise ValueError(f"record {index}: {name} must be an integer in [0, {limit})")
        key = (record["task_id"], record["trial"])
        if key in seen:
            raise ValueError(f"duplicate task/trial pair: {key}")
        seen.add(key)
        reward = record.get("reward")
        if (
            type(reward) not in (int, float)
            or not math.isfinite(reward)
            or not 0 <= reward <= 1
        ):
            raise ValueError(f"record {index}: reward must be a finite number in [0, 1]")
        info = record.get("info")
        if not isinstance(info, dict) or not isinstance(record.get("traj"), list):
            raise ValueError(f"record {index}: info must be an object and traj a list")
        if "error" in info and reward != 0:
            raise ValueError(f"record {index}: upstream error runs must have reward 0")
        reward_info = info.get("reward_info")
        if reward_info is not None:
            if not isinstance(reward_info, dict):
                raise ValueError(f"record {index}: reward_info must be an object or null")
            nested = reward_info.get("reward")
            if type(nested) not in (int, float) or not math.isfinite(nested):
                raise ValueError(f"record {index}: reward_info.reward must be finite")
            if not math.isclose(reward, nested, abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"record {index}: reward disagrees with reward_info.reward")
    expected = {(task, trial) for task in range(expected_tasks) for trial in range(expected_trials)}
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"incomplete task/trial grid; missing {len(missing)} pair(s): {missing[:10]}")


def summarize(
    records: list[dict[str, Any]], expected_tasks: int = 50, expected_trials: int = 4
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_records(records, expected_tasks, expected_trials)
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)
    rows = []
    for task_id, task_runs in sorted(by_task.items()):
        successes = sum(is_success(run["reward"]) for run in task_runs)
        row = {
            "task_id": task_id,
            "trials": expected_trials,
            "successes": successes,
            "success_rate": successes / expected_trials,
            "error_runs": sum("error" in run["info"] for run in task_runs),
        }
        for k in range(1, expected_trials + 1):
            row[f"pass^{k}"] = math.comb(successes, k) / math.comb(expected_trials, k)
        rows.append(row)
    logged_costs = [run["info"].get("user_cost") for run in records]
    valid_costs = [cost for cost in logged_costs if type(cost) in (int, float) and math.isfinite(cost)]
    summary = {
        "mode": "published_reward_recomputation",
        "fresh_model_evaluation": False,
        "regraded": False,
        "tasks": expected_tasks,
        "trials_per_task": expected_trials,
        "runs": len(records),
        "successful_runs": sum(row["successes"] for row in rows),
        "error_runs": sum(row["error_runs"] for row in rows),
        "mean_reward": sum(run["reward"] for run in records) / len(records),
        "pass_k": {str(k): sum(row[f"pass^{k}"] for row in rows) / expected_tasks
                   for k in range(1, expected_trials + 1)},
        "success_count_distribution": dict(sorted(Counter(row["successes"] for row in rows).items())),
        "formula": "mean over tasks of C(successes, k) / C(trials, k)",
        "interpretation": "Probability that all k sampled attempts succeed, without replacement; not pass@k.",
        "cost": {
            "new_api_cost_usd": 0.0,
            "historical_total_cost_usd": None,
            "historical_agent_cost_usd": None,
            "sum_of_logged_user_cost_values": sum(valid_costs) if valid_costs else None,
            "runs_with_logged_user_cost": len(valid_costs),
            "note": "Historical total cost is absent from the archive. In the pinned source, user_cost is overwritten after each response; summing the saved values does not establish full spend.",
        },
    }
    return summary, rows


def provenance(benchmark_dir: Path, trajectory: Path) -> dict[str, Any]:
    """Identify exact source bytes and reject a changed upstream snapshot."""
    actual_commit = None
    verified_files = 0
    manifest_sha256 = None
    if (benchmark_dir / ".git").exists():
        actual_commit = subprocess.run(
            ["git", "-C", str(benchmark_dir), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if actual_commit != UPSTREAM_COMMIT:
            raise ValueError(f"expected upstream commit {UPSTREAM_COMMIT}, found {actual_commit}")
        clean = subprocess.run(["git", "-C", str(benchmark_dir), "diff", "--quiet", "HEAD", "--"])
        if clean.returncode:
            raise ValueError("upstream checkout has modified tracked files")
    manifest_path = benchmark_dir.parent.parent / "vendor-manifest.json"
    if manifest_path.exists():
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("commit") != UPSTREAM_COMMIT:
            raise ValueError("vendor manifest does not identify the pinned commit")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("vendor manifest must contain a nonempty file hash mapping")
        for relative, expected_hash in files.items():
            file_path = (benchmark_dir / relative).resolve()
            if not file_path.is_relative_to(benchmark_dir):
                raise ValueError(f"invalid vendor manifest path: {relative}")
            if not file_path.is_file() or hashlib.sha256(file_path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"upstream snapshot checksum mismatch: {relative}")
        verified_files = len(files)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_commit is None and not verified_files:
        raise ValueError("upstream source needs a pinned Git checkout or vendor-manifest.json")
    return {
        "upstream_repository": UPSTREAM_URL,
        "pinned_commit": UPSTREAM_COMMIT,
        "git_head_verified": actual_commit is not None,
        "vendor_files_verified": verified_files,
        "vendor_manifest_sha256": manifest_sha256,
        "trajectory_file": str(trajectory.relative_to(benchmark_dir)) if trajectory.is_relative_to(benchmark_dir) else trajectory.name,
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "model_label_from_filename": "gpt-4o" if trajectory.name == ARCHIVE_PATH.name else None,
        "exact_model_snapshot": None,
        "historical_seed": None,
        "historical_temperature": None,
        "note": "The archived result file does not establish the exact model snapshot, seed, temperature, or full run configuration.",
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "external/tau-bench")
    parser.add_argument("--trajectories", type=Path, help="Default: published GPT-4o airline archive")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--expected-trials", type=int, default=4)
    args = parser.parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    trajectory = (args.trajectories or benchmark_dir / ARCHIVE_PATH).resolve()
    try:
        summary, rows = summarize(load_records(trajectory), args.expected_tasks, args.expected_trials)
        summary["provenance"] = provenance(benchmark_dir, trajectory)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "per_task.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Recomputed archived rewards: {summary['successful_runs']}/{summary['runs']} successes")
    print("  ".join(f"pass^{k}={value:.4%}" for k, value in summary["pass_k"].items()))
    print(f"Wrote {args.output_dir / 'summary.json'} and {args.output_dir / 'per_task.csv'}")


if __name__ == "__main__":
    main()
