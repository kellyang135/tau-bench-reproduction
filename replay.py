#!/usr/bin/env python3
"""Replay published airline tool calls using pinned upstream tools and grader.

No agent or simulated-user model runs. The conversation and tool arguments are
frozen to the archive. Pinned tools mutate a fresh database per run; the upstream
calculate_reward then scores that state and recorded assistant responses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from reproduce import ARCHIVE_PATH, ROOT, load_records, provenance, summarize, validate_records, write_json


def comparable(value: str) -> Any:
    """Ignore JSON serialization whitespace and object-key order only."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def replay_run(record: dict[str, Any], get_env: Any, Action: Any) -> dict[str, Any]:
    """Execute saved actions with the pinned environment tools and grader."""
    if any(not isinstance(message, dict) for message in record["traj"]):
        raise ValueError("every trajectory message must be an object")
    env = get_env("airline", user_strategy="human", user_model="unused",
                  task_split="test", task_index=record["task_id"])
    initial_hash = env.get_data_hash()
    stored_task = record["info"].get("task")
    if not isinstance(stored_task, dict):
        raise ValueError(f"task {record['task_id']}, trial {record['trial']}: no stored task")
    current_task = env.task.model_dump()
    target_matches = all(current_task[key] == stored_task.get(key)
                         for key in ("user_id", "actions", "outputs"))
    if not target_matches:
        raise ValueError(f"task {record['task_id']}: archived task targets differ from pinned task targets")

    tool_result_indices = {index for index, message in enumerate(record["traj"])
                           if message.get("role") == "tool"}
    used_results: set[int] = set()
    mismatches = []
    tools_used: Counter[str] = Counter()
    exact_matches = 0
    terminated = False
    termination_reason = "step_limit"
    assistant_steps = 0
    for index, message in enumerate(record["traj"]):
        if message.get("role") == "user" and index:
            previous = record["traj"][index - 1]
            if previous.get("role") == "assistant" and not previous.get("tool_calls"):
                if "###STOP###" in message.get("content", ""):
                    terminated = True
                    termination_reason = "user_stop"
        if message.get("role") != "assistant":
            continue
        if terminated:
            raise ValueError("archive contains agent actions after termination")
        assistant_steps += 1
        calls = message.get("tool_calls") or []
        if not calls:
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("assistant response must contain text")
            # Env.step(respond) would generate a user reply. Preserve the actual
            # recorded action instead; its content is used by the real grader.
            env.actions.append(Action(name="respond", kwargs={"content": content}))
            continue
        if len(calls) != 1:
            raise ValueError("expected one tool call per assistant message, as in the upstream agent")
        call = calls[0]
        call_id = call["id"]
        if not isinstance(call_id, str):
            raise ValueError("tool call id must be a string")
        if index + 1 not in tool_result_indices:
            raise ValueError(f"tool call {call_id} has no following result")
        used_results.add(index + 1)
        name = call["function"]["name"]
        kwargs = json.loads(call["function"]["arguments"])
        if not isinstance(kwargs, dict):
            raise ValueError("tool arguments must decode to an object")
        archived = record["traj"][index + 1]
        if (archived.get("tool_call_id") != call_id or archived.get("name") != name
                or not isinstance(archived.get("content"), str)):
            raise ValueError(f"tool result does not match call {call_id}")
        action = Action(name=name, kwargs=kwargs)
        env.actions.append(action)
        # This is the tool branch of upstream Env.step. Invoke the actual tool
        # directly so a terminating tool cannot grade/reset the state before
        # we capture the final hash. No tool implementation is copied here.
        if name in env.tools_map:
            try:
                observation = env.tools_map[name].invoke(data=env.data, **kwargs)
            except Exception as error:
                observation = f"Error: {error}"
        else:
            observation = f"Unknown action {name}"
        tools_used[name] += 1
        if observation == archived["content"]:
            exact_matches += 1
        if comparable(observation) != comparable(archived["content"]):
            mismatches.append({
                "message_index": index,
                "tool_name": name,
                "tool_call_id": call_id,
                "archived_observation_sha256": digest(archived["content"]),
                "replayed_observation_sha256": digest(observation),
            })
        terminated = name in env.terminate_tools
        if terminated:
            termination_reason = "terminating_tool"
    if used_results != tool_result_indices:
        raise ValueError("archive contains unpaired tool results")

    final_hash = env.get_data_hash()
    # This is the pinned, unmodified upstream grader. In this pinned version
    # output tasks check response substrings; other tasks compare state hashes.
    if not terminated and assistant_steps != 30:
        raise ValueError("unterminated archive did not reach the upstream 30-step limit")
    # An exhausted attempt receives zero in the pinned runner; grading its
    # final state post hoc could incorrectly turn an unfinished run into a pass.
    reward_result = env.calculate_reward() if terminated else None
    replayed_reward = reward_result.reward if reward_result else 0.0
    grade_info = reward_result.info.model_dump() if reward_result else {}
    stored_grade = record["info"].get("reward_info") or {}
    stored_grade_info = stored_grade.get("info") or {}
    gt_hash = grade_info.get("gt_data_hash")
    archived_gt_hash = stored_grade_info.get("gt_data_hash")
    return {
        "task_id": record["task_id"],
        "trial": record["trial"],
        "archived_reward": record["reward"],
        "replayed_reward": replayed_reward,
        "reward_matches": math.isclose(record["reward"], replayed_reward, abs_tol=1e-12, rel_tol=0),
        "termination_reason": termination_reason,
        "assistant_steps": assistant_steps,
        "graded": terminated,
        "task_targets_match": target_matches,
        "instruction_matches": current_task["instruction"] == stored_task.get("instruction"),
        "initial_data_hash": initial_hash,
        "final_data_hash_before_grading": final_hash,
        "grader_branch": ("response_outputs" if env.task.outputs else "database_state") if terminated else "not_graded",
        "ground_truth_data_hash": gt_hash,
        "archived_ground_truth_data_hash": archived_gt_hash,
        "ground_truth_hash_matches": gt_hash == archived_gt_hash if gt_hash and archived_gt_hash else None,
        "grader_info": grade_info,
        "tool_calls": sum(tools_used.values()),
        "tool_names": dict(sorted(tools_used.items())),
        "tool_observations_exact_match": exact_matches,
        "tool_observations_semantic_mismatch": len(mismatches),
        "observation_mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "external/tau-bench")
    parser.add_argument("--trajectories", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task-id", type=int, choices=range(50), metavar="0..49")
    parser.add_argument("--trial", type=int, choices=range(4), metavar="0..3")
    args = parser.parse_args()
    if args.trial is not None and args.task_id is None:
        parser.error("--trial requires --task-id")
    selected_subset = args.task_id is not None
    output_dir = args.output_dir or ROOT / ("results/smoke" if selected_subset else "results")
    benchmark_dir = args.benchmark_dir.resolve()
    trajectory = (args.trajectories or benchmark_dir / ARCHIVE_PATH).resolve()
    try:
        source = provenance(benchmark_dir, trajectory)
        records = load_records(trajectory)
        validate_records(records, 50, 4)
        if any("error" in record["info"] for record in records):
            raise ValueError("error runs cannot be tool-replayed; use reproduce.py to retain them in metrics")
        if selected_subset:
            records = [record for record in records if record["task_id"] == args.task_id
                       and (args.trial is None or record["trial"] == args.trial)]
        # LiteLLM is imported by upstream even in human-user mode. Its provider
        # price map is irrelevant here; suppress its optional network fetch.
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        sys.path.insert(0, str(benchmark_dir))
        from tau_bench.envs import get_env
        from tau_bench.types import Action
        replayed = [replay_run(record, get_env, Action) for record in records]
    except ImportError as error:
        parser.error(f"missing upstream dependency ({error}); install requirements-replay.txt into a virtual environment")
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    rescored = [{**record, "reward": result["replayed_reward"], "info": {}}
                for record, result in zip(records, replayed)]
    metrics = summarize(rescored)[0] if not selected_subset else None
    summary = {
        "mode": "recorded_action_replay_with_upstream_grader",
        "fresh_model_evaluation": False,
        "model_api_calls": 0,
        "new_api_cost_usd": 0.0,
        "runtime": {"python": sys.version.split()[0],
                    "litellm": importlib.metadata.version("litellm"),
                    "pydantic": importlib.metadata.version("pydantic")},
        "provenance": source,
        "runs": len(replayed),
        "selection": {"task_id": args.task_id, "trial": args.trial},
        "termination_reasons": dict(sorted(Counter(run["termination_reason"] for run in replayed).items())),
        "graded_runs": sum(run["graded"] for run in replayed),
        "reward_matches": sum(run["reward_matches"] for run in replayed),
        "reward_mismatches": sum(not run["reward_matches"] for run in replayed),
        "task_targets_match": all(run["task_targets_match"] for run in replayed),
        "instruction_mismatches": sum(not run["instruction_matches"] for run in replayed),
        "ground_truth_hash_comparisons": sum(run["ground_truth_hash_matches"] is not None for run in replayed),
        "ground_truth_hash_mismatches": sum(run["ground_truth_hash_matches"] is False for run in replayed),
        "tool_calls": sum(run["tool_calls"] for run in replayed),
        "tool_observations_exact_match": sum(run["tool_observations_exact_match"] for run in replayed),
        "tool_observations_semantic_mismatch": sum(run["tool_observations_semantic_mismatch"] for run in replayed),
        "replayed_pass_k": metrics["pass_k"] if metrics else None,
        "metrics_note": "No benchmark aggregate is computed for a smoke subset." if selected_subset else "Full 50-task, 4-trial grid.",
        "method": "Fresh database per recorded run; pinned tools execute saved arguments and the pinned grader scores terminated episodes. User replies and agent decisions are not regenerated.",
        "limitations": [
            "Matching replay does not measure current model performance or establish the original model snapshot, generation settings, or seed.",
            "The archive-upload commit is pinned; the commit originally used to generate the trajectories is unknown.",
            "The historical grader checks response outputs OR database state depending on task; it does not comprehensively audit permission or timing compliance.",
        ],
        "runs_detail": replayed,
    }
    write_json(output_dir / "replay.json", summary)
    columns = ["task_id", "trial", "archived_reward", "replayed_reward", "reward_matches",
               "termination_reason", "assistant_steps", "graded", "grader_branch", "tool_calls", "tool_observations_semantic_mismatch",
               "final_data_hash_before_grading", "ground_truth_data_hash", "ground_truth_hash_matches"]
    with (output_dir / "replay_runs.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(replayed)
    print(f"Replayed {len(replayed)} archived runs; rewards matched: {summary['reward_matches']}/{len(replayed)}")
    print(f"Tool observations: {summary['tool_calls']} calls, {summary['tool_observations_semantic_mismatch']} semantic mismatches")
    print(f"Wrote {output_dir / 'replay.json'} and {output_dir / 'replay_runs.csv'}")
    if any(summary[key] for key in ("reward_mismatches", "instruction_mismatches",
                                    "ground_truth_hash_mismatches", "tool_observations_semantic_mismatch")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
