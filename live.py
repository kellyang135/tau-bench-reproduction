#!/usr/bin/env python3
"""Run the pinned tau-bench airline agent and simulated user against OpenAI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from reproduce import ARCHIVE_PATH, ROOT, is_success, provenance, summarize, write_json

MODEL = "gpt-4o-2024-08-06"
API_BASE = "https://api.openai.com/v1"
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4o"
INPUT_RATE, CACHED_RATE, OUTPUT_RATE = map(Decimal, ("0.0000025", "0.00000125", "0.00001"))
CALL_RESERVE = 128000 * INPUT_RATE + 16384 * OUTPUT_RATE
REQUEST_TIMEOUT_SECONDS = 120


def now():
    return datetime.now(timezone.utc).isoformat()


class RunStopped(Exception):
    """A budget or API interruption; never classify this as agent behavior."""


def usage_cost(usage):
    """Compute an estimate from API token counts; reject absent/invalid usage."""
    prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if any(type(n) is not int or n < 0 for n in (prompt, output, cached)) or cached > prompt:
        raise ValueError("invalid_token_usage")
    return (prompt - cached) * INPUT_RATE + cached * CACHED_RATE + output * OUTPUT_RATE


def provider_request(source):
    """Keep API message fields; omit SDK response metadata and null fields."""
    allowed = {"system": {"role", "content", "name"}, "user": {"role", "content", "name"},
               "assistant": {"role", "content", "tool_calls", "function_call", "name", "refusal", "audio"},
               "tool": {"role", "content", "tool_call_id"}, "function": {"role", "content", "name"}}
    request = {k: source[k] for k in ("model", "messages", "temperature", "tools") if k in source}
    request["messages"] = [{k: v for k, v in m.items() if k in allowed[m["role"]] and v is not None}
                           for m in source["messages"]]
    return request


def openai_completion():
    """Use one SDK request, bypassing LiteLLM's automatic request-repair loop."""
    from openai import OpenAI
    from litellm import ModelResponse
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=API_BASE,
                    max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS)

    def complete(**request):
        native = client.chat.completions.create(**request)
        return ModelResponse(**native.model_dump(mode="json"))
    return complete


class CallLog:
    """Reserve a worst-case call before dispatch; retain it on unknown billing."""

    def __init__(self, path, budget, completion):
        self.path, self.budget, self.completion = path, Decimal(str(budget)), completion
        self.spent, self.uncertain = Decimal(0), Decimal(0)
        self.count, self.context, self.stopped = 0, {}, None

    def append(self, event):
        with self.path.open("a", encoding="utf-8") as target:
            target.write(json.dumps(event, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())

    def wrapper(self, role):
        def complete(**kwargs):
            if self.stopped:
                raise RunStopped(self.stopped)
            if kwargs.get("model") != MODEL or kwargs.get("custom_llm_provider") != "openai":
                raise ValueError("only the configured OpenAI snapshot is supported")
            if self.spent + self.uncertain + CALL_RESERVE > self.budget:
                self.stopped = "budget_exhausted"
                self.append({"event": "blocked", "at": now(), **self.context, "role": role,
                             "reason": self.stopped, "required_reserve_usd": float(CALL_RESERVE)})
                raise RunStopped(self.stopped)
            # Deep-copy before upstream mutates its conversation list. Never log credentials.
            source_request = json.loads(json.dumps(kwargs))
            request = provider_request(source_request)
            self.count += 1
            call = {"call_id": self.count, "role": role, **self.context}
            self.uncertain += CALL_RESERVE
            self.append({"event": "request", "at": now(), **call, "request": request,
                         "source_request": source_request,
                         "reserved_usd": float(CALL_RESERVE)})
            try:
                response = self.completion(**request)
            except BaseException as error:
                self.stopped = "api_interrupted"
                # Provider exception text can contain credentials or request headers.
                self.append({"event": "error", "at": now(), **call,
                             "error_type": type(error).__name__, "uncertain_usd": float(CALL_RESERVE)})
                raise RunStopped(self.stopped) from None
            try:
                data = response.model_dump(mode="json")
                self.append({"event": "response", "at": now(), **call, "response": data})
                if data.get("model") != MODEL:
                    self.stopped = "unexpected_model"
                    raise RunStopped(self.stopped)
                cost = usage_cost(data.get("usage") or {})
            except RunStopped:
                raise
            except Exception:
                self.stopped = "usage_unknown"
                raise RunStopped(self.stopped) from None
            self.uncertain -= CALL_RESERVE
            self.spent += cost
            self.append({"event": "cost", "at": now(), **call, "estimated_usd": float(cost)})
            if cost > CALL_RESERVE:
                self.stopped = "unexpected_cost"
                raise RunStopped(self.stopped)
            # The unchanged upstream agent and user read LiteLLM's cost field.
            response._hidden_params["response_cost"] = float(cost)
            return response
        return complete

    def costs(self):
        return {"estimated_api_cost_usd": float(self.spent),
                "uncertain_api_cost_upper_bound_usd": float(self.uncertain),
                "budget_usd": float(self.budget), "api_calls_dispatched": self.count,
                "note": "Token-based estimate, not an invoice; includes both roles and setup calls."}


def run_summary(records, task_ids, trials, status, log):
    complete = (status == "completed" and len(records) == len(task_ids) * trials
                and not any(r["info"].get("interrupted") for r in records))
    full = complete and sorted(task_ids) == list(range(50)) and trials == 4
    finished = [r for r in records if not r["info"].get("interrupted")]
    result = {"mode": "fresh_model_evaluation", "status": status,
              "scope": "baseline_attempt" if len(task_ids) == 50 and trials == 4 else "pilot",
              "selected_grid_complete": complete, "full_baseline_complete": full,
              "expected_runs": len(task_ids) * trials, "records_saved": len(records),
              "finished_runs": len(finished), "interrupted_runs": len(records) - len(finished),
              "successful_finished_runs": sum(is_success(r["reward"]) for r in finished),
              "execution_error_runs": sum("error" in r["info"] for r in finished),
              "pass_k": None, "cost": log.costs()}
    if full:
        result["pass_k"] = summarize(records, 50, 4)[0]["pass_k"]
        result["published_pass_1"] = 0.42
        result["pass_1_difference_percentage_points"] = 100 * (result["pass_k"]["1"] - 0.42)
    return result


def execute(args, source, completion=None):
    """Execute sequentially; save each record before starting the next task."""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import litellm
    sys.path.insert(0, str(args.benchmark_dir))
    from tau_bench.agents import tool_calling_agent
    from tau_bench.envs import get_env, user

    args.output_dir.mkdir(parents=True, exist_ok=False)
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    log = CallLog(args.output_dir / "calls.jsonl", args.budget_usd, completion or openai_completion())
    manifest = {"started_at": now(), "provenance": source, "agent_model": MODEL,
                "user_model": MODEL, "provider": "openai", "task_ids": args.task_ids,
                "trials": args.trials, "agent_strategy": "tool-calling", "user_strategy": "llm",
                "environment": "airline", "task_split": "test", "agent_temperature": 0,
                "user_temperature": "omitted (provider default)", "max_agent_steps": 30,
                "seed": 10, "seed_scope": "Python random only; no API seed is supplied",
                "concurrency": 1, "shuffle": False, "num_retries": 0,
                "transport": "OpenAI SDK with max_retries=0; adapted to LiteLLM ModelResponse for unchanged upstream consumers",
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS, "api_base": API_BASE,
                "message_formatting": "Only API-supported message fields are sent; null fields omitted. Raw upstream request is also logged.",
                "setup_task_index": args.task_ids[0],
                "budget_usd": float(log.budget), "per_call_reserve_usd": float(CALL_RESERVE),
                "prices_usd_per_million_tokens": {"input": 2.5, "cached_input": 1.25, "output": 10},
                "price_source": PRICE_SOURCE, "price_checked_date": "2026-10-11",
                "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ("litellm", "pydantic", "openai", "python-dotenv")},
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "comparison_limit": "The archive does not identify its model snapshot or full configuration; this is a reproduction attempt, not an exact historical rerun."}
    write_json(args.output_dir / "manifest.json", manifest)
    records, status = [], "running"
    write_json(args.output_dir / "summary.json", run_summary(records, args.task_ids, args.trials, status, log))
    env_args = dict(env_name="airline", user_strategy="llm", user_model=MODEL,
                    user_provider="openai", task_split="test")
    random.seed(10)
    try:
        with patch.object(user, "completion", log.wrapper("user")), \
                patch.object(tool_calling_agent, "completion", log.wrapper("agent")):
            log.context = {"phase": "setup", "task_id": None, "trial": None}
            setup_env = get_env(**env_args, task_index=args.task_ids[0])
            agent = tool_calling_agent.ToolCallingAgent(setup_env.tools_info, setup_env.wiki, MODEL, "openai", 0)
            for trial in range(args.trials):
                for task_id in args.task_ids:
                    log.context = {"phase": "task", "task_id": task_id, "trial": trial}
                    record = {"task_id": task_id, "trial": trial, "reward": 0.0, "info": {}, "traj": []}
                    try:
                        env = get_env(**env_args, task_index=task_id)
                        result = agent.solve(env=env, task_index=task_id, max_num_steps=30)
                        record.update(reward=result.reward, info=result.info, traj=result.messages)
                    except RunStopped as error:
                        status = str(error)
                        record["info"] = {"error": status, "interrupted": True,
                                          "note": "Reward zero is an upstream-format placeholder; this is not a completed agent attempt. Partial messages are in calls.jsonl."}
                    except Exception as error:
                        record["info"] = {"error": type(error).__name__, "error_kind": "execution_error",
                                          "note": "Counted as zero following upstream; requests and responses are retained in calls.jsonl."}
                    records.append(record)
                    write_json(args.output_dir / f"task-{task_id:02d}-trial-{trial}.json", record)
                    write_json(args.output_dir / "runs.json", records)
                    write_json(args.output_dir / "summary.json", run_summary(records, args.task_ids, args.trials, status, log))
                    print(f"task={task_id} trial={trial} reward={record['reward']} cost_estimate_usd={log.spent:.6f}", flush=True)
                    if status != "running":
                        raise RunStopped(status)
            status = "completed"
    except RunStopped as error:
        status = str(error)
    except KeyboardInterrupt:
        status = "interrupted_by_operator"
    except Exception as error:
        status = "setup_or_output_error"
        manifest["error_type"] = type(error).__name__
    finally:
        manifest.update(finished_at=now(), status=status)
        write_json(args.output_dir / "manifest.json", manifest)
        write_json(args.output_dir / "runs.json", records)
        write_json(args.output_dir / "summary.json", run_summary(records, args.task_ids, args.trials, status, log))
    return status


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "external/tau-bench")
    parser.add_argument("--output-dir", type=Path, help="Must not exist; default creates a unique results/live-* directory")
    parser.add_argument("--budget-usd", type=Decimal, required=True)
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(range(50)))
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--preflight", action="store_true", help="Check local source, dependencies and key presence; no API calls")
    args = parser.parse_args()
    if not args.budget_usd.is_finite() or args.budget_usd < CALL_RESERVE:
        parser.error(f"budget must be at least ${CALL_RESERVE} to reserve one possible call")
    if args.trials < 1 or len(set(args.task_ids)) != len(args.task_ids) or any(i not in range(50) for i in args.task_ids):
        parser.error("task IDs must be unique integers 0..49 and trials must be positive")
    args.benchmark_dir = args.benchmark_dir.resolve()
    args.output_dir = (args.output_dir or ROOT / "results" / f"live-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}").resolve()
    if args.output_dir.exists():
        parser.error("output directory already exists; refusing to overwrite")
    try:
        source = provenance(args.benchmark_dir, args.benchmark_dir / ARCHIVE_PATH)
        for package in ("litellm", "pydantic", "openai", "python-dotenv"):
            importlib.metadata.version(package)
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as error:
        parser.error(str(error))
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        parser.error("OPENAI_API_KEY is not configured; set it locally in .env or the environment")
    if args.preflight:
        print("Local preflight passed. No API request was made; model access and billing are unverified.")
        return
    status = execute(args, source)
    print(f"{status}: {args.output_dir}")
    if status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
