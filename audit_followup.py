"""Audit saved API accounting, temperatures and tool replay; makes no model calls."""

import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from live import MODEL, usage_cost
from replay import replay_run

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/followup"


def audit_calls(directory):
    path = directory / "calls.jsonl"
    with path.open() if path.exists() else gzip.open(str(path) + ".gz", "rt") as source:
        events = [json.loads(line) for line in source]
    by_id = defaultdict(list)
    for event in events:
        if "call_id" in event:
            by_id[event["call_id"]].append(event)
    total = Decimal(0)
    temperatures = defaultdict(set)
    models = Counter()
    fingerprints = Counter()
    for records in by_id.values():
        assert [r["event"] for r in records] == ["request", "response", "cost"], (
            "Incomplete or repeated API call"
        )
        request, response, cost = records
        actual = usage_cost(response["response"]["usage"])
        assert actual == Decimal(str(cost["estimated_usd"])), "Cost mismatch"
        total += actual
        temperatures[request["role"]].add(
            str(request["request"].get("temperature", "omitted"))
        )
        models[response["response"]["model"]] += 1
        fingerprints[response["response"].get("system_fingerprint")] += 1
    summary = json.loads((directory / "summary.json").read_text())
    assert len(by_id) == summary["cost"]["api_calls_dispatched"]
    assert set(models) == {MODEL}
    assert total == Decimal(str(summary["cost"]["estimated_api_cost_usd"]))
    assert summary["cost"]["uncertain_api_cost_upper_bound_usd"] == 0
    return {
        "calls": len(by_id),
        "cost_usd": float(total),
        "models": dict(models),
        "temperatures": {k: sorted(v) for k, v in temperatures.items()},
        "system_fingerprints": dict(fingerprints),
    }


def main():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    sys.path.insert(0, str(ROOT / "external/tau-bench"))
    from tau_bench.envs import get_env
    from tau_bench.types import Action

    report = {}
    ids = json.loads((OUT / "plan.json").read_text())["task_ids"]
    for name in ("original", "user-temp-zero"):
        directory = OUT / name
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["status"] == "completed" and summary["execution_error_runs"] == 0
        rows = json.loads((directory / "runs.json").read_text())
        assert {(r["task_id"], r["trial"]) for r in rows} == {
            (i, t) for i in ids for t in range(3)
        }
        assert len(rows) == 30
        manifest = json.loads((directory / "manifest.json").read_text())
        assert (
            manifest["runner_sha256"]
            == hashlib.sha256((ROOT / "live.py").read_bytes()).hexdigest()
        )
        replayed = [replay_run(r, get_env, Action) for r in rows]
        assert all(
            r["reward_matches"]
            and r["task_targets_match"]
            and r["instruction_matches"]
            and r["ground_truth_hash_matches"] is not False
            and r["tool_observations_semantic_mismatch"] == 0
            for r in replayed
        )
        (directory / "replay.json").write_text(json.dumps(replayed, indent=2) + "\n")
        report[name] = {
            **audit_calls(directory),
            "reward_matches": len(replayed),
            "tool_responses_matched": sum(
                r["tool_observations_exact_match"] for r in replayed
            ),
        }
        assert report[name]["temperatures"]["agent"] == ["0"]
        assert report[name]["temperatures"]["user"] == (
            ["omitted"] if name == "original" else ["0.0"]
        )
    report["mcp"] = audit_calls(OUT / "mcp")
    report["total_additional_cost_usd"] = sum(
        Decimal(str(r["cost_usd"])) for r in report.values()
    )
    assert report["total_additional_cost_usd"] <= 10
    report["total_additional_cost_usd"] = float(report["total_additional_cost_usd"])
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
