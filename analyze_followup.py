"""Compare archived, original fresh, and follow-up airline outcomes without API calls."""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/followup"


def load(path):
    return json.loads(path.read_text())


def task_counts(rows):
    counts = defaultdict(list)
    for r in rows:
        if not r["info"].get("interrupted"):
            counts[r["task_id"]].append(int(r["reward"] == 1))
    return counts


def describe(rows):
    valid = [r for r in rows if not r["info"].get("interrupted")]
    trials = defaultdict(list)
    for r in valid:
        trials[r["trial"]].append(int(r["reward"] == 1))
    return {
        "attempts": len(valid),
        "successes": sum(r["reward"] == 1 for r in valid),
        "trial_scores": {str(k): sum(v) / len(v) for k, v in sorted(trials.items())},
        "execution_errors": sum("error" in r["info"] for r in valid),
        "task_success_counts": {
            str(k): sum(v) for k, v in sorted(task_counts(valid).items())
        },
        "variable_task_ids": [
            k for k, v in sorted(task_counts(valid).items()) if 0 < sum(v) < len(v)
        ],
    }


def main():
    archive = load(
        ROOT / "external/tau-bench/historical_trajectories/gpt-4o-airline.json"
    )
    fresh = load(ROOT / "results/live-baseline/runs.json")
    a, b = task_counts(archive), task_counts(fresh)
    plan = load(OUT / "plan.json")
    ids = plan["task_ids"]
    diffs = [(sum(b[i]) - sum(a[i])) / 4 for i in range(50)]
    rng = random.Random(20261016)
    boot = sorted(sum(rng.choices(diffs, k=50)) / 50 for _ in range(20000))
    report = {
        "archive": describe(archive),
        "fresh": describe(fresh),
        "task_changes": dict(
            Counter(
                "improved" if d > 0 else "worsened" if d < 0 else "unchanged"
                for d in diffs
            )
        ),
        "paired_task_bootstrap": {
            "replicates": 20000,
            "seed": 20261016,
            "percentile_95_interval": [boot[500], boot[19499]],
            "interpretation": "Resamples paired task means. Describes uncertainty across tasks, not a causal explanation or a model of within-task generation variability.",
        },
        "selected_task_ids": ids,
        "archive_selected": describe([r for r in archive if r["task_id"] in ids]),
        "fresh_selected": describe([r for r in fresh if r["task_id"] in ids]),
    }
    report["contiguous_task_subsets"] = [
        {
            "task_ids": list(range(start, start + 10)),
            "archive": describe(
                [r for r in archive if start <= r["task_id"] < start + 10]
            ),
            "fresh": describe([r for r in fresh if start <= r["task_id"] < start + 10]),
        }
        for start in range(0, 50, 10)
    ]
    with (OUT / "task_comparison.csv").open("w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "task_id",
                "archive_successes_of_4",
                "fresh_successes_of_4",
                "change",
                "selected",
            ]
        )
        for i in range(50):
            writer.writerow([i, sum(a[i]), sum(b[i]), sum(b[i]) - sum(a[i]), i in ids])
    for name in ("original", "user-temp-zero"):
        p = OUT / name / "runs.json"
        if p.exists():
            report[name] = describe(load(p))
            report[name]["run_status"] = load(p.parent / "summary.json")["status"]
            report[name]["cost"] = load(p.parent / "summary.json")["cost"]
    if all(
        name in report and report[name]["run_status"] == "completed"
        for name in ("original", "user-temp-zero")
    ):
        left = report["original"]["task_success_counts"]
        right = report["user-temp-zero"]["task_success_counts"]
        report["temperature_comparison"] = {
            "score_difference_percentage_points": 100
            * (
                report["user-temp-zero"]["successes"] / 30
                - report["original"]["successes"] / 30
            ),
            "task_success_changes": {str(i): right[str(i)] - left[str(i)] for i in ids},
            "interpretation": "Exploratory comparison on ten tasks, three attempts per condition. Not a full benchmark score or proof of the historical gap's cause.",
        }
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
