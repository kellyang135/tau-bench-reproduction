# tau-bench reproduction

The fresh baseline completed all 200 attempts: **98 passes (49%)**, versus the published **42%**. Estimated API cost was **$18.6045975 including the separate pilot**, within the $20 cap. No API or execution errors occurred.

All 200 fresh scores and 1,163 tool responses also matched local replay. The archived audit separately matched the authors' 84/200 passes (42%).

## Setup

Tested with Python 3.12.4. Run these commands from this directory.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-replay.txt
```

The pinned dependencies support both replay and fresh evaluation. Installation needs network access.

## Fresh evaluation

Set `OPENAI_API_KEY` in a local `.env` file using `.env.example` as its format. Keep `.env` private. Git ignores it, and the download ZIP excludes it.

Commands used for the pilot and baseline (use new output directories to rerun):

```sh
.venv/bin/python live.py --task-ids 0 --trials 1 --budget-usd 2 --output-dir results/live-pilot
```

The pilot used $0.09795, leaving $19.90205 for all 50 tasks, four trials each:

```sh
.venv/bin/python live.py --budget-usd 19.90205 --output-dir results/live-baseline
```

Each invocation has a separate budget. Subtract earlier spend when working within a total cap. The runner reserves $0.48384 before each call and stops if the remaining budget cannot cover it. Cost uses returned token counts and [GPT-4o list prices](https://developers.openai.com/api/docs/models/gpt-4o); it is an estimate, not a billing statement. An uncertain API outcome retains that reservation and stops the run.

Both roles request `gpt-4o-2024-08-06`. Agent temperature is 0; user temperature is omitted. The pinned agent and grader run unchanged, with direct OpenAI SDK transport and retries disabled. Source/configuration, API requests and responses, token costs, trajectories, and scores are saved in a new `results/live-*` directory. Existing output directories cannot be overwritten. Only a completed 50×4 run produces baseline pass^k metrics.

Use the cost in `summary.json`; the upstream `info.user_cost` field contains only the last simulated-user call. The pilot is excluded from baseline scores.

The committed request logs are compressed without changing their contents. To read them locally:

```sh
gzip -dk results/live-pilot/calls.jsonl.gz results/live-baseline/calls.jsonl.gz
```

Fresh runs write `calls.jsonl`; those raw logs are ignored by Git.

Add `--preflight` to check local setup without making an API call; this does not verify account access.

## Archived audit

Recalculate saved scores without dependencies or API access:

```sh
python3 reproduce.py
```

Replay one saved attempt:

```sh
.venv/bin/python replay.py --task-id 13 --trial 1
```

Replay all 200 attempts and run the code tests:

```sh
.venv/bin/python replay.py
.venv/bin/python -m unittest discover -s tests -v
```

Expected full replay: 200 matching scores and 1,164 matching tool responses. This includes failed attempts; it does not mean 200 successful tasks. Results are recorded in [results/replay.json](results/replay.json).

Replay makes no model calls. To preserve the delivered results, add `--output-dir path/to/new-results` to either offline script.

## Files

| File | Purpose |
| --- | --- |
| `live.py` | Run fresh agent/user interactions with a per-invocation spending cap. |
| `reproduce.py` | Check coverage and recalculate pass^k from saved scores. |
| `replay.py` | Execute saved tool arguments; compare responses and scores. |
| `results/live-baseline/` | Fresh configuration, 200 trajectories, API logs, costs, and scores. |
| `results/live-pilot/` | Separate one-task evaluation; reward 0, estimated cost $0.09795. |
| `results/live-replay/` | Local verification of all 200 fresh trajectories. |
| `results/live-audit.json` | Call accounting, combined cost, repeated-run metrics, and comparison. |
| `results/summary.json`, `per_task.csv` | Aggregate and per-task archived scores. |
| `results/replay.json`, `replay_runs.csv` | Replay comparisons and state hashes. |
| `results/smoke/` | Single-attempt replay results. |
| `tests/` | Scoring, replay, budget, and simulated-response integration checks. |
| `vendor-manifest.json` | Commit and checksums for bundled upstream files. |
| `requirements-replay.txt` | Pinned dependencies. |

## Source

[Upstream commit](https://github.com/sierra-research/tau-bench/tree/7362a1708d720a4d697d042f21fd8637ea144c3b): `7362a1708d720a4d697d042f21fd8637ea144c3b`.

`external/tau-bench` contains an unmodified subset of that archive-upload snapshot and Sierra's [MIT license](external/tau-bench/LICENSE). All three scripts verify these files against `vendor-manifest.json`.

The archive does not identify the exact model snapshots, full generation settings, generating code revision, or total historical API cost. Replay agreement does not establish those missing details or prove the grader is correct.
