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

## Follow-up experiments

The original-settings subset scored **13/30 (43.3%)**; setting simulated-user
temperature to zero scored **12/30 (40%)**. Both conditions varied across repeats.
The MCPMark attempt completed execution but failed its verifier because `.DS_Store`
was placed in the wrong size category. Total additional API cost was **$4.5417225**.
See `results/followup/analysis.json`, `audit.json`, and `mcp/verification.txt`.

`results/followup/plan.json` fixes ten randomly selected task IDs and two conditions:
three repeats with the original settings, then three with only simulated-user
temperature set to zero. Each condition has a $4 cap. These are subset results,
not new full-benchmark scores. The earlier 200-attempt baseline stays separate.

```sh
python3 analyze_followup.py
.venv/bin/python audit_followup.py
```

The first command compares saved results without API calls. The second audits
API token accounting and replays the follow-up actions with the pinned grader.
`results/followup/task_comparison.csv` includes all 50 tasks, including declines.

To run another comparison, use a new output directory for each condition:

```sh
.venv/bin/python live.py --task-ids 8 18 26 28 32 35 42 43 46 47 --trials 3 --budget-usd 4 --output-dir results/repeat-original
.venv/bin/python live.py --task-ids 8 18 26 28 32 35 42 43 46 47 --trials 3 --budget-usd 4 --user-temperature 0 --output-dir results/repeat-zero
```

### MCPMark task

`run_mcp.py` runs MCPMark's `file_property/size_classification` task through its
stdio MCP filesystem server, using the upstream agent and verifier. It uses a
separate environment and a $2 cap. This is a single-task evaluation with a capped
OpenAI SDK transport, not a reproduction of the MCPMark leaderboard. Adaptations,
source revision, data hash, file hashes, model requests, tool responses and the
verifier output are saved in `results/followup/mcp/`.

```sh
python3 -m venv .venv-mcp
.venv-mcp/bin/python -m pip install -r requirements-mcp.txt
git clone https://github.com/eval-sys/mcpmark.git /tmp/mcpmark
git -C /tmp/mcpmark checkout cd45b7f57923b9b3985467f5139927575f83141c
curl -fL https://storage.mcpmark.ai/filesystem/file_property.zip -o /tmp/file_property.zip
.venv-mcp/bin/python run_mcp.py --upstream /tmp/mcpmark --data /tmp/file_property.zip --output /tmp/mcp-result --env-file .env
```

Node.js and `npx` are required; the upstream agent requests
`@modelcontextprotocol/server-filesystem@2025.12.18`. The output directory must
not already exist. The task works on a fresh extraction of the published data.
The original binary files are not stored in this repository.

## Third-party material

See [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) for the licenses and
provenance of the benchmark source, mock data, and text retained in run logs.
