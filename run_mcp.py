"""Run one upstream MCPMark task with a $2 API cap and retained evidence."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from live import API_BASE, MODEL, CallLog, openai_completion


def hashes(root):
    return {
        str(p.relative_to(root)): {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


async def run(args):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    sys.path.insert(0, str(args.upstream))
    import litellm
    from src.agents import MCPMarkAgent
    from src.agents.mcp import MCPStdioServer

    args.output.mkdir(parents=True, exist_ok=False)
    task = args.upstream / "tasks/filesystem/standard/file_property/size_classification"
    with zipfile.ZipFile(args.data) as z:
        for member in z.infolist():
            p = Path(member.filename)
            if p.parts[0] == "file_property" and ".." not in p.parts:
                z.extract(member, args.output)
    work = (args.output / "file_property").resolve()
    initial = hashes(work)

    def save(name, obj):
        (args.output / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")

    save("initial_files.json", initial)
    log = CallLog(args.output / "calls.jsonl", 2, openai_completion())
    log.context = {
        "phase": "mcp",
        "task_id": "file_property/size_classification",
        "trial": 0,
    }
    complete = log.wrapper("agent")

    async def completion(**kwargs):
        # Upstream uses 32768 output tokens, beyond this snapshot's 16384 limit.
        # This snapshot's default output limit is 16384; provider_request omits max_tokens.
        allowed = {
            k: kwargs[k] for k in ("messages", "temperature", "tools") if k in kwargs
        }
        return await asyncio.to_thread(
            complete, model=MODEL, custom_llm_provider="openai", **allowed
        )

    original_call = MCPStdioServer.call_tool

    async def observed_call(server, name, arguments):
        result = await original_call(server, name, arguments)
        with (args.output / "mcp_tools.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {"name": name, "arguments": arguments, "result": result},
                    default=str,
                )
                + "\n"
            )
        return result

    manifest = {
        "upstream_commit": (
            await asyncio.to_thread(
                subprocess.check_output,
                ["git", "rev-parse", "HEAD"],
                cwd=args.upstream,
                text=True,
            )
        ).strip(),
        "task": "filesystem/standard/file_property/size_classification",
        "model": MODEL,
        "agent": "upstream MCPMarkAgent",
        "temperature": 1.0,
        "trials": 1,
        "max_turns": 100,
        "mcp_server": "@modelcontextprotocol/server-filesystem@2025.12.18",
        "transport": "stdio",
        "budget_usd": 2,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "adaptations": [
            "Direct OpenAI SDK transport with no retries and cost reservation.",
            "Omit upstream max_tokens=32768 and non-OpenAI enforcer_mode/think_mode; snapshot limit 16384.",
            "Use upstream agent, task description and verifier directly; isolated extracted task directory instead of full multi-service pipeline.",
        ],
        "task_sha256": hashlib.sha256(
            (task / "description.md").read_bytes()
        ).hexdigest(),
        "verifier_sha256": hashlib.sha256(
            (task / "verify.py").read_bytes()
        ).hexdigest(),
    }
    save("manifest.json", manifest)
    instruction = (
        task / "description.md"
    ).read_text() + f"\n\nTest directory: {work}\n"
    agent = MCPMarkAgent(
        "openai/" + MODEL,
        os.environ["OPENAI_API_KEY"],
        API_BASE,
        "filesystem",
        timeout=1200,
        service_config={"test_directory": str(work)},
    )
    with (
        patch.object(litellm, "acompletion", completion),
        patch.object(MCPStdioServer, "call_tool", observed_call),
    ):
        result = await agent.execute(instruction, str(args.output / "tool_calls.txt"))
    save("agent_result.json", result)
    env = {**os.environ, "FILESYSTEM_TEST_DIR": str(work)}
    verified = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(task / "verify.py")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (args.output / "verification.txt").write_text(verified.stdout + verified.stderr)
    final = hashes(work)
    save("final_files.json", final)
    content_preserved = sorted(v["sha256"] for v in initial.values()) == sorted(
        v["sha256"] for v in final.values()
    )
    summary = {
        "verifier_pass": verified.returncode == 0,
        "agent_execution_success": result.get("success"),
        "turns": result.get("turn_count"),
        "cost": log.costs(),
        "all_file_contents_preserved": content_preserved,
    }
    save("summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env-file", type=Path, required=True)
    args = p.parse_args()
    load_dotenv(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")
    asyncio.run(run(args))
