"""Local tests with fake model responses. These are not model evaluation results."""

import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live import CALL_RESERVE, MODEL, CallLog, RunStopped, execute, openai_completion, provider_request, run_summary, usage_cost
from reproduce import ROOT


def response(usage=None, model=MODEL, tool=False):
    message = {"role": "assistant", "content": "test", "tool_calls": None}
    if tool:
        message["tool_calls"] = [{"id": "fake-tool", "type": "function", "function": {
            "name": "transfer_to_human_agents", "arguments": '{"summary":"local test"}'}}]
    data = {"id": "fake-response", "model": model, "system_fingerprint": "fake-fingerprint",
            "usage": usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 10},
            "choices": [{"message": message}]}
    return SimpleNamespace(model_dump=lambda **_: data,
                           choices=[SimpleNamespace(message=SimpleNamespace(
                               content=message["content"], model_dump=lambda: message))],
                           _hidden_params={"response_cost": 0.00035})


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.key = patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"})
        self.key.start()
        self.addCleanup(self.key.stop)

    def log(self, completion=lambda **_: response(), budget=1):
        return CallLog(self.path / "calls.jsonl", budget, completion)

    def call(self, log):
        return log.wrapper("user")(model=MODEL, custom_llm_provider="openai",
                                   messages=[{"role": "user", "content": "test"}])

    def events(self):
        return [json.loads(line) for line in (self.path / "calls.jsonl").read_text().splitlines()]

    def test_cached_tokens_are_discounted_and_missing_usage_rejected(self):
        self.assertEqual(usage_cost({"prompt_tokens": 100, "completion_tokens": 10,
                                    "prompt_tokens_details": {"cached_tokens": 80}}), Decimal("0.00025"))
        for usage in ({}, {"prompt_tokens": True, "completion_tokens": 0},
                      {"prompt_tokens": 1, "completion_tokens": 0,
                       "prompt_tokens_details": {"cached_tokens": 2}}):
            with self.assertRaises(ValueError):
                usage_cost(usage)

    def test_budget_blocks_before_dispatch(self):
        calls = []
        log = self.log(lambda **kwargs: calls.append(kwargs) or response(), CALL_RESERVE)
        self.call(log)
        with self.assertRaisesRegex(RunStopped, "budget_exhausted"):
            self.call(log)
        self.assertEqual(len(calls), 1)
        self.assertEqual(log.uncertain, 0)

    def test_api_error_retains_reservation_stops_and_redacts_exception(self):
        def failure(**_):
            raise RuntimeError("fake-test-key must never be logged")
        log = self.log(failure)
        for _ in range(2):
            with self.assertRaisesRegex(RunStopped, "api_interrupted"):
                self.call(log)
        self.assertEqual(log.count, 1)
        self.assertEqual(log.uncertain, CALL_RESERVE)
        self.assertNotIn("fake-test-key", log.path.read_text())

    def test_request_log_is_immutable_and_preserves_provider_defaults(self):
        captured = []
        log = self.log(lambda **kwargs: captured.append(kwargs) or response())
        messages = [{"role": "user", "content": "original"}]
        log.wrapper("user")(model=MODEL, custom_llm_provider="openai", messages=messages)
        messages[0]["content"] = "changed"
        request = self.events()[0]["request"]
        self.assertEqual(request["messages"][0]["content"], "original")
        self.assertNotIn("temperature", request)
        self.assertEqual(captured[0], request)
        self.assertNotIn("api_key", request)
        self.assertEqual(self.events()[1]["response"]["system_fingerprint"], "fake-fingerprint")

    def test_user_temperature_override_does_not_change_agent_request(self):
        captured = []
        log = CallLog(self.path / "calls.jsonl", 2, lambda **kw: captured.append(kw) or response(), 0)
        for role in ("user", "agent"):
            log.wrapper(role)(model=MODEL, custom_llm_provider="openai",
                              messages=[{"role": "user", "content": "test"}])
        self.assertEqual(captured[0]["temperature"], 0)
        self.assertNotIn("temperature", captured[1])
        self.assertNotIn("temperature", self.events()[0]["source_request"])

    def test_sdk_transport_has_no_retries_or_litellm_dispatch(self):
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        with patch("openai.OpenAI") as client_type:
            complete = openai_completion()
            self.assertEqual(client_type.call_args.kwargs["max_retries"], 0)
            self.assertEqual(client_type.call_args.kwargs["timeout"], 120)
            client_type.return_value.chat.completions.create.return_value = response()
            received = complete(model=MODEL, messages=[{"role": "user", "content": "test"}])
            client_type.return_value.chat.completions.create.assert_called_once()
            self.assertEqual(received.model, MODEL)

    def test_second_turn_api_payload_preserves_tools_and_removes_response_metadata(self):
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        from litellm import ModelResponse
        tool_calls = [{"id": "call_1", "type": "function", "function": {
            "name": "get_user_details", "arguments": '{"user_id":"test"}'}}]
        converted = ModelResponse(model=MODEL, choices=[{"index": 0, "message": {
            "role": "assistant", "content": None, "tool_calls": tool_calls,
            "annotations": [], "provider_specific_fields": {"extra": "metadata"}}}])
        source = {"model": MODEL, "custom_llm_provider": "openai", "temperature": 0,
                  "tools": [{"type": "function", "function": {"name": "get_user_details"}}],
                  "messages": [{"role": "system", "content": "policy"},
                               {"role": "user", "content": "task"},
                               converted.choices[0].message.model_dump(),
                               {"role": "tool", "tool_call_id": "call_1", "name": "get_user_details", "content": "result"}]}
        payload = provider_request(source)
        self.assertEqual(payload["messages"], [source["messages"][0], source["messages"][1],
                         {"role": "assistant", "tool_calls": tool_calls},
                         {"role": "tool", "tool_call_id": "call_1", "content": "result"}])
        self.assertEqual(payload["tools"], source["tools"])
        self.assertEqual(payload["temperature"], 0)

    def test_missing_usage_retains_reservation(self):
        log = self.log(lambda **_: response(usage={}))
        with self.assertRaisesRegex(RunStopped, "usage_unknown"):
            self.call(log)
        self.assertEqual(log.uncertain, CALL_RESERVE)

    def test_model_mismatch_stops_and_keeps_cost_uncertain(self):
        log = self.log(lambda **_: response(model="different-model"))
        with self.assertRaisesRegex(RunStopped, "unexpected_model"):
            self.call(log)
        self.assertEqual(log.spent, 0)
        self.assertEqual(log.uncertain, CALL_RESERVE)

    def test_pilot_and_interrupted_grid_do_not_claim_baseline_score(self):
        log = self.log()
        record = {"task_id": 0, "trial": 0, "reward": 0, "info": {}, "traj": []}
        pilot = run_summary([record], [0], 1, "completed", log)
        self.assertTrue(pilot["selected_grid_complete"])
        self.assertFalse(pilot["full_baseline_complete"])
        self.assertIsNone(pilot["pass_k"])
        runs = [dict(record, task_id=t, trial=r) for t in range(50) for r in range(4)]
        runs[-1]["info"] = {"interrupted": True}
        summary = run_summary(runs, list(range(50)), 4, "api_interrupted", log)
        self.assertFalse(summary["full_baseline_complete"])
        self.assertEqual(summary["finished_runs"], 199)
        self.assertIsNone(summary["pass_k"])

    def test_real_upstream_control_flow_with_fake_responses(self):
        args = SimpleNamespace(benchmark_dir=ROOT / "external/tau-bench",
                               output_dir=self.path / "run", budget_usd=1,
                               task_ids=[0], trials=1)
        status = execute(args, {"local_test": True}, lambda **kw: response(tool="tools" in kw))
        self.assertEqual(status, "completed")
        events = [json.loads(line) for line in (args.output_dir / "calls.jsonl").read_text().splitlines()]
        calls = [e for e in events if e["event"] == "request"]
        self.assertEqual([e["role"] for e in calls], ["user", "user", "user", "agent"])
        self.assertEqual(calls[0]["phase"], "setup")
        self.assertEqual(calls[-1]["request"]["temperature"], 0)
        self.assertIn("tools", calls[-1]["request"])
        saved = json.loads((args.output_dir / "runs.json").read_text())
        self.assertEqual(saved[0]["traj"][-1]["content"], "Transfer successful")
        self.assertEqual(saved[0]["reward"], 0)
        with self.assertRaises(FileExistsError):
            execute(args, {}, lambda **_: self.fail("overwrite must never dispatch a call"))


if __name__ == "__main__":
    unittest.main()
