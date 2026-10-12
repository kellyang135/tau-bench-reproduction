"""Replay control-flow tests; these test the wrapper, not benchmark accuracy.

The delivered results/replay.json is evidence from the real upstream environment.
The small environment here only detects unexpected grading or skipped actions.
"""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import replay_run

TASK = {"user_id": "test", "actions": [], "outputs": ["done"], "instruction": "test"}


@dataclass
class Action:
    name: str
    kwargs: dict


class RecordingEnvironment:
    def __init__(self):
        self.task = SimpleNamespace(outputs=["done"], model_dump=lambda: TASK.copy())
        self.actions = []
        self.tools_map = {}
        self.terminate_tools = ["transfer"]
        self.data = {}
        self.grade_calls = 0

    def get_data_hash(self):
        return "test-hash"

    def calculate_reward(self):
        self.grade_calls += 1
        return SimpleNamespace(reward=1.0, info=SimpleNamespace(model_dump=lambda: {"r_outputs": 1}))


def archived(messages, reward=0):
    return {"task_id": 0, "trial": 0, "reward": reward, "info": {"task": TASK}, "traj": messages}


class ReplayControlTests(unittest.TestCase):
    def setUp(self):
        self.env = RecordingEnvironment()

    def replay(self, messages, reward=0):
        return replay_run(archived(messages, reward), lambda *args, **kwargs: self.env, Action)

    def test_step_limit_failure_is_not_graded(self):
        messages = [{"role": "user", "content": "start"}]
        for _ in range(30):
            messages.extend([{"role": "assistant", "content": "done"},
                             {"role": "user", "content": "keep going"}])
        result = self.replay(messages)
        self.assertEqual(result["replayed_reward"], 0)
        self.assertEqual(result["termination_reason"], "step_limit")
        self.assertEqual(self.env.grade_calls, 0)

    def test_recorded_stop_triggers_grading_once(self):
        result = self.replay([{"role": "user", "content": "start"},
                              {"role": "assistant", "content": "done"},
                              {"role": "user", "content": "###STOP###"}], 1)
        self.assertEqual(result["termination_reason"], "user_stop")
        self.assertTrue(result["reward_matches"])
        self.assertEqual(self.env.grade_calls, 1)

    def test_initial_stop_is_not_a_terminal_response(self):
        with self.assertRaisesRegex(ValueError, "30-step"):
            self.replay([{"role": "user", "content": "###STOP###"}])

    def test_action_after_stop_rejected(self):
        with self.assertRaisesRegex(ValueError, "after termination"):
            self.replay([{"role": "assistant", "content": "done"},
                         {"role": "user", "content": "###STOP###"},
                         {"role": "assistant", "content": "another action"}])

    def test_repeated_call_id_pairs_by_position(self):
        messages = []
        for _ in range(2):
            messages.extend([
                {"role": "assistant", "tool_calls": [{"id": "same-id", "function": {
                    "name": "unknown", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "same-id", "name": "unknown",
                 "content": "Unknown action unknown"},
            ])
        messages.extend([{"role": "assistant", "content": "done"},
                         {"role": "user", "content": "###STOP###"}])
        result = self.replay(messages, 1)
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["tool_observations_semantic_mismatch"], 0)

    def test_malformed_message_rejected(self):
        with self.assertRaisesRegex(ValueError, "message must be an object"):
            self.replay([None])


if __name__ == "__main__":
    unittest.main()
