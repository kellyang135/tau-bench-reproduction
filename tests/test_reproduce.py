"""Tests for coverage and tau-bench's all-attempts-success metric."""

import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce import summarize, validate_records


def record(task, trial, reward, error=None):
    return {"task_id": task, "trial": trial, "reward": reward,
            "info": {"error": error} if error else {}, "traj": []}


class ScoringTests(unittest.TestCase):
    def setUp(self):
        # Task 0 succeeds in 3 of 4 attempts; task 1 succeeds in 1 of 4.
        self.runs = [record(task, trial, int(trial < successes))
                     for task, successes in enumerate((3, 1)) for trial in range(4)]

    def test_known_pass_k_is_all_k_successes(self):
        summary, rows = summarize(self.runs, 2, 4)
        self.assertEqual(summary["pass_k"], {"1": 0.5, "2": 0.25, "3": 0.125, "4": 0.0})
        self.assertEqual([row["successes"] for row in rows], [3, 1])

    def test_order_does_not_change_results(self):
        self.assertEqual(summarize(self.runs, 2, 4), summarize(list(reversed(self.runs)), 2, 4))

    def test_missing_run_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize(self.runs[:-1], 2, 4)

    def test_duplicate_run_rejected_even_if_total_count_is_right(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(self.runs[:-1] + [self.runs[0]], 2, 4)

    def test_failed_execution_counts_as_failed_attempt(self):
        self.runs[0] = record(0, 0, 0, "API failure")
        summary, _ = summarize(self.runs, 2, 4)
        self.assertEqual(summary["error_runs"], 1)
        self.assertEqual(summary["pass_k"]["1"], 3 / 8)
        self.assertEqual(summary["runs"], 8)

    def test_error_run_cannot_claim_success(self):
        self.runs[0] = record(0, 0, 1, "API failure")
        with self.assertRaisesRegex(ValueError, "error runs"):
            summarize(self.runs, 2, 4)

    def test_malformed_values(self):
        for key, invalid in (("reward", math.nan), ("reward", math.inf),
                             ("reward", True), ("reward", "1"), ("reward", -1),
                             ("task_id", True), ("task_id", 2), ("trial", 4),
                             ("info", []), ("traj", {})):
            with self.subTest(key=key, invalid=invalid):
                runs = copy.deepcopy(self.runs)
                runs[0][key] = invalid
                with self.assertRaises(ValueError):
                    summarize(runs, 2, 4)

    def test_nested_reward_inconsistency_rejected(self):
        self.runs[0]["info"]["reward_info"] = {"reward": 0}
        with self.assertRaisesRegex(ValueError, "disagrees"):
            summarize(self.runs, 2, 4)

    def test_reward_tolerance_matches_upstream(self):
        summary, _ = summarize([record(0, 0, 1 - 0.5e-6)], 1, 1)
        self.assertEqual(summary["pass_k"]["1"], 1)
        self.assertNotEqual(summary["mean_reward"], 1)

    def test_missing_costs_are_unknown(self):
        summary, _ = summarize(self.runs, 2, 4)
        self.assertIsNone(summary["cost"]["historical_total_cost_usd"])
        self.assertIsNone(summary["cost"]["sum_of_logged_user_cost_values"])

    def test_empty_or_zero_trial_input_rejected(self):
        for runs, tasks, trials in (([], 2, 4), (self.runs, 0, 4), (self.runs, 2, 0)):
            with self.assertRaises(ValueError):
                validate_records(runs, tasks, trials)


if __name__ == "__main__":
    unittest.main()
