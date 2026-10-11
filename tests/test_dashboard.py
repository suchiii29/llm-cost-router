import json
import os
import tempfile
import unittest
from unittest.mock import patch

from dashboard_data import (
    build_dashboard_summary,
    load_api_metrics,
    load_routing_history,
    process_submission,
)
from model_config import ModelConfig


CONFIG = ModelConfig(
    cheap_model_id="openai/gpt-oss-20b",
    strong_model_id="openai/gpt-oss-120b",
    cheap_input_usd_per_million=0.1,
    cheap_output_usd_per_million=0.2,
    strong_input_usd_per_million=0.3,
    strong_output_usd_per_million=0.4,
)


class DashboardDataTests(unittest.TestCase):
    def test_missing_logs_load_as_empty_without_network_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "missing.csv")
            metrics_path = os.path.join(directory, "missing.jsonl")
            self.assertEqual(load_routing_history(csv_path), [])
            self.assertEqual(load_api_metrics(metrics_path), [])

    def test_loads_csv_and_jsonl_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "routing.csv")
            metrics_path = os.path.join(directory, "metrics.jsonl")
            with open(csv_path, "w", encoding="utf-8") as file:
                file.write(
                    "timestamp,question,complexity,model_tier,model_id,"
                    "latency_seconds,fallback,provider_input_tokens,"
                    "provider_output_tokens,estimated_input_tokens,"
                    "estimated_output_tokens\n"
                    "2026-01-01T10:00:00,Hello,LOW,cheap,"
                    "openai/gpt-oss-20b,0.8,False,10,5,10,5\n"
                )
            with open(metrics_path, "w", encoding="utf-8") as file:
                file.write(json.dumps({
                    "model_id": "openai/gpt-oss-20b",
                    "latency_seconds": 0.8,
                }) + "\n")

            self.assertEqual(len(load_routing_history(csv_path)), 1)
            self.assertEqual(len(load_api_metrics(metrics_path)), 1)

    @patch("dashboard_data.get_model_config", return_value=CONFIG)
    def test_summary_uses_current_rates_and_logged_token_usage(self, _config):
        rows = [{
            "model_tier": "cheap",
            "model_id": CONFIG.cheap_model_id,
            "latency_seconds": "2.0",
            "provider_input_tokens": "100",
            "provider_output_tokens": "50",
        }]

        summary = build_dashboard_summary(rows, [], CONFIG)

        self.assertEqual(summary["total_queries"], 1)
        self.assertEqual(summary["cheap_calls"], 1)
        self.assertEqual(summary["strong_calls"], 0)
        self.assertEqual(summary["average_latency_seconds"], 2.0)
        self.assertAlmostEqual(summary["estimated_cost_usd"], 0.00002)

    @patch(
        "dashboard_data.get_model_config",
        return_value=ModelConfig(
            cheap_model_id=CONFIG.cheap_model_id,
            strong_model_id=CONFIG.strong_model_id,
            cheap_input_usd_per_million=None,
            cheap_output_usd_per_million=None,
            strong_input_usd_per_million=None,
            strong_output_usd_per_million=None,
        ),
    )
    def test_missing_rates_are_not_reported_as_zero_cost(self, _config):
        rows = [{
            "model_tier": "cheap",
            "model_id": CONFIG.cheap_model_id,
            "provider_input_tokens": "100",
            "provider_output_tokens": "50",
            "estimated_input_tokens": "100",
            "estimated_output_tokens": "50",
        }]

        summary = build_dashboard_summary(rows, [], None)

        self.assertIsNone(summary["estimated_cost_usd"])

    def test_metrics_can_supply_performance_without_fake_query_history(self):
        metrics = [
            {
                "model_id": CONFIG.cheap_model_id,
                "latency_seconds": 1.0,
            },
            {
                "model_id": CONFIG.strong_model_id,
                "latency_seconds": 3.0,
            },
        ]

        summary = build_dashboard_summary([], metrics, CONFIG)

        self.assertEqual(summary["total_queries"], 0)
        self.assertEqual(summary["cheap_calls"], 1)
        self.assertEqual(summary["strong_calls"], 1)
        self.assertEqual(summary["average_latency_seconds"], 2.0)
        self.assertIn("may include fallback retries", summary["performance_source"])

    @patch("dashboard_data.get_model_config", return_value=CONFIG)
    @patch(
        "dashboard_data.estimate_cost",
        return_value={"estimated_cost_usd": None},
    )
    def test_submission_calls_tracker_once_and_returns_answer_model_and_fallback(
        self,
        _estimate,
        _config,
    ):
        result = {
            "question": "Explain an algorithm",
            "answer": "A mocked answer",
            "model": CONFIG.cheap_model_id,
            "complexity": "HIGH",
            "score": 4,
            "routed_to": "strong",
            "model_tier": "cheap",
            "fallback": True,
            "fallback_reason": "rate limited",
            "reasons": ["reasoning keyword"],
            "latency_seconds": 1.2,
            "provider_usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
        }
        tracker = unittest.mock.Mock(return_value=result)
        state = {}

        displayed = process_submission(
            state,
            1,
            "Explain an algorithm",
            tracker_function=tracker,
        )

        tracker.assert_called_once_with("Explain an algorithm", display=False)
        self.assertEqual(displayed["answer"], "A mocked answer")
        self.assertEqual(displayed["model"], CONFIG.cheap_model_id)
        self.assertTrue(displayed["fallback"])
        self.assertEqual(displayed["fallback_reason"], "rate limited")

    @patch("dashboard_data.get_model_config", return_value=CONFIG)
    @patch("dashboard_data.estimate_cost", return_value={"estimated_cost_usd": None})
    def test_rerun_does_not_submit_same_request_twice(self, _estimate, _config):
        tracker = unittest.mock.Mock(return_value={
            "question": "q",
            "answer": "a",
            "model": CONFIG.cheap_model_id,
            "provider_usage": {},
        })
        state = {}

        first = process_submission(state, 1, "q", tracker_function=tracker)
        duplicate = process_submission(state, 1, "q", tracker_function=tracker)

        self.assertIsNotNone(first)
        self.assertIsNone(duplicate)
        tracker.assert_called_once()

    def test_tracker_error_is_propagated_to_dashboard(self):
        tracker = unittest.mock.Mock(side_effect=RuntimeError("mock routing failure"))

        with self.assertRaisesRegex(RuntimeError, "mock routing failure"):
            process_submission({}, 1, "q", tracker_function=tracker)

        tracker.assert_called_once()


if __name__ == "__main__":
    unittest.main()
