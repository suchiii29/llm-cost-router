import contextlib
import io
import json
import os
import unittest
from unittest.mock import Mock, mock_open, patch

import requests

from checker import extract_number
from model_adapter import (
    GroqAPIError,
    GroqConfigurationError,
    ModelResponse,
    ask_cheap_model,
    ask_strong_model,
    get_groq_api_key,
)
from model_config import ModelConfig, get_model_config
from evaluate import evaluate_cache
from router import route_question
from run_benchmark import run_benchmark, run_mode, select_sample_indices
from tracker import estimate_cost


CONFIG = ModelConfig(
    cheap_model_id="openai/gpt-oss-20b",
    strong_model_id="openai/gpt-oss-120b",
    cheap_input_usd_per_million=None,
    cheap_output_usd_per_million=None,
    strong_input_usd_per_million=None,
    strong_output_usd_per_million=None,
)


def response_for(model_id):
    return ModelResponse(
        answer="answer",
        model_id=model_id,
        provider="groq",
        latency_seconds=0.1,
        input_tokens=10,
        output_tokens=4,
        total_tokens=14,
    )


class GroqAdapterTests(unittest.TestCase):
    @patch("model_adapter.requests.post")
    @patch("model_adapter.load_dotenv")
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_fails_before_network_call(
        self,
        _load_dotenv,
        post,
    ):
        with self.assertRaises(GroqConfigurationError):
            get_groq_api_key()

        post.assert_not_called()

    @patch("model_adapter.record_metrics")
    @patch("model_adapter.get_groq_api_key", return_value="test-key")
    @patch("model_adapter.requests.post")
    @patch("model_adapter.get_model_config", return_value=CONFIG)
    def test_cheap_and_strong_use_configured_groq_model_ids(
        self,
        _config,
        post,
        _key,
        _record,
    ):
        api_response = Mock()
        api_response.json.return_value = {
            "model": "openai/gpt-oss-20b",
            "choices": [{"message": {"content": "answer"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        }
        post.return_value = api_response

        cheap = ask_cheap_model("question")
        strong_body = dict(api_response.json.return_value)
        strong_body["model"] = "openai/gpt-oss-120b"
        api_response.json.return_value = strong_body
        strong = ask_strong_model("question")

        self.assertEqual(cheap.model_id, CONFIG.cheap_model_id)
        self.assertEqual(strong.model_id, CONFIG.strong_model_id)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            post.call_args_list[0].kwargs["json"]["model"],
            CONFIG.cheap_model_id,
        )
        self.assertEqual(
            post.call_args_list[1].kwargs["json"]["model"],
            CONFIG.strong_model_id,
        )
        for call in post.call_args_list:
            self.assertEqual(
                call.kwargs["headers"]["Authorization"],
                "Bearer test-key",
            )
        self.assertEqual(cheap.input_tokens, 10)
        self.assertEqual(cheap.output_tokens, 4)

    @patch("model_adapter.requests.post")
    @patch("model_adapter.get_groq_api_key", return_value="test-key")
    def test_rate_limit_becomes_recoverable_adapter_error(self, _key, post):
        http_response = Mock()
        http_response.status_code = 429
        post.return_value.raise_for_status.side_effect = requests.HTTPError(
            response=http_response
        )

        with self.assertRaises(GroqAPIError) as raised:
            from model_adapter import ask_groq_model

            ask_groq_model("question", CONFIG.strong_model_id)

        self.assertEqual(raised.exception.status_code, 429)
        self.assertTrue(raised.exception.recoverable)

    @patch("model_adapter.get_groq_api_key", return_value="test-key")
    @patch("model_adapter.requests.post")
    def test_authentication_error_is_not_recoverable(self, post, _key):
        http_response = Mock()
        http_response.status_code = 401
        post.return_value.raise_for_status.side_effect = requests.HTTPError(
            response=http_response
        )

        from model_adapter import ask_groq_model

        with self.assertRaises(GroqAPIError) as raised:
            ask_groq_model("question", CONFIG.strong_model_id)

        self.assertFalse(raised.exception.recoverable)
        self.assertIn("same API key cannot recover", str(raised.exception))

    @patch("model_adapter.record_metrics")
    @patch("model_adapter.get_groq_api_key", return_value="test-key")
    @patch("model_adapter.requests.post")
    @patch("model_adapter.get_model_config", return_value=CONFIG)
    def test_adapter_does_not_log_answer_text(
        self,
        _config,
        post,
        _key,
        record,
    ):
        api_response = Mock()
        api_response.json.return_value = {
            "model": CONFIG.cheap_model_id,
            "choices": [{"message": {"content": "private response"}}],
            "usage": {},
        }
        post.return_value = api_response

        result = ask_cheap_model("question")

        self.assertEqual(result.answer, "private response")
        self.assertEqual(record.call_args.args[0], result)

    @patch("model_adapter.open", new_callable=mock_open, create=True)
    def test_metrics_file_does_not_contain_answer_text(self, open_file):
        from model_adapter import record_metrics

        record_metrics(response_for(CONFIG.cheap_model_id))
        logged_fields = json.loads(open_file().write.call_args.args[0])
        self.assertNotIn("answer", logged_fields)


class RouterTests(unittest.TestCase):
    @patch("router.get_model_config", return_value=CONFIG)
    @patch("router.ask_strong_model", return_value=response_for(CONFIG.strong_model_id))
    @patch("router.ask_cheap_model", return_value=response_for(CONFIG.cheap_model_id))
    def test_automatic_low_complexity_uses_cheap_model(
        self,
        cheap,
        strong,
        _config,
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            result = route_question("What is 2 + 2?")

        cheap.assert_called_once()
        strong.assert_not_called()
        self.assertEqual(result["routed_to"], "cheap")
        self.assertEqual(result["model"], CONFIG.cheap_model_id)

    @patch("router.get_model_config", return_value=CONFIG)
    @patch("router.ask_strong_model", return_value=response_for(CONFIG.strong_model_id))
    @patch("router.ask_cheap_model", return_value=response_for(CONFIG.cheap_model_id))
    def test_automatic_high_complexity_uses_strong_model(
        self,
        cheap,
        strong,
        _config,
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            result = route_question(
                "Explain and compare two Python algorithms with examples"
            )

        cheap.assert_not_called()
        strong.assert_called_once()
        self.assertEqual(result["routed_to"], "strong")
        self.assertEqual(result["model"], CONFIG.strong_model_id)

    @patch("router.get_model_config", return_value=CONFIG)
    @patch("router.ask_strong_model")
    @patch("router.ask_cheap_model", return_value=response_for(CONFIG.cheap_model_id))
    def test_manual_override_selects_cheap_groq_model(
        self,
        cheap,
        strong,
        _config,
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            result = route_question(
                "Explain and compare this Python algorithm",
                force_model="local",
            )

        cheap.assert_called_once()
        strong.assert_not_called()
        self.assertEqual(result["routed_to"], "strong")
        self.assertEqual(result["model"], CONFIG.cheap_model_id)
        self.assertTrue(result["overridden"])
        self.assertFalse(result["fallback"])

    @patch("router.get_model_config", return_value=CONFIG)
    @patch("router.ask_strong_model", return_value=response_for(CONFIG.strong_model_id))
    @patch("router.ask_cheap_model")
    def test_groq_override_selects_strong_model(
        self,
        cheap,
        strong,
        _config,
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            result = route_question("What is 2 + 2?", force_model="groq")

        cheap.assert_not_called()
        strong.assert_called_once()
        self.assertEqual(result["model"], CONFIG.strong_model_id)
        self.assertTrue(result["overridden"])

    @patch("router.get_model_config", return_value=CONFIG)
    @patch("router.ask_cheap_model", return_value=response_for(CONFIG.cheap_model_id))
    @patch(
        "router.ask_strong_model",
        side_effect=GroqAPIError("Groq returned HTTP 429", 429, True),
    )
    def test_recoverable_strong_api_error_falls_back_to_cheap(
        self,
        _strong,
        cheap,
        _config,
    ):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = route_question(
                "Explain and compare this Python algorithm"
            )

        cheap.assert_called_once()
        self.assertEqual(result["routed_to"], "strong")
        self.assertEqual(result["model"], CONFIG.cheap_model_id)
        self.assertEqual(result["model_tier"], "cheap")
        self.assertTrue(result["fallback"])
        self.assertEqual(result["fallback_reason"], "Groq returned HTTP 429")
        self.assertIn("[FALLBACK:", output.getvalue())
        self.assertEqual(
            [attempt["outcome"] for attempt in result["api_attempts"]],
            ["failure", "success"],
        )
        self.assertEqual(
            [attempt["model_id"] for attempt in result["api_attempts"]],
            [CONFIG.strong_model_id, CONFIG.cheap_model_id],
        )

    @patch("router.get_model_config", return_value=CONFIG)
    @patch("router.ask_cheap_model")
    @patch(
        "router.ask_strong_model",
        side_effect=GroqAPIError("Groq returned HTTP 401", 401, False),
    )
    def test_invalid_key_error_does_not_retry_same_provider(
        self,
        _strong,
        cheap,
        _config,
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(GroqAPIError):
                route_question("Explain and compare this Python algorithm")

        cheap.assert_not_called()

    @patch("router.get_model_config", return_value=CONFIG)
    @patch("router.ask_strong_model", side_effect=GroqConfigurationError("missing key"))
    @patch("router.ask_cheap_model")
    def test_missing_key_does_not_fallback(
        self,
        cheap,
        _strong,
        _config,
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(GroqConfigurationError):
                route_question("Explain and compare this Python algorithm")

        cheap.assert_not_called()


class MetricTests(unittest.TestCase):
    @patch(
        "tracker.get_model_config",
        return_value=ModelConfig(
            cheap_model_id="openai/gpt-oss-20b",
            strong_model_id="openai/gpt-oss-120b",
            cheap_input_usd_per_million=0.1,
            cheap_output_usd_per_million=0.2,
            strong_input_usd_per_million=0.3,
            strong_output_usd_per_million=0.4,
        ),
    )
    def test_cost_uses_reported_usage_when_available(self, _config):
        result = estimate_cost(
            "openai/gpt-oss-20b",
            "question words",
            "answer words",
            {"input_tokens": 100, "output_tokens": 50},
        )

        self.assertEqual(result["token_source"], "provider-reported")
        self.assertEqual(result["cost_input_tokens"], 100)
        self.assertEqual(result["cost_output_tokens"], 50)
        self.assertAlmostEqual(result["estimated_cost_usd"], 0.00002)

    def test_checker_extracts_final_answer_numbers(self):
        self.assertEqual(extract_number("Final answer: 160"), "160")
        self.assertEqual(extract_number("**Final answer:** $70,000"), "70000")

    @patch.dict(os.environ, {
        "GROQ_CHEAP_MODEL": "configured-cheap",
        "GROQ_STRONG_MODEL": "configured-strong",
        "GROQ_CHEAP_INPUT_USD_PER_MILLION": "",
        "GROQ_CHEAP_OUTPUT_USD_PER_MILLION": "",
        "GROQ_STRONG_INPUT_USD_PER_MILLION": "",
        "GROQ_STRONG_OUTPUT_USD_PER_MILLION": "",
    })
    def test_model_ids_can_be_configured_in_environment(self):
        config = get_model_config()

        self.assertEqual(config.cheap_model_id, "configured-cheap")
        self.assertEqual(config.strong_model_id, "configured-strong")

    @patch("tracker.get_model_config", return_value=CONFIG)
    def test_cost_is_unavailable_without_model_rates(self, _config):
        result = estimate_cost("openai/gpt-oss-20b", "q", "a")

        self.assertIsNone(result["estimated_cost_usd"])
        self.assertEqual(
            result["token_source"],
            "estimated (characters / 4)",
        )


class EvaluationTests(unittest.TestCase):
    @patch("evaluate.estimate_cost", return_value={"estimated_cost_usd": None})
    def test_missing_cache_pairs_are_reported_and_policies_share_questions(
        self,
        _estimate_cost,
    ):
        questions = [
            {"question": "What is one plus zero?", "correct_answer": "1"},
            {"question": "What is one plus one?", "correct_answer": "2"},
            {"question": "What is one plus two?", "correct_answer": "3"},
        ]

        def cache_entry(answer, model_id):
            return {
                "answer": f"Final answer: {answer}",
                "model_id": model_id,
                "prompt": "question",
                "provider_usage": {},
            }

        cache = {
            "0|openai/gpt-oss-20b": cache_entry("1", CONFIG.cheap_model_id),
            "0|openai/gpt-oss-120b": cache_entry("1", CONFIG.strong_model_id),
            "2|openai/gpt-oss-20b": cache_entry("3", CONFIG.cheap_model_id),
            "2|openai/gpt-oss-120b": cache_entry("3", CONFIG.strong_model_id),
        }

        result = evaluate_cache(questions, cache, CONFIG)

        self.assertEqual(result["dataset_total"], 3)
        self.assertEqual(result["evaluated"], 2)
        self.assertEqual(result["missing_question_indices"], [1])
        for policy in result["results"]:
            self.assertEqual(policy["evaluated"], 2)
            self.assertEqual(policy["question_indices"], [0, 2])


class LiveBenchmarkUnitTests(unittest.TestCase):
    def test_sample_selection_is_evenly_spaced_and_limited(self):
        self.assertEqual(
            select_sample_indices(60, 5),
            [0, 15, 30, 44, 59],
        )

    @patch("run_benchmark.ask_cheap_model")
    def test_mode_reports_provider_usage_and_configured_cost(
        self,
        cheap,
    ):
        cheap.side_effect = [
            ModelResponse(
                answer="Final answer: 1",
                model_id=CONFIG.cheap_model_id,
                provider="groq",
                latency_seconds=0.1,
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
            ),
            ModelResponse(
                answer="Final answer: 2",
                model_id=CONFIG.cheap_model_id,
                provider="groq",
                latency_seconds=0.1,
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
            ),
        ]
        priced_config = ModelConfig(
            cheap_model_id=CONFIG.cheap_model_id,
            strong_model_id=CONFIG.strong_model_id,
            cheap_input_usd_per_million=1.0,
            cheap_output_usd_per_million=1.0,
            strong_input_usd_per_million=2.0,
            strong_output_usd_per_million=2.0,
        )

        result = run_mode([
            {
                "question": "One plus zero",
                "correct_answer": "1",
                "question_index": 4,
            },
            {
                "question": "One plus one",
                "correct_answer": "2",
                "question_index": 8,
            },
        ], "always_cheap", priced_config)

        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["questions_attempted"], 2)
        self.assertEqual(result["successful_answers"], 2)
        self.assertEqual(result["input_tokens"], 2)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["model_ids"], [CONFIG.cheap_model_id])
        self.assertAlmostEqual(result["estimated_cost_usd"], 0.000004)

    @patch("run_benchmark.route_question")
    def test_fallback_cost_counts_success_and_marks_failed_attempt_unknown(
        self,
        route,
    ):
        priced_config = ModelConfig(
            cheap_model_id=CONFIG.cheap_model_id,
            strong_model_id=CONFIG.strong_model_id,
            cheap_input_usd_per_million=1.0,
            cheap_output_usd_per_million=1.0,
            strong_input_usd_per_million=2.0,
            strong_output_usd_per_million=2.0,
        )
        route.return_value = {
            "answer": "Final answer: 1",
            "model": CONFIG.cheap_model_id,
            "routed_to": "strong",
            "complexity": "HIGH",
            "score": 4,
            "reasons": ["reasoning"],
            "fallback": True,
            "api_attempts": [
                {
                    "model_id": CONFIG.strong_model_id,
                    "outcome": "failure",
                    "input_tokens": None,
                    "output_tokens": None,
                    "error_type": "GroqAPIError",
                    "status_code": 429,
                },
                {
                    "model_id": CONFIG.cheap_model_id,
                    "outcome": "success",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "latency_seconds": 0.2,
                },
            ],
        }

        result = run_mode([
            {
                "question": "Explain this algorithm",
                "correct_answer": "1",
                "question_index": 2,
            },
        ], "router", priced_config)

        self.assertEqual(result["api_attempt_count"], 2)
        self.assertEqual(result["failed_api_attempts"], 1)
        attempts = result["questions"][0]["attempts"]
        self.assertIsNone(attempts[0]["estimated_cost_usd"])
        self.assertAlmostEqual(attempts[1]["estimated_cost_usd"], 0.000015)
        self.assertIsNone(result["estimated_cost_usd"])
        self.assertEqual(result["input_tokens"], 10)
        self.assertEqual(result["output_tokens"], 5)

    @patch("run_benchmark.run_mode")
    def test_benchmark_runs_all_policies_on_identical_five_questions(
        self,
        run_mode_mock,
    ):
        from run_benchmark import MODES

        run_mode_mock.side_effect = [
            {
                "mode": mode,
                "questions": [
                    {
                        "question_index": index,
                        "success": True,
                        "correct": True,
                    }
                    for index in [0, 15, 30, 44, 59]
                ],
                "estimated_cost_usd": None,
            }
            for mode in MODES
        ]
        questions = [
            {"question": f"question {index}", "correct_answer": str(index)}
            for index in range(60)
        ]

        result = run_benchmark(questions, CONFIG)

        self.assertEqual(result["sample_indices"], [0, 15, 30, 44, 59])
        self.assertEqual(result["paired_question_indices"], [0, 15, 30, 44, 59])
        self.assertEqual(run_mode_mock.call_count, 3)
        for call in run_mode_mock.call_args_list:
            self.assertEqual(
                [item["question_index"] for item in call.args[0]],
                result["sample_indices"],
            )
        self.assertIsNone(result["cost_savings_percent"])


if __name__ == "__main__":
    unittest.main()
