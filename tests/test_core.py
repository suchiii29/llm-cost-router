import contextlib
import io
import json
import os
import unittest
from unittest.mock import Mock, mock_open, patch

import requests
from pydantic import SecretStr
from checker import extract_number
from evaluate import evaluate_cache
from app.config import Settings, TierSettings
from app.providers.base import ProviderHTTPError, ProviderResponse, ProviderTransportError
from app.providers.groq import GroqProvider
from model_adapter import (
    GroqAPIError,
    GroqConfigurationError,
    ModelResponse,
    ask_cheap_model,
    ask_strong_model,
    get_groq_api_key,
)
from model_config import ModelConfig, get_model_config
import router
from router import route_question
from run_benchmark import (
    _attempt_cost,
    run_benchmark,
    run_mode,
    select_sample_indices,
)
import tracker
from tracker import estimate_cost


CONFIG = ModelConfig(
    cheap_model_id="openai/gpt-oss-20b",
    strong_model_id="openai/gpt-oss-120b",
    cheap_input_usd_per_million=None,
    cheap_output_usd_per_million=None,
    strong_input_usd_per_million=None,
    strong_output_usd_per_million=None,
)


def app_settings(cheap_rates=(None, None), strong_rates=(None, None)):
    return Settings(
        _env_file=None,
        provider_mode="fake",
        db_url="sqlite://",
        tiers=[
            TierSettings(id=0, name="local", provider="ollama", model_id="qwen2.5:1.5b", input_usd_per_million=0, output_usd_per_million=0),
            TierSettings(
                id=1, name="cheap", provider="groq",
                model_id=CONFIG.cheap_model_id,
                input_usd_per_million=cheap_rates[0],
                output_usd_per_million=cheap_rates[1],
            ),
            TierSettings(
                id=2, name="strong", provider="groq",
                model_id=CONFIG.strong_model_id,
                input_usd_per_million=strong_rates[0],
                output_usd_per_million=strong_rates[1],
            ),
        ],
    )


APP_SETTINGS = app_settings()
APP_SETTINGS_RATED = app_settings((0.1, 0.2), (0.3, 0.4))


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


def provider_response_for(tier):
    return ProviderResponse(
        answer="answer",
        model_id=tier.model_id,
        provider=tier.provider,
        input_tokens=10,
        output_tokens=4,
        token_source="provider",
        latency_ms=100,
    )


class GroqAdapterTests(unittest.TestCase):
    @patch("model_adapter.load_dotenv")
    @patch("app.providers.base.requests.post")
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_fails_before_network_call(self, post, _load_dotenv):
        with self.assertRaises(GroqConfigurationError):
            get_groq_api_key()
        post.assert_not_called()

    @patch("app.providers.groq.record_metrics")
    @patch("model_adapter.get_groq_api_key", return_value="test-key")
    @patch("model_adapter.get_settings", return_value=APP_SETTINGS)
    @patch("app.providers.base.requests.post")
    def test_tier_models_use_configured_groq_ids(
        self, post, _settings, _key, _record
    ):
        response = Mock(status_code=200)
        response.json.side_effect = [
            {
                "model": CONFIG.cheap_model_id,
                "choices": [{"message": {"content": "cheap answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
            {
                "model": CONFIG.strong_model_id,
                "choices": [{"message": {"content": "strong answer"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5},
            },
        ]
        post.return_value = response
        cheap = ask_cheap_model("question")
        strong = ask_strong_model("question")
        self.assertEqual(cheap.model_id, CONFIG.cheap_model_id)
        self.assertEqual(strong.model_id, CONFIG.strong_model_id)
        self.assertEqual(post.call_count, 2)

    @patch("model_adapter.get_groq_api_key", return_value="test-key")
    @patch("model_adapter.get_settings", return_value=APP_SETTINGS)
    @patch("app.providers.base.requests.post")
    def test_http_429_is_recoverable(self, post, _settings, _key):
        post.return_value = Mock(status_code=429)
        from model_adapter import ask_groq_model
        with self.assertRaises(GroqAPIError) as raised:
            ask_groq_model("question", CONFIG.strong_model_id)
        self.assertEqual(raised.exception.status_code, 429)
        self.assertTrue(raised.exception.recoverable)
        self.assertEqual(post.call_count, 3)

    @patch("model_adapter.get_groq_api_key", return_value="test-key")
    @patch("model_adapter.get_settings", return_value=APP_SETTINGS)
    @patch("app.providers.base.requests.post")
    def test_authentication_error_is_not_retried(self, post, _settings, _key):
        post.return_value = Mock(status_code=401)
        from model_adapter import ask_groq_model
        with self.assertRaises(GroqAPIError) as raised:
            ask_groq_model("question", CONFIG.strong_model_id)
        self.assertFalse(raised.exception.recoverable)
        self.assertEqual(post.call_count, 1)

    def _assert_transport_failure_falls_back(self, transport_error):
        settings = app_settings()
        settings.provider_mode = "live"
        settings.groq_api_key = SecretStr("test-key")
        transport_errors = []
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": settings.tier(1).model_id,
            "choices": [{"message": {"content": "fallback answer"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }

        def generate_with_real_provider_adapter(question, tier, config):
            from app.providers import get_provider

            try:
                return get_provider(config, tier).generate(question, tier.model_id)
            except ProviderTransportError as error:
                transport_errors.append(error)
                raise

        with (
            patch(
                "app.providers.base.requests.post",
                side_effect=[transport_error, response],
            ) as post,
            patch("app.providers.groq.record_metrics"),
            patch("app.routing.router.log_request"),
            patch(
                "app.routing.router.ask_tier",
                side_effect=generate_with_real_provider_adapter,
            ),
        ):
            result = route_question(
                "Explain and compare two algorithms",
                display=False,
                threshold=0,
                settings=settings,
            )

        self.assertEqual(len(transport_errors), 1)
        self.assertIsInstance(transport_errors[0], ProviderTransportError)
        self.assertTrue(transport_errors[0].recoverable)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["tier_id"], 1)
        self.assertEqual(post.call_count, 2)

    def test_timeout_becomes_recoverable_transport_error_and_router_falls_back(self):
        self._assert_transport_failure_falls_back(requests.Timeout("timed out"))

    def test_connection_error_becomes_recoverable_transport_error_and_router_falls_back(self):
        self._assert_transport_failure_falls_back(
            requests.ConnectionError("connection failed")
        )

    @patch("app.providers.base.requests.post")
    def test_http_503_retries_exactly_configured_attempts_then_remains_recoverable(
        self, post
    ):
        post.return_value = Mock(status_code=503)
        provider = GroqProvider(
            "test-key", retry_attempts=4, sleep=lambda _delay: None
        )

        with self.assertRaises(ProviderHTTPError) as raised:
            provider.generate("question", CONFIG.strong_model_id)

        self.assertEqual(post.call_count, 4)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertTrue(raised.exception.recoverable)

    @patch("app.providers.base.requests.post")
    def test_http_400_is_not_retried_and_is_non_recoverable(self, post):
        post.return_value = Mock(status_code=400)
        provider = GroqProvider(
            "test-key", retry_attempts=4, sleep=lambda _delay: None
        )

        with self.assertRaises(ProviderHTTPError) as raised:
            provider.generate("question", CONFIG.strong_model_id)

        self.assertEqual(post.call_count, 1)
        self.assertEqual(raised.exception.status_code, 400)
        self.assertFalse(raised.exception.recoverable)

    @patch("app.providers.base.open", new_callable=mock_open, create=True)
    def test_metrics_file_does_not_contain_answer_text(self, open_file):
        from model_adapter import record_metrics
        record_metrics(response_for(CONFIG.cheap_model_id))
        logged_fields = json.loads(open_file().write.call_args.args[0])
        self.assertNotIn("answer", logged_fields)


class RouterTests(unittest.TestCase):
    @patch("app.routing.router.log_request")
    @patch("app.routing.router.get_settings", return_value=APP_SETTINGS)
    @patch("app.routing.router.ask_tier")
    def test_tier_mapping_low_and_high(self, ask, _settings, _log):
        ask.side_effect = lambda question, tier, _config: provider_response_for(tier)
        with contextlib.redirect_stdout(io.StringIO()):
            low = route_question("What is 2 + 2?")
            high = route_question("Explain and compare two Python algorithms")
        self.assertEqual(low["tier_id"], 0)
        self.assertEqual(low["provider"], "ollama")
        self.assertEqual(high["tier_id"], 2)
        self.assertEqual(high["provider"], "groq")
        self.assertEqual([call.args[1].id for call in ask.call_args_list], [0, 2])

    @patch("app.routing.router.log_request")
    @patch("app.routing.router.get_settings", return_value=APP_SETTINGS)
    @patch("app.routing.router.ask_tier")
    def test_fallback_order_is_tier_two_then_one_then_local(
        self, ask, _settings, _log
    ):
        local = APP_SETTINGS.tier(0)
        ask.side_effect = [
            GroqAPIError("429 rate limit", 429, True),
            GroqAPIError("503 unavailable", 503, True),
            provider_response_for(local),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            result = route_question("Explain and compare two Python algorithms")
        self.assertEqual([call.args[1].id for call in ask.call_args_list], [2, 1, 0])
        self.assertEqual(result["tier_id"], 0)
        self.assertTrue(result["fallback"])
        self.assertIn("429", result["fallback_reason"])
        self.assertIn("503", result["fallback_reason"])

    @patch("app.routing.router.log_request")
    @patch("app.routing.router.get_settings", return_value=APP_SETTINGS)
    @patch("app.routing.router.ask_tier")
    def test_authentication_failure_does_not_fallback(self, ask, _settings, _log):
        ask.side_effect = GroqAPIError("401 unauthorized", 401, False)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(GroqAPIError):
                route_question("Explain and compare two Python algorithms")
        ask.assert_called_once()

    @patch("app.routing.router.log_request")
    @patch("app.routing.router.get_settings", return_value=APP_SETTINGS)
    @patch("app.routing.router.ask_tier")
    def test_local_cheap_strong_and_groq_overrides(self, ask, _settings, _log):
        ask.side_effect = lambda question, tier, _config: provider_response_for(tier)
        with contextlib.redirect_stdout(io.StringIO()):
            local = route_question("What is 2 + 2?", force_model="local")
            cheap = route_question("What is 2 + 2?", force_model="cheap")
            strong = route_question("What is 2 + 2?", force_model="strong")
            groq = route_question("What is 2 + 2?", force_model="groq")
        self.assertEqual([local["tier_id"], cheap["tier_id"], strong["tier_id"], groq["tier_id"]], [0, 1, 2, 2])
        self.assertTrue(all(result["overridden"] for result in (local, cheap, strong, groq)))

    @patch("app.routing.router.log_request")
    @patch("app.routing.router.ask_tier")
    def test_cloud_auto_uses_only_enabled_groq_tiers(self, ask, _log):
        settings = app_settings()
        settings.enabled_tiers = [1, 2]
        ask.side_effect = lambda question, tier, _config: provider_response_for(tier)
        low = route_question("hello", display=False, settings=settings)
        high = route_question("Explain and compare two Python algorithms", display=False, settings=settings)
        self.assertEqual(low["tier_id"], 1)
        self.assertEqual(high["tier_id"], 2)

    @patch("app.routing.router.log_request")
    @patch("app.routing.router.ask_tier")
    def test_cloud_strong_falls_back_to_cheap(self, ask, _log):
        settings = app_settings()
        settings.enabled_tiers = [1, 2]
        ask.side_effect = [
            GroqAPIError("503 unavailable", 503, True),
            provider_response_for(settings.tier(1)),
        ]
        result = route_question("Explain and compare two Python algorithms", display=False, settings=settings)
        self.assertEqual([call.args[1].id for call in ask.call_args_list], [2, 1])
        self.assertEqual(result["tier_id"], 1)
        self.assertTrue(result["fallback"])

    @patch("app.routing.router.log_request")
    def test_cloud_local_override_is_clear_error(self, _log):
        settings = app_settings()
        settings.enabled_tiers = [1, 2]
        with self.assertRaisesRegex(ValueError, "local tier is disabled"):
            route_question("hello", force_model="local", display=False, settings=settings)

    @patch("app.routing.router.log_request")
    @patch("app.routing.router.ask_tier")
    def test_local_failure_escalates_to_enabled_cheap_tier(self, ask, _log):
        settings = app_settings()
        ask.side_effect = [
            GroqAPIError("connection unavailable", None, True),
            provider_response_for(settings.tier(1)),
        ]
        result = route_question("hello", display=False, settings=settings)
        self.assertEqual([call.args[1].id for call in ask.call_args_list], [0, 1])
        self.assertEqual(result["tier_id"], 1)
        self.assertTrue(result["fallback"])


class TrackerOverrideTests(unittest.TestCase):
    @patch("app.tracking.track_question")
    @patch(
        "builtins.input",
        side_effect=[
            "force local: one", "force cheap: two", "force strong: three",
            "force groq: four", "exit",
        ],
    )
    def test_tracker_cli_preserves_force_tier_prefixes(self, _input, track):
        tracker.main()
        self.assertEqual(
            [call.kwargs["force_model"] for call in track.call_args_list],
            ["local", "cheap", "strong", "groq"],
        )


class RouterOverrideTests(unittest.TestCase):
    @patch("app.routing.router.route_question")
    @patch(
        "builtins.input",
        side_effect=[
            "force local: one", "force cheap: two", "force strong: three",
            "force groq: four", "exit",
        ],
    )
    def test_router_cli_preserves_force_tier_prefixes(self, _input, route):
        router.main()
        self.assertEqual(
            [call.kwargs["force_model"] for call in route.call_args_list],
            ["local", "cheap", "strong", "strong"],
        )


class MetricTests(unittest.TestCase):
    @patch("app.tracking.get_settings", return_value=APP_SETTINGS_RATED)
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

    @patch("model_config.get_settings", side_effect=lambda: Settings(_env_file=None))
    @patch.dict(os.environ, {
        "GROQ_CHEAP_MODEL": "configured-cheap",
        "GROQ_STRONG_MODEL": "configured-strong",
        "GROQ_CHEAP_INPUT_USD_PER_MILLION": "",
        "GROQ_CHEAP_OUTPUT_USD_PER_MILLION": "",
        "GROQ_STRONG_INPUT_USD_PER_MILLION": "",
        "GROQ_STRONG_OUTPUT_USD_PER_MILLION": "",
    })
    def test_model_ids_can_be_configured_in_environment(self, _settings):
        config = get_model_config()

        self.assertEqual(config.cheap_model_id, "configured-cheap")
        self.assertEqual(config.strong_model_id, "configured-strong")
        self.assertEqual(
            [(tier.id, tier.provider) for tier in config.tiers],
            [(0, "ollama"), (1, "groq"), (2, "groq")],
        )

    @patch("model_config.get_settings", side_effect=lambda: Settings(_env_file=None))
    @patch.dict(os.environ, {
        "GROQ_CHEAP_MODEL": "",
        "GROQ_STRONG_MODEL": "",
        "OLLAMA_MODEL": "",
        "OLLAMA_BASE_URL": "",
    })
    def test_blank_model_placeholders_use_documented_defaults(self, _settings):
        config = get_model_config()
        self.assertEqual(config.ollama_model_id, "qwen2.5:1.5b")
        self.assertEqual(config.ollama_base_url, "http://localhost:11434")

    @patch("app.tracking.get_settings", return_value=APP_SETTINGS)
    def test_cost_is_unavailable_without_model_rates(self, _config):
        result = estimate_cost("openai/gpt-oss-20b", "q", "a")

        self.assertIsNone(result["estimated_cost_usd"])
        self.assertEqual(
            result["token_source"],
            "estimated (characters / 4)",
        )

    @patch("app.tracking.get_settings", return_value=APP_SETTINGS)
    def test_local_model_cost_is_zero(self, _config):
        result = estimate_cost(CONFIG.ollama_model_id, "question", "answer")
        self.assertEqual(result["estimated_cost_usd"], 0.0)

    def test_benchmark_local_model_cost_is_zero_without_token_usage(self):
        cost = _attempt_cost({
            "outcome": "success",
            "model_id": CONFIG.ollama_model_id,
            "input_tokens": None,
            "output_tokens": None,
        }, CONFIG)
        self.assertEqual(cost, 0.0)


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
