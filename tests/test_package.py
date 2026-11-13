import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, TierSettings, get_settings
from app.cost import estimate_request_cost
from app.providers.base import ProviderHTTPError
from app.providers.fake import FakeProvider
from app.providers.groq import GroqProvider
from app.providers.ollama import OllamaProvider
from app.routing.router import route_question
from app.storage.db import RequestRecord, get_engine


def sample_settings() -> Settings:
    return Settings(
        provider_mode="fake",
        db_url="sqlite://",
        log_prompts=False,
        tiers=[
            TierSettings(
                id=0, name="local", provider="ollama", model_id="local-model",
                input_usd_per_million=0, output_usd_per_million=0,
            ),
            TierSettings(
                id=1, name="cheap", provider="groq", model_id="small",
                input_usd_per_million=None, output_usd_per_million=None,
            ),
            TierSettings(
                id=2, name="strong", provider="groq", model_id="large",
                input_usd_per_million=2, output_usd_per_million=3,
            ),
        ],
    )


class ProviderParsingTests(unittest.TestCase):
    @patch("app.providers.base.requests.post")
    @patch("app.providers.ollama.record_metrics")
    def test_ollama_parses_response_tokens_and_latency(self, record, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": "qwen2.5:1.5b",
            "response": "local response",
            "prompt_eval_count": 9,
            "eval_count": 5,
        }
        post.return_value = response
        provider = OllamaProvider(
            "http://localhost:11434", sleep=lambda _delay: None
        )
        result = provider.generate("question", "qwen2.5:1.5b")
        self.assertEqual(result.answer, "local response")
        self.assertEqual(result.input_tokens, 9)
        self.assertEqual(result.output_tokens, 5)
        self.assertEqual(result.token_source, "provider")
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(post.call_args.kwargs["json"]["stream"], False)
        self.assertEqual(post.call_args.kwargs["timeout"], 120)
        record.assert_called_once_with(result)

    @patch("app.providers.base.requests.post")
    @patch("app.providers.groq.record_metrics")
    def test_groq_parses_chat_completion_usage(self, record, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": "openai/gpt-oss-20b",
            "choices": [{"message": {"content": "hosted response"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
        post.return_value = response
        provider = GroqProvider("test-key", sleep=lambda _delay: None)
        result = provider.generate("question", "openai/gpt-oss-20b")
        self.assertEqual(result.answer, "hosted response")
        self.assertEqual((result.input_tokens, result.output_tokens), (12, 4))
        self.assertEqual(result.token_source, "provider")
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key"
        )
        record.assert_called_once_with(result)

    @patch("app.providers.base.requests.post")
    def test_retry_applies_only_to_429_and_5xx(self, post):
        retry_response = Mock(status_code=503)
        retry_response.json.return_value = {"response": "ok"}
        success_response = Mock(status_code=200)
        success_response.json.return_value = {"response": "ok"}
        post.side_effect = [retry_response, success_response]
        provider = OllamaProvider(
            "http://localhost:11434", sleep=lambda _delay: None
        )
        result = provider.generate("q", "local")
        self.assertEqual(result.answer, "ok")
        self.assertEqual(post.call_count, 2)

        post.reset_mock()
        post.side_effect = None
        post.return_value = Mock(status_code=401)
        with self.assertRaises(ProviderHTTPError):
            provider.generate("q", "local")
        self.assertEqual(post.call_count, 1)

        post.reset_mock()
        post.side_effect = None
        post.side_effect = requests.Timeout("slow")
        with self.assertRaises(Exception):
            provider.generate("q", "local")
        self.assertEqual(post.call_count, 1)

    def test_fake_provider_is_deterministic_and_marks_estimated_tokens(self):
        provider = FakeProvider()
        first = provider.generate("question", "local")
        second = provider.generate("question", "local")
        self.assertEqual(first.answer, second.answer)
        self.assertEqual(first.token_source, "estimated")


class CostAndStorageTests(unittest.TestCase):
    def test_cost_is_none_when_rates_are_unset(self):
        settings = sample_settings()
        estimate = estimate_request_cost(settings, 1, 20, 10)
        self.assertIsNone(estimate.cost_usd)
        self.assertEqual(estimate.baseline_cost_usd, 0.00007)
        self.assertIn("counterfactual", estimate.baseline_label)

    def test_local_cost_is_zero_even_when_local_rates_are_unset(self):
        settings = sample_settings()
        settings.tiers[0] = settings.tier(0).model_copy(
            update={"input_usd_per_million": None, "output_usd_per_million": None}
        )
        estimate = estimate_request_cost(settings, 0, 20, 10)
        self.assertEqual(estimate.cost_usd, 0.0)

    def test_one_database_row_per_routed_request_and_no_prompt_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = sample_settings().model_copy(
                update={"db_url": f"sqlite:///{Path(directory) / 'requests.db'}"}
            )
            result = route_question(
                "What is 2 + 2?", settings=settings, display=False
            )
            engine = get_engine(settings.db_url)
            with Session(engine) as session:
                count = session.scalar(
                    select(func.count()).select_from(RequestRecord)
                )
                row = session.scalars(select(RequestRecord)).one()
            self.assertEqual(count, 1)
            self.assertIsNone(row.question_text)
            self.assertEqual(len(row.question_hash), 64)
            self.assertEqual(row.model, result["model"])
            self.assertEqual(row.cache_hit, False)


class SettingsTests(unittest.TestCase):
    @patch.dict("os.environ", {"DB_URL": "sqlite://", "PROVIDER_MODE": "fake"})
    def test_settings_load_yaml_defaults_and_environment_overrides(self):
        settings = get_settings()
        self.assertEqual(settings.db_url, "sqlite://")
        self.assertEqual(settings.provider_mode, "fake")
        self.assertEqual([tier.id for tier in settings.tiers], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
