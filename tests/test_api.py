import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api import app, _rate_events, _rate_lock
from app.config import Settings, TierSettings, get_settings
from app.providers.base import ProviderHTTPError, ProviderResponse


def api_settings(db_url: str, *, rate_limit: int = 60, api_key: str = "test-api-key"):
    return Settings(
        provider_mode="fake",
        db_url=db_url,
        log_prompts=False,
        router_api_key=api_key,
        api_rate_limit_per_minute=rate_limit,
        groq_api_key=None,
        ollama_base_url="http://127.0.0.1:11434",
        tiers=[
            TierSettings(id=0, name="local", provider="ollama", model_id="local-model", input_usd_per_million=0, output_usd_per_million=0),
            TierSettings(id=1, name="cheap", provider="groq", model_id="small", input_usd_per_million=None, output_usd_per_million=None),
            TierSettings(id=2, name="strong", provider="groq", model_id="large", input_usd_per_million=None, output_usd_per_million=None),
        ],
    )


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.settings = api_settings(f"sqlite:///{Path(self.tempdir.name) / 'api.db'}")
        app.dependency_overrides[get_settings] = lambda: self.settings
        with _rate_lock:
            _rate_events.clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.clear()
        with _rate_lock:
            _rate_events.clear()
        self.tempdir.cleanup()

    def auth_headers(self):
        return {"X-API-Key": "test-api-key"}

    def test_auth_required_and_errors_have_consistent_shape(self):
        response = self.client.post("/v1/route", json={"question": "hello"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(set(response.json()), {"error"})
        self.assertEqual(
            self.client.get("/v1/config", headers={"X-API-Key": "wrong"}).status_code,
            401,
        )

    def test_route_validation_and_local_override_with_fake_provider(self):
        bad = self.client.post(
            "/v1/route",
            headers=self.auth_headers(),
            json={"question": "hello", "force_model": "nope"},
        )
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(bad.json()["error"]["code"], "validation_error")

        response = self.client.post(
            "/v1/route",
            headers=self.auth_headers(),
            json={"question": "hello", "force_model": "local"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "local-model")
        self.assertTrue(payload["overridden"])
        self.assertFalse(payload["fallback"])
        self.assertIsInstance(payload["request_id"], int)
        self.assertEqual(payload["token_source"], "estimated")

    @patch("app.api.route_question")
    def test_successful_route_response_includes_served_tier_fields(self, route_question):
        route_question.return_value = {
            "answer": "mock answer",
            "model": "mock-model-v2",
            "model_tier": "strong",
            "tier_id": 2,
            "provider": "groq",
            "routed_to": "strong",
            "score": 5,
            "reasons": ["mock reason"],
            "fallback": False,
            "fallback_reason": None,
            "overridden": False,
            "provider_usage": {"input_tokens": 8, "output_tokens": 4},
            "token_source": "provider",
            "latency_ms": 120,
            "cost_usd": 0.01,
            "baseline_cost_usd": 0.01,
            "request_id": 17,
        }

        response = self.client.post(
            "/v1/route", headers=self.auth_headers(), json={"question": "hello"}
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_tier"], "strong")
        self.assertEqual(payload["tier_id"], 2)
        self.assertEqual(payload["provider"], "groq")
        route_question.assert_called_once()

    @patch("app.routing.router.ask_tier")
    def test_fallback_flags_and_reason_are_returned(self, ask_tier):
        ask_tier.side_effect = [
            ProviderHTTPError("HTTP 503", status_code=503),
            ProviderResponse(
                answer="fallback answer",
                input_tokens=4,
                output_tokens=2,
                token_source="provider",
                latency_ms=5,
                model_id="small",
                provider="groq",
            ),
        ]
        response = self.client.post(
            "/v1/route",
            headers=self.auth_headers(),
            json={"question": "a complex reasoning problem", "force_model": "strong"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["fallback"])
        self.assertIn("503", payload["fallback_reason"])
        self.assertTrue(payload["overridden"])
        self.assertEqual(ask_tier.call_count, 2)

    def test_rate_limit_returns_retry_after(self):
        self.settings.api_rate_limit_per_minute = 1
        first = self.client.get("/v1/config", headers=self.auth_headers())
        second = self.client.get("/v1/config", headers=self.auth_headers())
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Retry-After", second.headers)
        self.assertEqual(second.json()["error"]["code"], "rate_limited")

    @patch("app.api.requests.get", side_effect=requests.ConnectionError("offline"))
    def test_health_is_public_and_reports_unconfigured_services(self, get):
        settings = api_settings(self.settings.db_url, api_key="")
        settings.groq_api_key = None
        app.dependency_overrides[get_settings] = lambda: settings
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["status"], "ok",
        )
        self.assertFalse(response.json()["providers"]["ollama"]["reachable"])
        self.assertFalse(response.json()["providers"]["groq"]["credentials_configured"])
        get.assert_called_once()

    @patch("app.api.requests.get", side_effect=requests.ConnectionError("offline"))
    def test_health_reports_cloud_profile_without_ollama_health_failure(self, _get):
        self.settings.enabled_tiers = [1, 2]
        self.settings.deployment_profile = "cloud"
        self.settings.groq_api_key = SecretStr("test-groq-key")
        payload = self.client.get("/health").json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["deployment_profile"], "cloud")
        self.assertEqual(payload["enabled_routing_tiers"], [1, 2])
        self.assertTrue(payload["providers"]["groq"]["configured"])
        self.assertTrue(payload["providers"]["groq"]["credentials_configured"])
        self.assertTrue(payload["groq_key_configured"])
        self.assertFalse(payload["providers"]["ollama"]["reachable"])
        self.assertNotIn("test-groq-key", str(payload))

    @patch("app.api.requests.get", side_effect=requests.ConnectionError("offline"))
    def test_health_treats_empty_and_whitespace_groq_keys_as_unconfigured(self, _get):
        for secret in ("", "  \t\n"):
            with self.subTest(secret=repr(secret)):
                self.settings.groq_api_key = SecretStr(secret)
                payload = self.client.get("/health").json()
                self.assertFalse(payload["providers"]["groq"]["credentials_configured"])
                self.assertFalse(payload["providers"]["groq"]["configured"])
                self.assertFalse(payload["groq_key_configured"])

    def test_provider_failure_returns_safe_actionable_error(self):
        with patch("app.routing.router.ask_tier", side_effect=ProviderHTTPError("HTTP 401", 401)):
            response = self.client.post(
                "/v1/route", headers=self.auth_headers(),
                json={"question": "Explain and compare algorithms", "force_model": "strong"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("check the configured provider key", response.json()["error"]["message"])
        self.assertNotIn("HTTP 401", response.text)

    def test_manual_disabled_local_tier_returns_clear_client_error(self):
        self.settings.enabled_tiers = [1, 2]
        response = self.client.post(
            "/v1/route", headers=self.auth_headers(),
            json={"question": "hello", "force_model": "local"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "tier_disabled")
        self.assertIn("disabled", response.json()["error"]["message"])

    def test_cloud_cheap_failure_has_safe_error_when_no_fallback_is_enabled(self):
        self.settings.enabled_tiers = [1, 2]
        with patch("app.routing.router.ask_tier", side_effect=ProviderHTTPError("HTTP 503", 503)):
            response = self.client.post(
                "/v1/route", headers=self.auth_headers(),
                json={"question": "hello", "force_model": "cheap"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "provider_error")
        self.assertIn("temporarily unavailable", response.json()["error"]["message"])
        self.assertNotIn("HTTP 503", response.text)

    def test_config_only_returns_tier_ids_and_history_reads_logged_rows(self):
        config = self.client.get("/v1/config", headers=self.auth_headers())
        self.assertEqual(config.json(), {"tiers": [{"id": 0}, {"id": 1}, {"id": 2}]})
        routed = self.client.post(
            "/v1/route", headers=self.auth_headers(), json={"question": "hello"}
        )
        history = self.client.get("/v1/requests", headers=self.auth_headers())
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()["items"]), 1)
        self.assertEqual(history.json()["items"][0]["request_id"], routed.json()["request_id"])
        self.assertIsNone(history.json()["items"][0]["question_text"])


if __name__ == "__main__":
    unittest.main()
