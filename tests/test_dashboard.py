import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.config import Settings, TierSettings
from app.storage.db import log_request
from dashboard_data import (
    filter_requests,
    format_cost,
    load_request_history,
    request_tier,
    served_tier_label,
    submit_in_replay_mode,
    submit_to_api,
    summarize_requests,
)


def settings_for(database_url: str) -> Settings:
    return Settings(
        provider_mode="live",
        db_url=database_url,
        log_prompts=False,
        tiers=[
            TierSettings(
                id=0,
                name="local",
                provider="ollama",
                model_id="qwen-local",
                input_usd_per_million=0,
                output_usd_per_million=0,
            ),
            TierSettings(
                id=1,
                name="cheap",
                provider="groq",
                model_id="groq-cheap",
                input_usd_per_million=1,
                output_usd_per_million=2,
            ),
            TierSettings(
                id=2,
                name="strong",
                provider="groq",
                model_id="groq-strong",
                input_usd_per_million=3,
                output_usd_per_million=4,
            ),
        ],
    )


def add_request(
    settings: Settings,
    *,
    question: str,
    tier: int,
    score: int,
    latency: int,
    fallback: bool = False,
    overridden: bool = False,
    cost: float | None = 0.0,
    baseline: float | None = 0.0,
) -> int:
    chosen_tier = settings.tier(tier)
    return log_request(
        question=question,
        score=score,
        reasons=["reasoning keyword"],
        routed_to="strong" if score >= 3 else "local",
        model=chosen_tier.model_id,
        fallback=fallback,
        fallback_reason="HTTP 503" if fallback else None,
        overridden=overridden,
        input_tokens=10,
        output_tokens=5,
        token_source="estimated",
        latency_ms=latency,
        cost_usd=cost,
        baseline_cost_usd=baseline,
        settings=settings,
    )


class DashboardDataTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tempdir.name) / "requests.db"
        self.settings = settings_for(f"sqlite:///{db_path}")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_served_tier_is_unavailable_when_either_field_is_missing(self):
        self.assertEqual(
            served_tier_label({"tier_id": 2, "model_tier": "strong"}),
            "tier 2 · strong",
        )
        self.assertEqual(served_tier_label({"model_tier": "strong"}), "Unavailable")
        self.assertEqual(served_tier_label({"tier_id": 2}), "Unavailable")

    def test_missing_database_reads_empty_without_creating_it(self):
        records = load_request_history(self.settings)
        self.assertEqual(records, [])
        db_path = Path(self.tempdir.name) / "requests.db"
        self.assertFalse(db_path.exists())

    def test_reads_request_table_and_decodes_reasons(self):
        request_id = add_request(
            self.settings,
            question="Explain this",
            tier=1,
            score=4,
            latency=250,
        )
        rows = load_request_history(self.settings)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_id"], request_id)
        self.assertEqual(rows[0]["reasons"], ["reasoning keyword"])
        self.assertIsNone(rows[0]["question_text"])

    def test_summary_shares_fallbacks_cost_and_nearest_rank_latencies(self):
        add_request(
            self.settings,
            question="one",
            tier=0,
            score=1,
            latency=100,
            cost=0,
            baseline=0.0001,
        )
        add_request(
            self.settings,
            question="two",
            tier=1,
            score=3,
            latency=200,
            fallback=True,
            cost=0.00002,
            baseline=0.0001,
        )
        add_request(
            self.settings,
            question="three",
            tier=2,
            score=5,
            latency=300,
            overridden=True,
            cost=0.00003,
            baseline=0.00003,
        )
        add_request(
            self.settings,
            question="four",
            tier=2,
            score=6,
            latency=400,
            cost=0.00004,
            baseline=0.00004,
        )

        summary = summarize_requests(load_request_history(self.settings), self.settings)
        self.assertEqual(summary["requests"], 4)
        self.assertEqual(summary["tier_counts"], {0: 1, 1: 1, 2: 2})
        self.assertEqual(summary["tier_shares"], {0: 0.25, 1: 0.25, 2: 0.5})
        self.assertEqual(summary["fallbacks"], 1)
        self.assertEqual(summary["overrides"], 1)
        self.assertAlmostEqual(summary["estimated_cost_usd"], 0.00009)
        self.assertAlmostEqual(summary["counterfactual_strong_cost_usd"], 0.00027)
        self.assertEqual(summary["p50_latency_ms"], 200)
        self.assertEqual(summary["p95_latency_ms"], 400)

    def test_missing_rates_or_token_costs_are_unavailable(self):
        add_request(
            self.settings,
            question="one",
            tier=1,
            score=4,
            latency=100,
            cost=None,
            baseline=None,
        )
        summary = summarize_requests(load_request_history(self.settings), self.settings)
        self.assertIsNone(summary["estimated_cost_usd"])
        self.assertIsNone(summary["counterfactual_strong_cost_usd"])
        self.assertEqual(format_cost(None), "Cost unavailable")
        self.assertIn("(estimated)", format_cost(0.0))

    def test_filters_by_tier_fallback_override_and_text(self):
        add_request(
            self.settings,
            question="math reasoning",
            tier=1,
            score=4,
            latency=100,
            fallback=True,
            overridden=True,
        )
        add_request(
            self.settings,
            question="short prompt",
            tier=0,
            score=1,
            latency=50,
        )
        rows = load_request_history(self.settings)
        filters = filter_requests(
            rows,
            tier="tier 1 · cheap",
            fallback="Yes",
            overridden="Yes",
            search="reasoning",
            settings=self.settings,
        )
        self.assertEqual(len(filters), 1)
        self.assertEqual(request_tier(filters[0], self.settings), "tier 1 · cheap")

    @patch("dashboard_data.requests.post")
    def test_api_submission_posts_one_authenticated_request(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {"answer": "api answer", "request_id": 7}
        post.return_value = response

        result = submit_to_api(
            "question",
            api_url="http://router:8000/",
            api_key="test-key",
            force_model="strong",
            threshold=4,
        )

        self.assertEqual(result["request_id"], 7)
        post.assert_called_once_with(
            "http://router:8000/v1/route",
            headers={"X-API-Key": "test-key"},
            json={
                "question": "question",
                "threshold": 4,
                "force_model": "strong",
            },
            timeout=180,
        )

    @patch("dashboard_data.requests.post")
    def test_api_submission_shows_safe_backend_error_message(self, post):
        response = Mock(status_code=503)
        response.json.return_value = {
            "error": {"code": "provider_error", "message": "Check provider configuration."}
        }
        post.return_value = response
        with self.assertRaisesRegex(RuntimeError, "Check provider configuration"):
            submit_to_api(
                "question", api_url="http://router", api_key="test-key",
                force_model=None, threshold=3,
            )

    @patch("dashboard_data.requests.post", side_effect=AssertionError("network used"))
    def test_replay_mode_uses_fake_provider_and_logs_without_network(self, post):
        result = submit_in_replay_mode(
            "question",
            settings=self.settings,
            force_model="local",
            threshold=4,
        )
        self.assertIn("deterministic fake", result["answer"])
        self.assertEqual(result["model"], "qwen-local")
        self.assertEqual(result["token_source"], "estimated")
        self.assertTrue(result["overridden"])
        self.assertEqual(len(load_request_history(self.settings)), 1)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
