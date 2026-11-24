import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings, TierSettings
from app.storage.db import log_request
from dashboard_data import (
    filter_requests,
    format_cost,
    load_api_config,
    load_api_request_history,
    load_request_history,
    merge_request_histories,
    missing_cost_rate_settings,
    request_tier,
    served_tier_label,
    settings_with_api_tiers,
    submit_in_replay_mode,
    submit_to_api,
    summarize_requests,
    unavailable_cost_explanation,
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
        token_source="provider",
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
        self.assertAlmostEqual(summary["estimated_savings_usd"], 0.00018)
        self.assertAlmostEqual(summary["savings_percentage"], 66.6666666667)

    def test_summary_handles_empty_and_unavailable_usage_without_fake_savings(self):
        empty = summarize_requests([], self.settings)
        self.assertIsNone(empty["measured_api_cost_usd"])
        self.assertEqual(empty["total_input_tokens"], 0)
        self.assertIsNone(empty["savings_percentage"])
        add_request(
            self.settings, question="unknown", tier=1, score=3, latency=0,
            cost=None, baseline=None,
        )
        summary = summarize_requests(load_request_history(self.settings), self.settings)
        self.assertIsNone(summary["estimated_savings_usd"])
        self.assertEqual(summary["unavailable_cost_requests"], 1)

    def test_remote_config_and_history_use_authenticated_backend_endpoints(self):
        config_response = Mock(status_code=200)
        config_response.json.return_value = {
            "tiers": [
                {
                    "id": tier.id,
                    "name": tier.name,
                    "provider": tier.provider,
                    "model_id": tier.model_id,
                    "input_usd_per_million": tier.input_usd_per_million,
                    "output_usd_per_million": tier.output_usd_per_million,
                }
                for tier in self.settings.tiers
            ]
        }
        history_response = Mock(status_code=200)
        history_response.json.side_effect = [
            {"items": [{"request_id": 4, "model": "groq-strong"}]},
            {"items": []},
        ]
        with patch(
            "dashboard_data.requests.get",
            side_effect=[config_response, history_response, history_response],
        ) as get:
            config = load_api_config(api_url="https://backend", api_key="router-key")
            backend_settings = settings_with_api_tiers(self.settings, config)
            rows = load_api_request_history(
                api_url="https://backend", api_key="router-key", page_size=1
            )

        self.assertEqual(backend_settings.tier(2).model_id, "groq-strong")
        self.assertEqual(backend_settings.tier(1).input_usd_per_million, 1)
        self.assertEqual(rows, [{"request_id": 4, "model": "groq-strong"}])
        self.assertEqual(get.call_count, 3)
        for call in get.call_args_list:
            self.assertEqual(call.kwargs["headers"], {"X-API-Key": "router-key"})

    @patch("dashboard_data.requests.get")
    def test_backend_history_errors_are_not_reported_as_empty(self, get):
        get.return_value = Mock(status_code=401)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            load_api_request_history(api_url="https://backend", api_key="wrong")

    @patch("dashboard_data.get_engine", side_effect=SQLAlchemyError("database detail"))
    def test_local_database_errors_have_safe_dashboard_message(self, _get_engine):
        settings = self.settings.model_copy(update={"db_url": "sqlite://"})
        with self.assertRaisesRegex(RuntimeError, "Could not read dashboard request history"):
            load_request_history(settings)

    def test_merging_backend_and_replay_rows_deduplicates_shared_database_entries(self):
        backend_row = {
            "request_id": 1,
            "created_at": "2026-11-14T12:00:00+00:00",
            "question_hash": "hash",
            "model": "groq-strong",
            "score": 4,
        }
        local_row = {
            **backend_row,
            "created_at": datetime.fromisoformat(backend_row["created_at"]),
        }
        rows = merge_request_histories([backend_row], [local_row])
        self.assertEqual(len(rows), 1)

    def test_missing_cost_rates_name_backend_environment_variables(self):
        settings = settings_for("sqlite://")
        settings.tiers[1] = settings.tier(1).model_copy(
            update={"input_usd_per_million": None}
        )
        settings.tiers[2] = settings.tier(2).model_copy(
            update={"output_usd_per_million": None}
        )
        self.assertEqual(
            missing_cost_rate_settings(settings),
            [
                "GROQ_CHEAP_INPUT_USD_PER_MILLION",
                "GROQ_STRONG_OUTPUT_USD_PER_MILLION",
            ],
        )

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

    def test_format_cost_labels_each_cost_status_explicitly(self):
        self.assertIn("(actual)", format_cost(0.1, "actual"))
        self.assertIn("(estimated)", format_cost(0.1, "estimated"))
        self.assertIn("(simulated)", format_cost(0.1, "simulated"))
        self.assertIn("hardware/electricity unmeasured", format_cost(0.0, "local"))
        self.assertEqual(format_cost(None, "unknown"), "Cost unavailable")
        self.assertIn("(unknown)", format_cost(0.1, "unknown"))

    def test_unavailable_cost_explanations_use_existing_status_metadata(self):
        self.assertEqual(
            unavailable_cost_explanation({
                "cost_status": "unknown",
                "input_tokens": None,
                "output_tokens": None,
            }),
            "Request cost incomplete because an attempt's usage is unknown.",
        )
        self.assertEqual(
            unavailable_cost_explanation({
                "cost_status": "unavailable_usage",
                "input_tokens": None,
                "output_tokens": None,
            }),
            "Provider usage information unavailable.",
        )
        self.assertEqual(
            unavailable_cost_explanation({
                "cost_status": "unavailable_pricing",
                "input_tokens": 10,
                "output_tokens": 5,
            }),
            "Model pricing is not configured.",
        )
        self.assertEqual(
            unavailable_cost_explanation({
                "cost_status": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "token_source": "provider",
            }),
            "Provider usage information unavailable.",
        )

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
        self.assertEqual(result["token_source"], "provider")
        self.assertEqual(result["cost_status"], "simulated")
        self.assertIsNotNone(result["cost_usd"])
        self.assertTrue(result["overridden"])
        self.assertEqual(len(load_request_history(self.settings)), 1)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
