import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from app.config import Settings, TierSettings
from app.providers.base import ProviderHTTPError, ProviderResponse
from scripts.cache_store import (
    CacheConflictError,
    load_cache,
    save_cache,
    store_success,
)
from scripts.import_old_cache import import_old_cache, main as import_main
from scripts.run_cache import main as cache_main, prompt_for_item, run_cache


def settings_for(mode: str = "fake") -> Settings:
    return Settings(
        _env_file=None,
        provider_mode=mode,
        db_url="sqlite://",
        tiers=[
            TierSettings(
                id=0, name="local", provider="ollama", model_id="local-model",
                input_usd_per_million=0, output_usd_per_million=0,
            ),
            TierSettings(
                id=1, name="cheap", provider="groq", model_id="cheap-model",
                input_usd_per_million=1, output_usd_per_million=2,
            ),
            TierSettings(
                id=2, name="strong", provider="groq", model_id="strong-model",
                input_usd_per_million=3, output_usd_per_million=4,
            ),
        ],
    )


def dev_items():
    return [
        {
            "id": "dev-existing-000",
            "question": "First development question?",
            "answer": "4",
        },
        {
            "id": "dev-existing-001",
            "question": "Second development question?",
            "answer": "8",
        },
    ]


def dataset_items(split: str, count: int = 2):
    return [
        {
            "id": f"{split}-{index:03d}",
            "split": split,
            "task_type": "math",
            "question": f"Question {split} {index}?",
            "answer": str(index + 1),
            "checker": "numeric",
        }
        for index in range(count)
    ]


def write_split(data_dir: Path, split: str, items) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{split}.jsonl"
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in items),
        encoding="utf-8",
    )


class OldCacheImportTests(unittest.TestCase):
    def test_import_maps_tiers_and_preserves_answers_and_metadata(self):
        legacy = {
            "0|local": {
                "answer": "local exact answer",
                "seconds": 1.25,
                "input_tokens": 10,
                "provider": "ollama",
            },
            "0|strong": {
                "answer": "strong exact answer",
                "seconds": 2.5,
                "provider_usage": {"input_tokens": 20, "output_tokens": 5},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answers_dev.json"
            report = import_old_cache(legacy, dev_items(), output)
            result = json.loads(output.read_text(encoding="utf-8"))

        entry = result["entries"]["dev-existing-000"]
        self.assertEqual(entry["source_id"], "0")
        self.assertEqual(entry["question_id"], "0")
        self.assertEqual(entry["question"], dev_items()[0]["question"])
        self.assertEqual(entry["answers"]["local"]["tier"], "local")
        self.assertEqual(entry["answers"]["strong"]["tier"], "strong")
        self.assertTrue(entry["answers"]["local"]["imported"])
        self.assertTrue(entry["answers"]["strong"]["imported"])
        self.assertEqual(entry["answers"]["local"]["answer"], "local exact answer")
        self.assertEqual(entry["answers"]["local"]["seconds"], 1.25)
        self.assertEqual(entry["answers"]["local"]["input_tokens"], 10)
        self.assertEqual(
            entry["answers"]["strong"]["provider_usage"],
            {"input_tokens": 20, "output_tokens": 5},
        )
        self.assertEqual(report["imported"], 2)
        self.assertEqual(report["missing_count"], 2)
        self.assertEqual(report["unmatched_count"], 0)

    def test_missing_and_unmatched_entries_are_reported_without_fabrication(self):
        legacy = {
            "0|local": {"answer": "local"},
            "0|cheap": {"answer": "not a legacy tier"},
            "1|strong": {"seconds": 1.0},
            "99|local": {"answer": "outside dev"},
            "malformed": {"answer": "bad key"},
            "1|local": {
                "question": "not the development question",
                "answer": "mismatched",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answers_dev.json"
            report = import_old_cache(legacy, dev_items(), output)
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["imported"], 1)
        self.assertGreaterEqual(report["missing_count"], 3)
        self.assertEqual(report["unmatched_count"], 5)
        self.assertEqual(
            set(result["entries"]["dev-existing-000"]["answers"]),
            {"local"},
        )
        self.assertNotIn("dev-existing-001", result["entries"])

    def test_import_is_idempotent_and_keeps_existing_equal_answer(self):
        legacy = {"0|local": {"answer": "same answer", "seconds": 1}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answers_dev.json"
            first = import_old_cache(legacy, dev_items(), output)
            original = output.read_bytes()
            second = import_old_cache(legacy, dev_items(), output)
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(first["imported"], 1)
        self.assertEqual(second["imported"], 0)
        self.assertEqual(second["already_present"], 1)
        self.assertEqual(len(result["entries"]), 1)
        self.assertEqual(result["entries"]["dev-existing-000"]["answers"]["local"]["answer"], "same answer")
        self.assertNotEqual(original, b"")

    def test_different_existing_success_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answers_dev.json"
            cache = {
                "schema_version": 1,
                "split": "dev",
                "entries": {
                    "dev-existing-000": {
                        "id": "dev-existing-000",
                        "source_id": "0",
                        "question": dev_items()[0]["question"],
                        "answers": {"local": {"answer": "existing"}},
                    }
                },
                "failures": [],
            }
            save_cache(output, cache)
            before = output.read_bytes()
            with self.assertRaises(CacheConflictError):
                import_old_cache(
                    {"0|local": {"answer": "different"}},
                    dev_items(),
                    output,
                )
            self.assertEqual(output.read_bytes(), before)

    def test_original_cache_file_is_not_modified_by_cli_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = root / "answers_cache.json"
            old_payload = {"0|local": {"answer": "original", "seconds": 1.1}}
            old_path.write_text(json.dumps(old_payload), encoding="utf-8")
            benchmark_path = root / "benchmark_questions.json"
            benchmark_path.write_text(
                json.dumps(
                    [{"question": dev_items()[0]["question"], "correct_answer": "4"}]
                ),
                encoding="utf-8",
            )
            output_path = root / "cache" / "answers_dev.json"
            original = old_path.read_bytes()

            self.assertEqual(
                import_main(
                    [
                        "--source", str(old_path),
                        "--benchmark", str(benchmark_path),
                        "--data-dir", str(root / "data"),
                        "--output", str(output_path),
                    ]
                ),
                0,
            )
            self.assertEqual(old_path.read_bytes(), original)
            imported = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                imported["entries"]["dev-existing-000"]["answers"]["local"]["answer"],
                "original",
            )


class CacheGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.data_dir = self.root / "data"
        self.cache_dir = self.root / "cache"
        self.settings = settings_for()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_all_supported_tiers_use_configured_tier_and_model(self):
        for split in ("dev", "train_pool", "test"):
            write_split(self.data_dir, split, dataset_items(split, 1))
        for tier_name, expected_id, expected_model in (
            ("local", 0, "local-model"),
            ("cheap", 1, "cheap-model"),
            ("strong", 2, "strong-model"),
        ):
            write_split(self.data_dir, "dev", dataset_items("dev", 1))
            generated = ProviderResponse(
                answer=f"answer-{tier_name}",
                input_tokens=4,
                output_tokens=2,
                token_source="provider",
                latency_ms=17,
                model_id=expected_model,
                provider=self.settings.tier(expected_id).provider,
            )
            provider = Mock()
            provider.generate.return_value = generated
            factory = Mock(return_value=provider)
            result = run_cache(
                split="dev",
                tier_name=tier_name,
                settings=self.settings,
                data_dir=self.data_dir,
                cache_dir=self.cache_dir / tier_name,
                limit=1,
                request_interval=0,
                provider_factory=factory,
            )
            factory.assert_called_once_with(self.settings, self.settings.tier(expected_id))
            provider.generate.assert_called_once_with(
                "Question dev 0?\n\nSolve the problem, then return only the final number. "
                "Do not include a label, explanation, or other text.",
                expected_model,
            )
            record = json.loads(
                (self.cache_dir / tier_name / "answers_dev.json").read_text()
            )["entries"]["dev-000"]["answers"][tier_name]
            self.assertEqual(record["model"], expected_model)
            self.assertEqual(record["model_id"], expected_model)
            self.assertEqual(record["provider"], self.settings.tier(expected_id).provider)
            self.assertEqual(record["input_tokens"], 4)
            self.assertEqual(record["output_tokens"], 2)
            self.assertEqual(record["latency_ms"], 17)
            self.assertEqual(record["seconds"], 0.017)
            self.assertIsNotNone(datetime.fromisoformat(record["timestamp"]))
            self.assertEqual(result["succeeded"], 1)

    def test_prompts_request_only_the_output_for_each_task_checker(self):
        cases = {
            "exact": "Return only the requested output, with no label or extra text.",
            "contains": "Return only the requested output, with no label or extra text.",
            "letter": "Return only the single answer letter, with no explanation.",
            "numeric": (
                "Solve the problem, then return only the final number. "
                "Do not include a label, explanation, or other text."
            ),
        }
        for checker, instruction in cases.items():
            with self.subTest(checker=checker):
                prompt = prompt_for_item({"question": "Task?", "checker": checker})
                self.assertEqual(prompt, f"Task?\n\n{instruction}")

    def test_metadata_survives_cache_serialization_and_legacy_records_load(self):
        write_split(self.data_dir, "dev", dataset_items("dev", 1))
        provider = Mock()
        provider.generate.return_value = ProviderResponse(
            answer="4", input_tokens=3, output_tokens=1, token_source="provider",
            latency_ms=1200, model_id="local-model", provider="ollama",
        )
        run_cache(
            split="dev", tier_name="local", settings=self.settings,
            data_dir=self.data_dir, cache_dir=self.cache_dir, limit=1,
            request_interval=0, provider_factory=Mock(return_value=provider),
        )
        destination = self.cache_dir / "answers_dev.json"
        record = json.loads(destination.read_text())["entries"]["dev-000"]["answers"]["local"]
        self.assertEqual(record["answer"], "4")
        self.assertEqual(record["provider"], "ollama")
        self.assertEqual(record["model_id"], "local-model")
        self.assertEqual(record["tier"], "local")
        self.assertEqual(record["input_tokens"], 3)
        self.assertEqual(record["output_tokens"], 1)
        self.assertEqual(record["seconds"], 1.2)
        self.assertTrue(record["timestamp"])
        self.assertNotIn("correct", record)

        # Imported and pre-metadata records remain readable and are never rewritten
        # just to add fields that were unavailable at import time.
        legacy_path = self.cache_dir / "legacy.json"
        legacy_path.write_text(json.dumps({
            "split": "dev", "entries": {"old": {"answers": {"local": {"answer": "old"}}}}
        }))
        legacy = load_cache(legacy_path, "dev")
        self.assertEqual(legacy["entries"]["old"]["answers"]["local"]["answer"], "old")

    def test_resume_after_failure_and_skip_previously_cached_answers(self):
        items = dataset_items("dev", 2)
        write_split(self.data_dir, "dev", items)
        cache_file = self.cache_dir / "answers_dev.json"
        cache = load_cache(cache_file, "dev")
        store_success(
            cache,
            item_id=items[0]["id"],
            source_id=None,
            question=items[0]["question"],
            tier="local",
            record={"answer": "already cached"},
        )
        save_cache(cache_file, cache)

        first_provider = Mock()
        first_provider.generate.side_effect = RuntimeError("internal detail")
        first = run_cache(
            split="dev",
            tier_name="local",
            settings=self.settings,
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            limit=1,
            request_interval=0,
            provider_factory=Mock(return_value=first_provider),
        )
        self.assertEqual(first["skipped_successes"], 1)
        self.assertEqual(first["failed"], 1)
        self.assertEqual(first["status_after_run"]["failed"], 1)

        response = ProviderResponse(
            answer="resumed answer",
            input_tokens=3,
            output_tokens=4,
            token_source="provider",
            latency_ms=10,
            model_id="local-model",
            provider="ollama",
        )
        second_provider = Mock()
        second_provider.generate.return_value = response
        second = run_cache(
            split="dev",
            tier_name="local",
            settings=self.settings,
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            limit=1,
            request_interval=0,
            provider_factory=Mock(return_value=second_provider),
        )
        self.assertEqual(second["succeeded"], 1)
        self.assertEqual(second["status_after_run"]["successful"], 2)
        cache = load_cache(cache_file, "dev")
        self.assertEqual(cache["entries"]["dev-000"]["answers"]["local"]["answer"], "already cached")
        self.assertEqual(cache["entries"]["dev-001"]["answers"]["local"]["answer"], "resumed answer")
        self.assertEqual(len(cache["failures"]), 1)

    def _write_manifest(self, items, *, path=None, **overrides):
        dataset_file = self.data_dir / "test.jsonl"
        manifest_path = path or self.root / "pilot_manifest.json"
        manifest = {
            "split": "test",
            "source_test_sha256": hashlib.sha256(dataset_file.read_bytes()).hexdigest(),
            "questions": [
                {"id": item["id"], "task_type": item["task_type"]}
                for item in items
            ],
        }
        manifest.update(overrides)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_manifest_mode_collects_only_listed_ids(self):
        items = dataset_items("test", 4)
        write_split(self.data_dir, "test", items)
        manifest = self._write_manifest([items[1], items[3]])
        provider = Mock()
        provider.generate.side_effect = [
            ProviderResponse("answer-1", 1, 1, "provider", 10, "local-model", "ollama"),
            ProviderResponse("answer-3", 1, 1, "provider", 10, "local-model", "ollama"),
        ]
        factory = Mock(return_value=provider)

        report = run_cache(
            split="test", tier_name="local", settings=self.settings,
            data_dir=self.data_dir, cache_dir=self.cache_dir,
            manifest_path=manifest, request_interval=0, provider_factory=factory,
        )

        self.assertEqual(report["selected_questions"], 2)
        self.assertEqual(report["remaining_requests"], 2)
        self.assertEqual(report["scheduled_requests"], 2)
        self.assertEqual(provider.generate.call_count, 2)
        cache = load_cache(self.cache_dir / "answers_test.json", "test")
        self.assertEqual(set(cache["entries"]), {items[1]["id"], items[3]["id"]})

    def test_manifest_mode_rejects_invalid_ids_hash_split_and_duplicates_before_provider(self):
        items = dataset_items("test", 3)
        write_split(self.data_dir, "test", items)
        cases = [
            ({"questions": [{"id": items[0]["id"]}, {"id": items[0]["id"]}]}, "duplicate"),
            ({"questions": [{"id": "not-in-dataset"}]}, "not present"),
            ({"source_test_sha256": "0" * 64}, "hash does not match"),
            ({"split": "dev"}, "does not match requested split"),
        ]
        for index, (overrides, message) in enumerate(cases):
            with self.subTest(message=message):
                manifest = self._write_manifest(
                    [items[0]], path=self.root / f"bad-{index}.json", **overrides
                )
                factory = Mock()
                with self.assertRaisesRegex(ValueError, message):
                    run_cache(
                        split="test", tier_name="local", settings=self.settings,
                        data_dir=self.data_dir, cache_dir=self.cache_dir / str(index),
                        manifest_path=manifest, request_interval=0,
                        provider_factory=factory,
                    )
                factory.assert_not_called()

    def test_manifest_limit_reports_scope_and_preserves_existing_successes(self):
        items = dataset_items("test", 4)
        write_split(self.data_dir, "test", items)
        manifest = self._write_manifest([items[0], items[2], items[3]])
        cache_file = self.cache_dir / "answers_test.json"
        cache = load_cache(cache_file, "test")
        for item, answer in ((items[2], "original successful answer"),
                             (items[1], "outside manifest answer")):
            store_success(
                cache, item_id=item["id"], source_id=None,
                question=item["question"], tier="local", record={"answer": answer},
            )
        save_cache(cache_file, cache)

        provider = Mock()
        provider.generate.return_value = ProviderResponse(
            "generated", 1, 1, "provider", 10, "local-model", "ollama"
        )
        output = StringIO()
        with redirect_stdout(output):
            report = run_cache(
                split="test", tier_name="local", settings=self.settings,
                data_dir=self.data_dir, cache_dir=self.cache_dir,
                manifest_path=manifest, limit=1, request_interval=0,
                provider_factory=Mock(return_value=provider),
            )

        self.assertIn("3 questions; already cached successes: 1; remaining requests: 2", output.getvalue())
        self.assertEqual(report["selected_questions"], 3)
        self.assertEqual(report["already_cached_successes"], 1)
        self.assertEqual(report["remaining_requests"], 2)
        self.assertEqual(report["scheduled_requests"], 1)
        provider.generate.assert_called_once()
        result = load_cache(cache_file, "test")
        self.assertEqual(
            result["entries"][items[2]["id"]]["answers"]["local"]["answer"],
            "original successful answer",
        )
        self.assertEqual(
            result["entries"][items[1]["id"]]["answers"]["local"]["answer"],
            "outside manifest answer",
        )

    def test_cli_accepts_manifest_path(self):
        manifest = self.root / "pilot.json"
        manifest.write_text("{}", encoding="utf-8")
        with patch("scripts.run_cache.run_cache", return_value={"ok": True}) as run:
            self.assertEqual(cache_main([
                "--split", "test", "--tier", "local", "--manifest", str(manifest),
                "--data-dir", str(self.data_dir), "--cache-dir", str(self.cache_dir),
            ]), 0)
        self.assertEqual(run.call_args.kwargs["manifest_path"], manifest)

    def test_rate_limit_failure_records_failure_and_honors_retry_after(self):
        write_split(self.data_dir, "test", dataset_items("test", 1))
        error = ProviderHTTPError(
            "HTTP 429",
            status_code=429,
            retry_after_seconds=7,
        )
        provider = Mock()
        provider.generate.side_effect = error
        sleep = Mock()

        report = run_cache(
            split="test",
            tier_name="strong",
            settings=self.settings,
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            request_interval=0,
            sleep=sleep,
            provider_factory=Mock(return_value=provider),
        )

        sleep.assert_called_once_with(7.0)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["still_failed"], 1)
        cache = load_cache(self.cache_dir / "answers_test.json", "test")
        failure = cache["failures"][0]
        self.assertEqual(failure["status_code"], 429)
        self.assertEqual(failure["retry_after_seconds"], 7)
        self.assertEqual(failure["error"], "HTTP 429")
        self.assertEqual(failure["model"], "strong-model")
        self.assertEqual(failure["provider"], "groq")
        self.assertIsNone(failure["input_tokens"])
        self.assertNotIn("internal detail", failure["error"])

    def test_rate_limit_retry_after_header_is_parsed_by_provider(self):
        from app.providers.base import _parse_retry_after

        self.assertEqual(_parse_retry_after("2.5"), 2.5)
        self.assertIsNone(_parse_retry_after("not a delay"))

    @patch("app.providers.base.requests.post")
    @patch("app.providers.ollama.record_metrics")
    def test_http_provider_retry_waits_for_retry_after(self, _metrics, post):
        limited = Mock(status_code=429, headers={"Retry-After": "3"})
        success = Mock(status_code=200)
        success.json.return_value = {"response": "ok"}
        post.side_effect = [limited, success]
        from app.providers.ollama import OllamaProvider

        delays = []
        provider = OllamaProvider(
            "http://localhost:11434",
            retry_attempts=2,
            sleep=delays.append,
        )
        response = provider.generate("question", "local")
        self.assertEqual(response.answer, "ok")
        self.assertEqual(delays, [3.0])
        self.assertEqual(post.call_count, 2)

    @patch("app.providers.ollama.record_metrics")
    @patch("app.providers.ollama.OllamaProvider.post_json")
    def test_ollama_response_parses_answer_model_usage_and_latency(self, post, _metrics):
        post.return_value = {
            "response": "generated text",
            "model": "qwen2.5:1.5b",
            "prompt_eval_count": 11,
            "eval_count": 7,
        }
        from app.providers.ollama import OllamaProvider

        response = OllamaProvider("http://localhost:11434").generate(
            "prompt", "configured-model"
        )
        self.assertEqual(response.answer, "generated text")
        self.assertEqual(response.model_id, "qwen2.5:1.5b")
        self.assertEqual(response.provider, "ollama")
        self.assertEqual(response.input_tokens, 11)
        self.assertEqual(response.output_tokens, 7)
        self.assertGreaterEqual(response.latency_ms, 0)
        self.assertEqual(response.token_source, "provider")


if __name__ == "__main__":
    unittest.main()
