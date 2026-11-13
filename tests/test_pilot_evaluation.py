import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_pilot_manifest import REQUIRED_COUNTS, build_pilot_manifest
from scripts.evaluate_split_cache import evaluate_pilot


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def pilot_rows():
    rows = []
    for task, count in REQUIRED_COUNTS.items():
        checker, answer = {
            "math": ("numeric", "1000"),
            "mcq": ("letter", "B"),
            "extract": ("exact", "user@example.org"),
            "reformat": ("exact", '["red","blue"]'),
            "classify": ("contains", "billing"),
        }[task]
        for index in range(count):
            rows.append({
                "id": f"test-{task}-{index:03d}",
                "split": "test",
                "task_type": task,
                "checker": checker,
                "question": f"Question {index} for {task}",
                "answer": answer,
            })
    return rows


class PilotManifestTests(unittest.TestCase):
    def test_manifest_has_required_task_counts_is_reproducible_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "test.jsonl"
            output = root / "pilot_manifest.json"
            manifest_rows = pilot_rows()
            for task in ("math", "mcq", "extract", "reformat"):
                template = next(row for row in manifest_rows if row["task_type"] == task)
                for index in range(7):
                    manifest_rows.append({**template, "id": f"extra-{task}-{index}"})
            write_jsonl(dataset, manifest_rows)
            original = dataset.read_bytes()
            expected_hash = hashlib.sha256(original).hexdigest()

            first = build_pilot_manifest(dataset, output)
            first_bytes = output.read_bytes()
            second = build_pilot_manifest(dataset, output)

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, output.read_bytes())
            self.assertEqual(first["seed"], 506)
            self.assertEqual(first["sample_size"], 96)
            self.assertEqual(first["task_counts"], REQUIRED_COUNTS)
            self.assertEqual(first["source_test_sha256"], expected_hash)
            self.assertEqual(len({row["id"] for row in first["questions"]}), 96)
            self.assertEqual(dataset.read_bytes(), original)

    def test_manifest_rejects_wrong_classify_population(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "test.jsonl"
            rows = pilot_rows()[:-1]
            write_jsonl(dataset, rows)
            with self.assertRaisesRegex(ValueError, "exactly 16 classify"):
                build_pilot_manifest(dataset, root / "manifest.json")


class PilotEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset = self.root / "test.jsonl"
        self.manifest = self.root / "manifest.json"
        self.cache = self.root / "answers_test.json"
        self.rows = [
            {"id": "m", "split": "test", "task_type": "math", "checker": "numeric",
             "question": "sum", "answer": "1000"},
            {"id": "q", "split": "test", "task_type": "mcq", "checker": "letter",
             "question": "choice", "answer": "B"},
            {"id": "e", "split": "test", "task_type": "extract", "checker": "exact",
             "question": "email", "answer": "user@example.org"},
            {"id": "r", "split": "test", "task_type": "reformat", "checker": "exact",
             "question": "json", "answer": '["red","blue"]'},
            {"id": "c", "split": "test", "task_type": "classify", "checker": "contains",
             "question": "ticket", "answer": "billing"},
        ]
        full_rows = list(self.rows)
        per_task_needed = {"math": 20, "mcq": 20, "extract": 20, "reformat": 20, "classify": 16}
        for task, count in per_task_needed.items():
            template = next(row for row in self.rows if row["task_type"] == task)
            current_count = sum(row["task_type"] == task for row in full_rows)
            for index in range(count - current_count):
                full_rows.append({**template, "id": f"extra-{task}-{index}"})
        write_jsonl(self.dataset, full_rows)
        self.manifest.write_text(json.dumps({
            "split": "test", "seed": 506,
            "source_test_sha256": hashlib.sha256(self.dataset.read_bytes()).hexdigest(),
            "sample_size": len(full_rows),
            "task_counts": per_task_needed,
            "questions": [{"id": row["id"], "task_type": row["task_type"]} for row in full_rows],
        }), encoding="utf-8")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_all_checker_types_missing_incorrect_and_paired_results(self):
        cache = {
            "schema_version": 1, "split": "test", "entries": {},
            "failures": [{"id": "m", "tier": "strong", "error": "mock failure"}],
        }
        answers = {
            "local": {"m": "1,000", "q": "Answer: B", "e": "WRONG@example.org",
                      "r": '["red", "blue"]', "c": "billing"},
            "cheap": {"m": "999", "q": "A", "e": "user@example.org",
                      "r": '["red","blue"]', "c": "billing explanation"},
            "strong": {"q": "B", "r": '["red","blue"]', "c": "billing"},
        }
        for item in self.rows:
            cache["entries"][item["id"]] = {
                "id": item["id"], "question": item["question"], "answers": {},
            }
            for tier, tier_answers in answers.items():
                if item["id"] in tier_answers:
                    cache["entries"][item["id"]]["answers"][tier] = {
                        "answer": tier_answers[item["id"]], "tier": tier,
                        "seconds": 0.1, "input_tokens": 5, "output_tokens": 2,
                        "provider": "mock", "model_id": f"{tier}-mock",
                        "timestamp": "2026-01-01T00:00:00Z",
                    }
        self.cache.write_text(json.dumps(cache), encoding="utf-8")

        result = evaluate_pilot(self.manifest, self.dataset, self.cache)
        local = result["tier_results"]["local"]
        cheap = result["tier_results"]["cheap"]
        strong = result["tier_results"]["strong"]
        self.assertEqual((local["correct"], local["incorrect"], local["accuracy"]), (3, 2, 0.6))
        self.assertEqual((cheap["correct"], cheap["incorrect"], cheap["accuracy"]), (2, 3, 0.4))
        self.assertEqual(strong["correct"], 3)
        self.assertEqual(strong["missing_answers"], 92)
        self.assertEqual(strong["failed_answers"], 1)
        self.assertEqual(local["per_task"]["mcq"]["accuracy"], 1.0)
        self.assertEqual(local["metadata"]["latency_seconds"]["n"], 5)
        self.assertEqual(local["metadata"]["available_records"]["input_tokens"], 5)
        self.assertEqual(
            result["paired_comparisons"]["local_vs_cheap"]["paired_questions"], 5
        )
        pair = result["paired_comparisons"]["local_vs_cheap"]["correctness_contingency"]
        self.assertEqual(pair["both_correct"], 0)
        self.assertEqual(pair["left_only_correct"], 3)
        self.assertEqual(pair["right_only_correct"], 2)
        self.assertEqual(pair["both_incorrect"], 0)

    def test_malformed_answer_is_not_counted_as_incorrect_and_metadata_is_not_fabricated(self):
        cache = {"split": "test", "entries": {
            "m": {"id": "m", "question": "sum", "answers": {
                "local": {"answer": 1000, "seconds": "0.2", "provider": "", "input_tokens": -1}
            }},
        }, "failures": []}
        self.cache.write_text(json.dumps(cache), encoding="utf-8")
        result = evaluate_pilot(self.manifest, self.dataset, self.cache)
        local = result["tier_results"]["local"]
        self.assertEqual(local["successful"], 0)
        self.assertEqual(local["incorrect"], 0)
        self.assertEqual(local["missing_answers"], 96)
        self.assertEqual(len(local["malformed_records"]), 1)
        self.assertEqual(local["metadata"]["latency_seconds"]["n"], 0)
        self.assertFalse(result["cost"]["reported"])

    def test_cache_question_mismatch_is_rejected(self):
        cache = {"split": "test", "entries": {
            "m": {"id": "m", "question": "wrong question", "answers": {
                "local": {"answer": "1000"}
            }},
        }, "failures": []}
        self.cache.write_text(json.dumps(cache), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "question mismatch"):
            evaluate_pilot(self.manifest, self.dataset, self.cache)

    def test_stale_manifest_hash_is_rejected(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["source_test_sha256"] = "0" * 64
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        self.cache.write_text(json.dumps({"split": "test", "entries": {}, "failures": []}))
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            evaluate_pilot(self.manifest, self.dataset, self.cache)


if __name__ == "__main__":
    unittest.main()
