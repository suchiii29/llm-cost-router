import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_training_labels import LabelBuildError, build_training_labels


def sample_items():
    return [
        {
            "id": "math-1",
            "split": "train_pool",
            "task_type": "math",
            "question": "What is 2 + 2?",
            "answer": "4",
            "checker": "numeric",
        },
        {
            "id": "mcq-1",
            "split": "train_pool",
            "task_type": "mcq",
            "question": "Choose the correct option.",
            "answer": "C",
            "checker": "letter",
        },
        {
            "id": "reformat-1",
            "split": "train_pool",
            "task_type": "reformat",
            "question": "Uppercase the note.",
            "answer": "SHIPMENT ARRIVES NEXT WEEK.",
            "checker": "exact",
        },
        {
            "id": "classify-1",
            "split": "train_pool",
            "task_type": "classify",
            "question": "Pick a category.",
            "answer": "technical",
            "checker": "contains",
        },
    ]


def sample_cache(items):
    answers = {
        "math-1": {"answer": "Final answer: 4", "model_id": "local-model",
                   "provider": "ollama", "seconds": 0.4, "timestamp": "2026-01-01T00:00:00+00:00"},
        "mcq-1": {"answer": "The answer is (C).", "model_id": "local-model",
                  "provider": "ollama", "seconds": 0.2, "timestamp": "2026-01-01T00:00:00+00:00"},
        "reformat-1": {"answer": "NOTE 0059: SHIPMENT ARRIVES NEXT WEEK."},
        "classify-1": {"answer": "technical", "model_id": "local-model",
                       "provider": "ollama", "latency_ms": 300,
                       "timestamp": "2026-01-01T00:00:00+00:00"},
    }
    return {
        "schema_version": 1,
        "split": "train_pool",
        "entries": {
            item["id"]: {
                "id": item["id"],
                "question": item["question"],
                "answers": {"local": answers[item["id"]]},
            }
            for item in items
        },
        "failures": [],
    }


class TrainingLabelBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_path = self.root / "train_pool.jsonl"
        self.cache_path = self.root / "answers_train_pool.json"
        self.output_path = self.root / "train_labels.jsonl"
        self.items = sample_items()
        self.write_inputs(self.items)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_inputs(self, items, cache=None):
        self.dataset_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
            encoding="utf-8",
        )
        payload = sample_cache(items) if cache is None else cache
        self.cache_path.write_text(json.dumps(payload), encoding="utf-8")

    def build(self, expected_count=4):
        return build_training_labels(
            dataset_path=self.dataset_path,
            cache_path=self.cache_path,
            output_path=self.output_path,
            expected_count=expected_count,
        )

    def test_all_checker_types_label_raw_answers_and_report_distribution(self):
        report = self.build()
        labels = [json.loads(line) for line in self.output_path.read_text().splitlines()]
        self.assertEqual(len(labels), 4)
        self.assertEqual([row["local_ok"] for row in labels], [1, 1, 0, 1])
        self.assertEqual(report["overall"], {
            "total": 4, "correct": 3, "incorrect": 1, "accuracy": 0.75,
        })
        self.assertEqual(report["by_task_type"]["math"]["accuracy"], 1.0)
        self.assertEqual(report["by_task_type"]["reformat"]["accuracy"], 0.0)
        self.assertEqual(report["checker_failure_count"], 1)
        self.assertEqual(report["checker_failures_by_type"], {"exact": 1})
        self.assertEqual(
            labels[2]["local_answer"], "NOTE 0059: SHIPMENT ARRIVES NEXT WEEK."
        )
        self.assertEqual(labels[2]["ground_truth_answer"], "SHIPMENT ARRIVES NEXT WEEK.")
        self.assertEqual(report["missing_metadata"], [
            {"id": "reformat-1", "fields": ["model_id", "provider", "seconds", "timestamp"]}
        ])
        self.assertEqual(labels[3]["seconds"], 0.3)

    def test_missing_answer_fails_validation_without_writing_partial_output(self):
        cache = sample_cache(self.items)
        del cache["entries"]["mcq-1"]["answers"]["local"]["answer"]
        self.write_inputs(self.items, cache)
        with self.assertRaisesRegex(LabelBuildError, "Missing local cache answers"):
            self.build()
        self.assertFalse(self.output_path.exists())

    def test_json_equivalence_is_diagnostic_only_and_unmatched_cache_is_reported(self):
        json_item = {
            "id": "json-1",
            "split": "train_pool",
            "task_type": "reformat",
            "question": "Return a JSON array.",
            "answer": '["amber","birch"]',
            "checker": "exact",
        }
        items = self.items + [json_item]
        cache = sample_cache(self.items)
        cache["entries"][json_item["id"]] = {
            "id": json_item["id"],
            "question": json_item["question"],
            "answers": {"local": {"answer": '```JSON\n["amber", "birch"]\n```'}},
        }
        cache["entries"]["simple-train-0064"] = {
            "id": "simple-train-0064",
            "answers": {"local": {"answer": "kept in cache"}},
        }
        self.write_inputs(items, cache)

        report = self.build(expected_count=5)
        labels = [json.loads(line) for line in self.output_path.read_text().splitlines()]
        json_label = next(label for label in labels if label["id"] == "json-1")
        self.assertEqual(json_label["local_ok"], 0)
        self.assertIs(json_label["semantic_json_equivalent"], True)
        self.assertEqual(report["semantic_json_diagnostic"], {
            "eligible": 1,
            "equivalent": 1,
            "not_equivalent": 0,
            "accuracy": 1.0,
        })
        self.assertEqual(report["unlabelled_cache_entries"], [{
            "id": "simple-train-0064",
            "reason": "no matching row in current train_pool",
        }])

    def test_non_json_rows_are_excluded_from_semantic_json_diagnostic(self):
        report = self.build()
        labels = [json.loads(line) for line in self.output_path.read_text().splitlines()]
        self.assertTrue(all(label["semantic_json_equivalent"] is None for label in labels))
        self.assertEqual(report["semantic_json_diagnostic"]["eligible"], 0)

    def test_duplicate_training_ids_are_rejected(self):
        duplicate = self.items + [dict(self.items[0])]
        self.write_inputs(duplicate)
        with self.assertRaisesRegex(LabelBuildError, "Duplicate training question ID"):
            self.build(expected_count=5)
        self.assertFalse(self.output_path.exists())

    def test_duplicate_cache_json_ids_are_rejected(self):
        cache_text = json.dumps(sample_cache(self.items))
        duplicate_entries = cache_text.replace(
            '"failures": []', '"entries": {}, "failures": []'
        )
        self.cache_path.write_text(duplicate_entries, encoding="utf-8")
        with self.assertRaisesRegex(LabelBuildError, "Duplicate JSON object key"):
            self.build()
        self.assertFalse(self.output_path.exists())

    def test_output_is_reproducible_and_inputs_remain_unchanged(self):
        dataset_before = self.dataset_path.read_bytes()
        cache_before = self.cache_path.read_bytes()
        self.build()
        first_output = self.output_path.read_bytes()
        self.build()
        self.assertEqual(self.output_path.read_bytes(), first_output)
        self.assertEqual(self.dataset_path.read_bytes(), dataset_before)
        self.assertEqual(self.cache_path.read_bytes(), cache_before)

    def test_wrong_expected_question_count_is_rejected(self):
        with self.assertRaisesRegex(LabelBuildError, "Expected 600 unique training questions"):
            self.build(expected_count=600)
        self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
