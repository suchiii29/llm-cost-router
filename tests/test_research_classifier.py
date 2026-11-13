import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_local_classifier import (
    _derive_dev_rows,
    _routing_results,
    feature_text,
    run_training,
)


class ResearchClassifierTests(unittest.TestCase):
    def test_feature_builder_uses_only_question_and_task_type(self):
        record = {
            "question": "How many apples?", "task_type": "math", "local_ok": 1,
            "ground_truth_answer": "42", "local_answer": "spoiler", "id": "secret",
            "provider": "ollama", "explanation": "leak",
        }
        original = feature_text(record)
        for key in ("local_ok", "ground_truth_answer", "local_answer", "id", "provider", "explanation"):
            record[key] = f"changed {key}"
        self.assertEqual(feature_text(record), original)
        self.assertIn("How many apples?", original)
        self.assertIn("math", original)

    def test_dev_labels_use_correct_checker_and_local_strong_cache_answers(self):
        dataset = [{"id": "d1", "split": "dev", "question": "2+2?",
                    "answer": "4", "task_type": "math", "checker": "numeric"}]
        cache = {"split": "dev", "entries": {"d1": {"answers": {
            "local": {"answer": "4"}, "strong": {"answer": "5"}}}}}
        rows = _derive_dev_rows(dataset, cache)
        self.assertEqual(rows[0]["local_ok"], 1)
        self.assertEqual(rows[0]["strong_ok"], 0)

    def test_routing_uses_probability_threshold_and_reports_call_shares(self):
        rows = [
            {"id": "a", "task_type": "math", "local_ok": 1, "strong_ok": 0},
            {"id": "b", "task_type": "extract", "local_ok": 0, "strong_ok": 1},
        ]
        result = _routing_results(rows, __import__("numpy").array([0.8, 0.2]))
        policies = result["policies"]
        self.assertEqual(policies["always_local"]["accuracy"], 0.5)
        self.assertEqual(policies["always_strong"]["accuracy"], 0.5)
        selective = policies["classifier_selective"]
        self.assertEqual(selective["accuracy"], 1.0)
        self.assertEqual(selective["local_call_share"], 0.5)
        self.assertEqual(selective["strong_call_share"], 0.5)

    def test_training_and_cv_metrics_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            rows = []
            for index in range(598):
                label = index % 2
                rows.append({"id": str(index), "question": f"query token{index % 30} type{label}",
                             "task_type": "math" if index % 2 else "extract",
                             "local_ok": label, "local_answer": "not a feature",
                             "ground_truth_answer": "also not a feature"})
            train_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            dev_path = root / "dev.jsonl"
            dev_rows = [{"id": "d1", "split": "dev", "question": "query token1 type1",
                         "answer": "1", "task_type": "math", "checker": "numeric"},
                        {"id": "d2", "split": "dev", "question": "query token2 type0",
                         "answer": "0", "task_type": "math", "checker": "numeric"}]
            dev_path.write_text("".join(json.dumps(row) + "\n" for row in dev_rows))
            cache_path = root / "cache.json"
            cache_path.write_text(json.dumps({"split": "dev", "entries": {
                "d1": {"answers": {"local": {"answer": "1"}, "strong": {"answer": "1"}}},
                "d2": {"answers": {"local": {"answer": "1"}, "strong": {"answer": "0"}}},
            }}))
            first = run_training(train_path, dev_path, cache_path,
                                 root / "m1.joblib", root / "r1.json")
            second = run_training(train_path, dev_path, cache_path,
                                  root / "m2.joblib", root / "r2.json")
            self.assertEqual(first["cross_validation"], second["cross_validation"])
            self.assertEqual(first["dev"], second["dev"])
            self.assertTrue((root / "m1.joblib").exists())
            self.assertTrue((root / "r1.json").exists())
            self.assertEqual(first["cross_validation"]["metrics"]["n"], 598)


if __name__ == "__main__":
    unittest.main()
