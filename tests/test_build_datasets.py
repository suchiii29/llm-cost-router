import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_datasets import (
    DEFAULT_SEED,
    build_dataset_items,
    generate_simple_tasks,
    main,
    validate_item,
    write_datasets,
)


def gsm_examples(prefix: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "question": f"{prefix} arithmetic question {index}?",
            "answer": f"work steps\n#### {index + 10}",
        }
        for index in range(count)
    ]


def arc_examples(prefix: str, count: int) -> list[dict]:
    return [
        {
            "question": f"{prefix} science question {index}?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": [
                    f"choice {index} alpha",
                    f"choice {index} beta",
                    f"choice {index} gamma",
                    f"choice {index} delta",
                ],
            },
            "answerKey": "C",
        }
        for index in range(count)
    ]


class DatasetBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev_source = json.loads(
            (Path(__file__).resolve().parents[1] / "benchmark_questions.json")
            .read_text(encoding="utf-8")
        )

    def build_fixture(self):
        gsm_test = gsm_examples("gsm8k test", 101)
        gsm_test[0]["question"] = self.dev_source[0]["question"]
        gsm_test[0]["answer"] = f"work\n#### {self.dev_source[0]['correct_answer']}"
        return build_dataset_items(
            dev_questions=self.dev_source,
            gsm8k_train=gsm_examples("gsm8k train", 205),
            gsm8k_test=gsm_test,
            arc_train=arc_examples("arc train", 205),
            arc_test=arc_examples("arc test", 105),
            seed=DEFAULT_SEED,
        )

    def test_schema_counts_sources_and_dev_content_are_preserved(self):
        datasets = self.build_fixture()
        self.assertEqual(len(datasets["train_pool"]), 598)
        self.assertEqual(len(datasets["dev"]), 60)
        self.assertEqual(len(datasets["test"]), 300)
        for split, items in datasets.items():
            for item in items:
                validate_item(item, split)
        for source, converted in zip(self.dev_source, datasets["dev"]):
            self.assertEqual(converted["question"], source["question"])
            self.assertEqual(converted["answer"], source["correct_answer"])
        train_counts = {}
        for item in datasets["train_pool"]:
            train_counts[item["id"].split("-")[0]] = (
                train_counts.get(item["id"].split("-")[0], 0) + 1
            )
        self.assertEqual(train_counts, {"gsm8k": 200, "arc": 200, "simple": 198})
        normalized_train = {
            " ".join(item["question"].casefold().split())
            for item in datasets["train_pool"]
        }
        normalized_test = {
            " ".join(item["question"].casefold().split())
            for item in datasets["test"]
        }
        self.assertFalse(normalized_train & normalized_test)
        self.assertNotIn("simple-train-0064", {item["id"] for item in datasets["train_pool"]})
        self.assertNotIn("simple-train-0010", {item["id"] for item in datasets["train_pool"]})
        mcq = [item for item in datasets["test"] if item["task_type"] == "mcq"]
        self.assertEqual(len(mcq), 100)
        self.assertTrue(all(item["checker"] == "letter" for item in mcq))
        self.assertTrue(all(len(item["answer"]) == 1 for item in mcq))
        dev_questions = {item["question"].casefold() for item in datasets["dev"]}
        selected_gsm_test = [
            item for item in datasets["test"] if item["id"].startswith("gsm8k-test-")
        ]
        self.assertEqual(len(selected_gsm_test), 100)
        self.assertFalse(
            any(item["question"].casefold() in dev_questions for item in selected_gsm_test)
        )

    def test_fixed_seed_reproduces_identical_items(self):
        first = self.build_fixture()
        second = self.build_fixture()
        self.assertEqual(first, second)
        simple_a = generate_simple_tasks(60, split="test", seed=99)
        simple_b = generate_simple_tasks(60, split="test", seed=99)
        self.assertEqual(simple_a, simple_b)
        self.assertEqual(
            {item["task_type"] for item in simple_a},
            {"extract", "reformat", "classify"},
        )

    def test_manifest_counts_and_hashes_match_generated_files(self):
        datasets = self.build_fixture()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            manifest = write_datasets(datasets, data_dir=data_dir)
            manifest_from_disk = json.loads(
                (data_dir / "MANIFEST.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest, manifest_from_disk)
            for filename, entry in manifest["files"].items():
                payload = (data_dir / filename).read_bytes()
                import hashlib

                self.assertEqual(
                    entry["sha256"], hashlib.sha256(payload).hexdigest()
                )
                self.assertEqual(entry["item_count"], len(payload.splitlines()))

    def test_existing_test_set_is_never_replaced_without_force(self):
        datasets = self.build_fixture()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            write_datasets(datasets, data_dir=data_dir)
            test_path = data_dir / "test.jsonl"
            frozen = test_path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_datasets(datasets, data_dir=data_dir)
            self.assertEqual(test_path.read_bytes(), frozen)

            changed = {key: list(value) for key, value in datasets.items()}
            changed["test"][0] = dict(changed["test"][0], answer="changed")
            write_datasets(changed, data_dir=data_dir, force_test=True)
            self.assertNotEqual(test_path.read_bytes(), frozen)

    @patch("scripts.build_datasets.load_remote_datasets")
    def test_cli_refuses_frozen_test_before_attempting_download(self, download):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "test.jsonl").write_text("frozen\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                main(["--data-dir", directory])
        download.assert_not_called()

    @patch("scripts.build_datasets.load_remote_datasets")
    def test_explicit_cli_build_writes_expected_files_with_mocked_sources(self, download):
        download.return_value = (
            gsm_examples("gsm8k train", 205),
            gsm_examples("gsm8k test", 105),
            arc_examples("arc train", 205),
            arc_examples("arc test", 105),
        )
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(main(["--data-dir", directory]), 0)
            data_dir = Path(directory)
            self.assertTrue((data_dir / "train_pool.jsonl").is_file())
            self.assertTrue((data_dir / "dev.jsonl").is_file())
            self.assertTrue((data_dir / "test.jsonl").is_file())
            manifest = json.loads(
                (data_dir / "MANIFEST.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [entry["item_count"] for entry in manifest["files"].values()],
                [598, 60, 300],
            )
            self.assertEqual(
                manifest["metadata"]["composition"]["train_pool"]["synthetic_simple"],
                198,
            )
        download.assert_called_once()

    def test_simple_templates_have_varied_questions(self):
        items = generate_simple_tasks(120, split="train", seed=7)
        self.assertEqual(len({item["question"] for item in items}), 120)
        self.assertGreaterEqual(
            len({item["question"].split(":")[0] for item in items}), 6
        )


if __name__ == "__main__":
    unittest.main()
