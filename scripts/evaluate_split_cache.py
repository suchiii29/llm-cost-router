"""Evaluate cached tier answers for a pilot manifest without provider calls."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checker import check_answer  # noqa: E402
from scripts.build_pilot_manifest import REQUIRED_COUNTS, SEED  # noqa: E402
from scripts.cache_store import has_success, load_cache  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "data" / "pilot_manifest.json"
DATASET_PATH = PROJECT_ROOT / "data" / "test.jsonl"
CACHE_PATH = PROJECT_ROOT / "cache" / "answers_test.json"
RESULTS_PATH = PROJECT_ROOT / "results" / "pilot_cache_evaluation.json"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "PILOT_CACHE_EVALUATION.md"
TIERS = ("local", "cheap", "strong")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}, line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {path}, line {line_number}")
        rows.append(row)
    return rows


def _valid_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _latency_seconds(record: dict[str, Any]) -> float | None:
    for key in ("seconds", "latency_seconds"):
        value = record.get(key)
        if _valid_number(value):
            return float(value)
    milliseconds = record.get("latency_ms")
    if _valid_number(milliseconds):
        return float(milliseconds) / 1000
    return None


def _metadata_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        latency for record in records
        if (latency := _latency_seconds(record)) is not None
    ]
    successful = len(records)
    return {
        "successful_records": successful,
        "available_records": {
            "input_tokens": sum(
                isinstance(record.get("input_tokens"), int)
                and not isinstance(record.get("input_tokens"), bool)
                and record["input_tokens"] >= 0
                for record in records
            ),
            "output_tokens": sum(
                isinstance(record.get("output_tokens"), int)
                and not isinstance(record.get("output_tokens"), bool)
                and record["output_tokens"] >= 0
                for record in records
            ),
            "latency": len(latencies),
            "provider": sum(isinstance(record.get("provider"), str) and bool(record["provider"].strip()) for record in records),
            "model_id": sum(
                isinstance(record.get("model_id") or record.get("model"), str)
                and bool((record.get("model_id") or record.get("model")).strip())
                for record in records
            ),
            "timestamp": sum(isinstance(record.get("timestamp"), str) and bool(record["timestamp"].strip()) for record in records),
        },
        "missing_records": {
            "input_tokens": successful - sum(
                isinstance(record.get("input_tokens"), int)
                and not isinstance(record.get("input_tokens"), bool)
                and record["input_tokens"] >= 0
                for record in records
            ),
            "output_tokens": successful - sum(
                isinstance(record.get("output_tokens"), int)
                and not isinstance(record.get("output_tokens"), bool)
                and record["output_tokens"] >= 0
                for record in records
            ),
            "latency": successful - len(latencies),
            "provider": successful - sum(isinstance(record.get("provider"), str) and bool(record["provider"].strip()) for record in records),
            "model_id": successful - sum(
                isinstance(record.get("model_id") or record.get("model"), str)
                and bool((record.get("model_id") or record.get("model")).strip())
                for record in records
            ),
            "timestamp": successful - sum(isinstance(record.get("timestamp"), str) and bool(record["timestamp"].strip()) for record in records),
        },
        "latency_seconds": {
            "n": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


def _load_pilot(manifest_path: Path, dataset_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("split") != "test":
        raise ValueError("Pilot manifest must describe the test split")
    dataset_bytes = dataset_path.read_bytes()
    actual_hash = hashlib.sha256(dataset_bytes).hexdigest()
    if manifest.get("source_test_sha256") != actual_hash:
        raise ValueError("Pilot manifest hash does not match the test dataset")
    dataset_rows = _read_jsonl(dataset_path)
    dataset_by_id = {row.get("id"): row for row in dataset_rows}
    if len(dataset_by_id) != len(dataset_rows):
        raise ValueError("Test dataset contains duplicate IDs")
    selected = manifest.get("questions")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Pilot manifest has no question list")
    selected_ids = [row.get("id") for row in selected if isinstance(row, dict)]
    if (
        len(selected_ids) != len(selected)
        or any(not isinstance(item_id, str) or not item_id for item_id in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
    ):
        raise ValueError("Pilot manifest contains malformed or duplicate IDs")
    expected_counts = {task: REQUIRED_COUNTS[task] for task in REQUIRED_COUNTS}
    actual_counts = defaultdict(int)
    for row in selected:
        if isinstance(row.get("task_type"), str):
            actual_counts[row["task_type"]] += 1
    if manifest.get("seed") != SEED or manifest.get("sample_size") != 96:
        raise ValueError("Pilot manifest seed or sample size is invalid")
    if len(selected) != 96 or dict(actual_counts) != expected_counts:
        raise ValueError("Pilot manifest does not have the required task composition")
    rows = []
    for selected_row in selected:
        item = dataset_by_id.get(selected_row["id"])
        if item is None:
            raise ValueError(f"Pilot ID {selected_row['id']!r} is not in test dataset")
        if selected_row.get("task_type") != item.get("task_type"):
            raise ValueError(f"Task type mismatch for pilot ID {selected_row['id']!r}")
        rows.append(item)
    return manifest, rows


def _tier_status(
    items: list[dict[str, Any]], cache: dict[str, Any], tier: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    failure_keys = {
        (failure.get("id"), failure.get("tier"))
        for failure in cache.get("failures", [])
        if isinstance(failure, dict)
    }
    result_rows = []
    answers = []
    malformed = []
    status_counts = {"successful": 0, "failed": 0, "missing": 0, "incorrect": 0, "correct": 0}
    per_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata_records = []

    for item in items:
        item_id = item["id"]
        entry = cache["entries"].get(item_id)
        record = None
        malformed_reason = None
        if entry is not None:
            if not isinstance(entry, dict):
                malformed_reason = "cache entry is not an object"
            elif entry.get("id") != item_id:
                raise ValueError(f"Cache entry ID mismatch for pilot ID {item_id!r}")
            elif entry.get("question") != item["question"]:
                raise ValueError(f"Cache question mismatch for pilot ID {item_id!r}")
            else:
                tier_records = entry.get("answers")
                if not isinstance(tier_records, dict):
                    malformed_reason = "answers mapping is missing or invalid"
                else:
                    candidate = tier_records.get(tier)
                    if candidate is not None and not isinstance(candidate, dict):
                        malformed_reason = "tier record is not an object"
                    elif isinstance(candidate, dict):
                        if not has_success({"entries": {item_id: entry}}, item_id, tier):
                            malformed_reason = "answer is missing or is not text"
                        elif candidate.get("tier") not in (None, tier):
                            malformed_reason = "record tier does not match cache tier"
                        else:
                            record = candidate
        if malformed_reason:
            malformed.append({"id": item_id, "tier": tier, "reason": malformed_reason})

        failure = (item_id, tier) in failure_keys
        if record is None:
            status = "failed" if failure else "missing"
            status_counts[status] += 1
            row = {"id": item_id, "task_type": item["task_type"], "status": status,
                   "correct": None}
        else:
            correct = bool(check_answer(record["answer"], item["answer"], item["checker"]))
            status = "correct" if correct else "incorrect"
            status_counts["successful"] += 1
            status_counts[status] += 1
            metadata_records.append(record)
            answer_row = {"id": item_id, "task_type": item["task_type"], "correct": correct}
            answers.append(answer_row)
            row = {**answer_row, "status": status}
        result_rows.append(row)
        per_task[item["task_type"]].append(row)

    task_results = {}
    for task, task_rows in sorted(per_task.items()):
        task_successes = [row for row in task_rows if row["correct"] is not None]
        task_correct = sum(row["correct"] is True for row in task_successes)
        task_results[task] = {
            "questions": len(task_rows),
            "successful_answers": len(task_successes),
            "missing_answers": sum(row["status"] == "missing" for row in task_rows),
            "failed_answers": sum(row["status"] == "failed" for row in task_rows),
            "correct": task_correct,
            "incorrect": len(task_successes) - task_correct,
            "accuracy": task_correct / len(task_successes) if task_successes else None,
        }
    status_counts["missing_answers"] = status_counts.pop("missing")
    status_counts["failed_answers"] = status_counts.pop("failed")
    evaluated = status_counts["successful"]
    report = {
        "questions": len(items),
        **status_counts,
        "accuracy": status_counts["correct"] / evaluated if evaluated else None,
        "per_task": task_results,
        "metadata": _metadata_report(metadata_records),
        "malformed_records": malformed,
    }
    return report, answers, result_rows


def _paired_comparisons(tier_answers: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    answer_maps = {
        tier: {row["id"]: row for row in rows}
        for tier, rows in tier_answers.items()
    }
    for left, right in itertools.combinations(TIERS, 2):
        common_ids = sorted(set(answer_maps[left]) & set(answer_maps[right]))
        cells = {"both_correct": 0, "left_only_correct": 0,
                 "right_only_correct": 0, "both_incorrect": 0}
        for item_id in common_ids:
            left_ok = answer_maps[left][item_id]["correct"]
            right_ok = answer_maps[right][item_id]["correct"]
            if left_ok and right_ok:
                cells["both_correct"] += 1
            elif left_ok:
                cells["left_only_correct"] += 1
            elif right_ok:
                cells["right_only_correct"] += 1
            else:
                cells["both_incorrect"] += 1
        result[f"{left}_vs_{right}"] = {
            "left_tier": left,
            "right_tier": right,
            "paired_questions": len(common_ids),
            "left_accuracy_on_pairs": (
                (cells["both_correct"] + cells["left_only_correct"]) / len(common_ids)
                if common_ids else None
            ),
            "right_accuracy_on_pairs": (
                (cells["both_correct"] + cells["right_only_correct"]) / len(common_ids)
                if common_ids else None
            ),
            "correctness_contingency": cells,
        }
    return result


def evaluate_pilot(
    manifest_path: Path = MANIFEST_PATH,
    dataset_path: Path = DATASET_PATH,
    cache_path: Path = CACHE_PATH,
) -> dict[str, Any]:
    manifest, items = _load_pilot(manifest_path, dataset_path)
    cache = load_cache(cache_path, "test")
    tier_reports = {}
    tier_answers = {}
    for tier in TIERS:
        report, answers, _ = _tier_status(items, cache, tier)
        tier_reports[tier] = report
        tier_answers[tier] = answers
    return {
        "schema_version": 1,
        "split": "test",
        "pilot_seed": manifest.get("seed"),
        "pilot_sample_size": len(items),
        "source_test_sha256": manifest.get("source_test_sha256"),
        "tier_results": tier_reports,
        "paired_comparisons": _paired_comparisons(tier_answers),
        "cost": {
            "reported": False,
            "reason": "This offline evaluator does not infer pricing or calculate dollar cost.",
        },
    }


def _format_seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pilot Cache Evaluation",
        "",
        "Offline evaluation of cached responses for the deterministic test pilot.",
        "",
        f"- Questions: {report['pilot_sample_size']}",
        f"- Seed: {report['pilot_seed']}",
        f"- Test dataset SHA-256: `{report['source_test_sha256']}`",
        "- Provider calls: none",
        "- Dollar cost: not calculated; this evaluator does not infer pricing.",
        "",
        "## Per-tier results",
        "",
        "| Tier | Successful | Correct | Incorrect | Failed | Missing | Accuracy | Valid latency n | Mean s | Median s | Missing latency metadata |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in TIERS:
        data = report["tier_results"][tier]
        latency = data["metadata"]["latency_seconds"]
        accuracy = "—" if data["accuracy"] is None else f"{data['accuracy']:.3%}"
        lines.append(
            f"| {tier} | {data['successful']} | {data['correct']} | {data['incorrect']} | "
            f"{data['failed_answers']} | {data['missing_answers']} | {accuracy} | "
            f"{latency['n']} | {_format_seconds(latency['mean'])} | {_format_seconds(latency['median'])} | "
            f"{data['metadata']['missing_records']['latency']} |"
        )
    lines.extend(["", "## Per-task accuracy", ""])
    tasks = sorted({
        task for tier in TIERS for task in report["tier_results"][tier]["per_task"]
    })
    lines.append("| Task | Pilot n | " + " | ".join(TIERS) + " |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for task in tasks:
        values = []
        task_n = 0
        for tier in TIERS:
            task_result = report["tier_results"][tier]["per_task"].get(task)
            if not task_result:
                values.append("—")
                continue
            task_n = task_result["questions"]
            accuracy = "—" if task_result["accuracy"] is None else f"{task_result['accuracy']:.3%}"
            values.append(
                f"{accuracy} ({task_result['correct']}/{task_result['successful_answers']} success; "
                f"{task_result['missing_answers']} missing; {task_result['failed_answers']} failed)"
            )
        lines.append(f"| {task} | {task_n} | " + " | ".join(values) + " |")
    lines.extend(["", "## Metadata availability", "", "Counts are among successful cached answers; no values are inferred.", "", "| Tier | Input tokens | Output tokens | Latency | Provider | Model ID | Timestamp | Malformed records |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for tier in TIERS:
        data = report["tier_results"][tier]
        available = data["metadata"]["available_records"]
        lines.append(
            f"| {tier} | {available['input_tokens']} available / {data['successful'] - available['input_tokens']} missing | "
            f"{available['output_tokens']} available / {data['successful'] - available['output_tokens']} missing | "
            f"{available['latency']} available / {data['metadata']['missing_records']['latency']} missing | "
            f"{available['provider']} available / {data['metadata']['missing_records']['provider']} missing | "
            f"{available['model_id']} available / {data['metadata']['missing_records']['model_id']} missing | "
            f"{available['timestamp']} available / {data['metadata']['missing_records']['timestamp']} missing | "
            f"{len(data['malformed_records'])} |"
        )
    lines.extend(["", "## Paired tier comparisons", "", "| Pair | Paired n | Left accuracy | Right accuracy | Both correct | Left only | Right only | Both incorrect |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name, pair in report["paired_comparisons"].items():
        cells = pair["correctness_contingency"]
        left_accuracy = "—" if pair["left_accuracy_on_pairs"] is None else f"{pair['left_accuracy_on_pairs']:.3%}"
        right_accuracy = "—" if pair["right_accuracy_on_pairs"] is None else f"{pair['right_accuracy_on_pairs']:.3%}"
        lines.append(
            f"| {name} | {pair['paired_questions']} | {left_accuracy} | {right_accuracy} | "
            f"{cells['both_correct']} | {cells['left_only_correct']} | "
            f"{cells['right_only_correct']} | {cells['both_incorrect']} |"
        )
    lines.extend(["", "Latency uses only valid numeric cache metadata. Missing answers are not counted as incorrect; accuracy uses successful cached answers only.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    report = evaluate_pilot(args.manifest, args.dataset, args.cache)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(_summary_markdown(report), encoding="utf-8")
    print(json.dumps({
        "results": str(args.results), "summary": str(args.summary),
        "pilot_sample_size": report["pilot_sample_size"],
        "tier_results": {
            tier: {key: value[key] for key in ("successful", "correct", "incorrect", "failed_answers", "missing_answers", "accuracy")}
            for tier, value in report["tier_results"].items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
