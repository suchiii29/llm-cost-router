"""Build offline local-correctness labels from the train pool and answer cache."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checker import check_answer  # noqa: E402

TRAIN_POOL_PATH = PROJECT_ROOT / "data" / "train_pool.jsonl"
TRAIN_CACHE_PATH = PROJECT_ROOT / "cache" / "answers_train_pool.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "train_labels.jsonl"
EXPECTED_COUNT = 598


class LabelBuildError(ValueError):
    """Raised when the training dataset or cache cannot produce a complete label set."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise LabelBuildError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _read_training_items(path: Path) -> list[dict[str, Any]]:
    items = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, LabelBuildError) as error:
            raise LabelBuildError(
                f"Invalid training JSON at {path}, line {line_number}: {error}"
            ) from error
        if not isinstance(item, dict):
            raise LabelBuildError(f"Training row {line_number} must be a JSON object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise LabelBuildError(f"Training row {line_number} has no valid ID")
        if item_id in seen_ids:
            raise LabelBuildError(f"Duplicate training question ID: {item_id}")
        seen_ids.add(item_id)
        if item.get("split") != "train_pool":
            raise LabelBuildError(
                f"Training row {line_number} has split {item.get('split')!r}, "
                "expected 'train_pool'"
            )
        for field in ("question", "answer", "task_type", "checker"):
            if field not in item:
                raise LabelBuildError(
                    f"Training row {line_number} is missing field {field!r}"
                )
        if not isinstance(item["question"], str):
            raise LabelBuildError(f"Question {item_id!r} has a non-text question")
        if not isinstance(item["task_type"], str) or not item["task_type"]:
            raise LabelBuildError(f"Question {item_id!r} has no valid task type")
        if not isinstance(item["checker"], str) or not item["checker"]:
            raise LabelBuildError(f"Question {item_id!r} has no valid checker type")
        items.append(item)
    return items


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        cache = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, LabelBuildError) as error:
        raise LabelBuildError(f"Invalid training cache {path}: {error}") from error
    if not isinstance(cache, dict) or cache.get("split") != "train_pool":
        raise LabelBuildError(f"Cache {path} is not a train_pool cache")
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        raise LabelBuildError(f"Cache {path} has no entries mapping")
    return cache


def _available_seconds(record: dict[str, Any]) -> Any:
    for field in ("seconds", "latency_seconds"):
        value = record.get(field)
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            return value
    latency_ms = record.get("latency_ms")
    if isinstance(latency_ms, (int, float)) and math.isfinite(latency_ms) and latency_ms >= 0:
        return latency_ms / 1000
    return None


def _label_for_item(
    item: dict[str, Any], record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    raw_answer = record["answer"]
    checker_error = None
    try:
        is_correct = check_answer(raw_answer, item["answer"], item["checker"])
    except Exception as error:  # Keep the item and report checker errors instead of dropping it.
        is_correct = False
        checker_error = {
            "id": item["id"],
            "error_type": type(error).__name__,
            "error": str(error),
        }

    model_id = record.get("model_id") or record.get("model")
    seconds = _available_seconds(record)
    required_metadata = {
        "model_id": model_id,
        "provider": record.get("provider"),
        "seconds": seconds,
        "timestamp": record.get("timestamp"),
    }
    missing_metadata = [
        field for field, value in required_metadata.items()
        if value is None or value == ""
    ]

    label = {
        "id": item["id"],
        "question": item["question"],
        "ground_truth_answer": item["answer"],
        "local_answer": raw_answer,
        "task_type": item["task_type"],
        "checker": item["checker"],
        "local_ok": int(bool(is_correct)),
        "semantic_json_equivalent": _semantic_json_equivalent(
            item["answer"], raw_answer, item["task_type"], item["checker"]
        ),
        "model_id": model_id,
        "provider": record.get("provider"),
        "seconds": seconds,
        "latency_ms": record.get("latency_ms"),
        "latency_seconds": record.get("latency_seconds"),
        "timestamp": record.get("timestamp"),
    }
    return label, checker_error, missing_metadata


def _semantic_json_equivalent(
    expected_answer: Any, raw_answer: str, task_type: str, checker: str
) -> bool | None:
    """Diagnose JSON-array equivalence without affecting the primary checker result."""
    if task_type != "reformat" or checker != "exact":
        return None
    try:
        expected_value = json.loads(str(expected_answer))
    except (TypeError, ValueError):
        return None
    if not isinstance(expected_value, list):
        return None

    fenced = re.fullmatch(
        r"\s*```(?:json)?\s*(.*?)\s*```\s*",
        raw_answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidate = fenced.group(1) if fenced else raw_answer
    try:
        actual_value = json.loads(candidate)
    except (TypeError, ValueError):
        return False
    return isinstance(actual_value, list) and actual_value == expected_value


def _distribution(labels: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(label["local_ok"] for label in labels)
    total = len(labels)
    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else 0.0,
    }


def _report(
    labels: list[dict[str, Any]], checker_errors, missing_metadata,
    unlabelled_cache_entries,
):
    grouped = defaultdict(list)
    for label in labels:
        grouped[label["task_type"]].append(label)
    failures_by_checker = defaultdict(int)
    for label in labels:
        if label["local_ok"] == 0:
            failures_by_checker[label["checker"]] += 1
    json_diagnostic_rows = [
        label for label in labels
        if label["semantic_json_equivalent"] is not None
    ]
    json_equivalent = sum(
        label["semantic_json_equivalent"] for label in json_diagnostic_rows
    )
    return {
        "total_labels": len(labels),
        "overall": _distribution(labels),
        "by_task_type": {
            task_type: _distribution(grouped[task_type])
            for task_type in sorted(grouped)
        },
        "checker_failure_count": sum(failures_by_checker.values()),
        "checker_failures_by_type": {
            checker_type: failures_by_checker[checker_type]
            for checker_type in sorted(failures_by_checker)
        },
        "checker_error_count": len(checker_errors),
        "checker_errors": checker_errors,
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata": missing_metadata,
        "semantic_json_diagnostic": {
            "eligible": len(json_diagnostic_rows),
            "equivalent": json_equivalent,
            "not_equivalent": len(json_diagnostic_rows) - json_equivalent,
            "accuracy": (
                json_equivalent / len(json_diagnostic_rows)
                if json_diagnostic_rows else None
            ),
        },
        "unlabelled_cache_entry_count": len(unlabelled_cache_entries),
        "unlabelled_cache_entries": unlabelled_cache_entries,
    }


def _write_jsonl_atomic(path: Path, labels: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n"
        for label in labels
    ).encode("utf-8")
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_training_labels(
    *,
    dataset_path: Path = TRAIN_POOL_PATH,
    cache_path: Path = TRAIN_CACHE_PATH,
    output_path: Path = OUTPUT_PATH,
    expected_count: int = EXPECTED_COUNT,
) -> dict[str, Any]:
    """Validate source inputs, score every local answer and atomically write labels."""
    items = _read_training_items(dataset_path)
    if len(items) != expected_count:
        raise LabelBuildError(
            f"Expected {expected_count} unique training questions, found {len(items)}"
        )

    cache = _read_cache(cache_path)
    item_ids = {item["id"] for item in items}
    cache_ids = set(cache["entries"])
    extra_cache_ids = sorted(cache_ids - item_ids)
    unlabelled_cache_entries = [
        {"id": item_id, "reason": "no matching row in current train_pool"}
        for item_id in extra_cache_ids
    ]

    missing_answers = []
    for item in items:
        entry = cache["entries"].get(item["id"])
        if not isinstance(entry, dict):
            missing_answers.append(item["id"])
            continue
        if entry.get("id", item["id"]) != item["id"]:
            raise LabelBuildError(
                f"Cache entry key {item['id']!r} does not match its stored ID"
            )
        record = entry.get("answers", {}).get("local", {})
        if not isinstance(record, dict) or not isinstance(record.get("answer"), str):
            missing_answers.append(item["id"])
    if missing_answers:
        raise LabelBuildError(
            f"Missing local cache answers for {len(missing_answers)} IDs: "
            + ", ".join(missing_answers)
        )

    labels = []
    checker_errors = []
    missing_metadata = []
    for item in items:
        record = cache["entries"][item["id"]]["answers"]["local"]
        label, checker_error, absent_fields = _label_for_item(item, record)
        labels.append(label)
        if checker_error is not None:
            checker_errors.append(checker_error)
        if absent_fields:
            missing_metadata.append({"id": item["id"], "fields": absent_fields})

    if len(labels) != expected_count or len({label["id"] for label in labels}) != expected_count:
        raise LabelBuildError("Generated labels failed unique ID/count validation")
    report = _report(
        labels, checker_errors, missing_metadata, unlabelled_cache_entries
    )
    _write_jsonl_atomic(output_path, labels)
    return report


def main() -> int:
    report = build_training_labels()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
