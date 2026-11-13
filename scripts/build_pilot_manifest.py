"""Build a deterministic, stratified manifest for the frozen test pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATASET = PROJECT_ROOT / "data" / "test.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "pilot_manifest.json"
SEED = 506
REQUIRED_COUNTS = {
    "math": 20,
    "mcq": 20,
    "extract": 20,
    "reformat": 20,
    "classify": 16,
}


def _read_test_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}, line {line_number}") from error
        if not isinstance(row, dict) or row.get("split") != "test":
            raise ValueError(f"Invalid test dataset row at line {line_number}")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"Missing test question ID at line {line_number}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate test question ID: {row['id']}")
        if not isinstance(row.get("task_type"), str):
            raise ValueError(f"Missing task type for test question {row['id']}")
        seen.add(row["id"])
        rows.append(row)
    return rows


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
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


def build_pilot_manifest(
    dataset_path: Path = TEST_DATASET,
    output_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Select the fixed per-task pilot without modifying its source dataset."""
    rows = _read_test_rows(dataset_path)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)

    if len(by_task.get("classify", [])) != REQUIRED_COUNTS["classify"]:
        raise ValueError(
            f"Expected exactly {REQUIRED_COUNTS['classify']} classify rows; "
            f"found {len(by_task.get('classify', []))}"
        )
    for task, count in REQUIRED_COUNTS.items():
        if len(by_task.get(task, [])) < count:
            raise ValueError(f"Need {count} {task} rows, found {len(by_task.get(task, []))}")

    rng = random.Random(SEED)
    selected_ids = set()
    for task, count in REQUIRED_COUNTS.items():
        candidates = by_task[task]
        chosen = candidates if task == "classify" else rng.sample(candidates, count)
        selected_ids.update(row["id"] for row in chosen)

    selected = [
        {"id": row["id"], "task_type": row["task_type"]}
        for row in rows if row["id"] in selected_ids
    ]
    selected_counts = Counter(row["task_type"] for row in selected)
    if len(selected) != 96 or len({row["id"] for row in selected}) != 96:
        raise ValueError("Pilot selection did not produce 96 unique test IDs")
    if dict(selected_counts) != REQUIRED_COUNTS:
        raise ValueError(f"Unexpected pilot task counts: {dict(selected_counts)}")
    available_ids = {row["id"] for row in rows}
    if not {row["id"] for row in selected} <= available_ids:
        raise ValueError("Pilot contains an ID absent from the frozen test dataset")

    manifest = {
        "schema_version": 1,
        "split": "test",
        "seed": SEED,
        "source_test_dataset": "data/test.jsonl",
        "source_test_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "sample_size": len(selected),
        "task_counts": {task: selected_counts[task] for task in REQUIRED_COUNTS},
        "questions": selected,
    }
    _write_json_atomic(output_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    manifest = build_pilot_manifest(output_path=args.output)
    print(json.dumps({key: manifest[key] for key in (
        "seed", "sample_size", "task_counts", "source_test_sha256"
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
