"""Import the historical local/strong development answers into the v1 cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_store import (  # noqa: E402
    CacheConflictError,
    ensure_entry,
    load_cache,
    save_cache,
    store_success,
)


DEFAULT_SOURCE = PROJECT_ROOT / "answers_cache.json"
DEFAULT_BENCHMARK = PROJECT_ROOT / "benchmark_questions.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "cache" / "answers_dev.json"
LEGACY_TIER_MAP = {"local": "local", "strong": "strong"}


def load_dev_items(
    benchmark_path: Path = DEFAULT_BENCHMARK,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[dict[str, Any]]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    dataset_path = data_dir / "dev.jsonl"
    if not dataset_path.exists():
        return [
            {
                "id": f"dev-existing-{index:03d}",
                "source_id": str(index),
                "question": row["question"],
                "answer": row["correct_answer"],
            }
            for index, row in enumerate(benchmark)
        ]

    items = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(items) != len(benchmark):
        raise ValueError(
            "data/dev.jsonl item count does not match benchmark_questions.json"
        )
    for index, (item, source) in enumerate(zip(items, benchmark)):
        if (
            item.get("question") != source["question"]
            or item.get("answer") != source["correct_answer"]
        ):
            raise ValueError(
                f"Development item {index} differs from benchmark_questions.json"
            )
        item.setdefault("source_id", str(index))
    return items


def _parse_legacy_key(key: str) -> tuple[int, str]:
    try:
        index_text, legacy_tier = key.split("|", 1)
        index = int(index_text)
    except (AttributeError, ValueError) as error:
        raise ValueError("Malformed legacy cache key") from error
    tier = LEGACY_TIER_MAP.get(legacy_tier)
    if tier is None:
        raise ValueError(f"Unknown legacy tier {legacy_tier!r}")
    return index, tier


def import_old_cache(
    legacy_cache: dict[str, Any],
    dev_items: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Merge matching old answers into the new dev cache without overwriting."""
    cache = load_cache(output_path, "dev")
    report: dict[str, Any] = {
        "source_entries": len(legacy_cache),
        "imported": 0,
        "already_present": 0,
        "missing": [],
        "unmatched": [],
    }

    seen: set[tuple[int, str]] = set()
    conversions = []
    for key, source_record in legacy_cache.items():
        try:
            index, tier = _parse_legacy_key(key)
        except ValueError as error:
            report["unmatched"].append({"key": str(key), "reason": str(error)})
            continue
        if index < 0 or index >= len(dev_items):
            report["unmatched"].append(
                {"key": key, "reason": "question ID is outside the dev set"}
            )
            continue
        if not isinstance(source_record, dict) or not isinstance(
            source_record.get("answer"), str
        ):
            report["unmatched"].append(
                {"key": key, "reason": "cache entry has no string answer"}
            )
            continue

        item = dev_items[index]
        embedded_question = source_record.get("question")
        if embedded_question is not None and embedded_question != item["question"]:
            report["unmatched"].append(
                {"key": key, "reason": "cached question does not match dev question"}
            )
            continue
        if (index, tier) in seen:
            report["unmatched"].append(
                {"key": key, "reason": "duplicate question/tier mapping"}
            )
            continue
        seen.add((index, tier))
        record = dict(source_record)
        record["tier"] = tier
        record["imported"] = True
        conversions.append((index, item, tier, record))

    for index, item in enumerate(dev_items):
        for legacy_tier in LEGACY_TIER_MAP:
            tier = LEGACY_TIER_MAP[legacy_tier]
            if (index, tier) not in seen:
                report["missing"].append(
                    {
                        "source_id": str(index),
                        "question_id": item["id"],
                        "tier": tier,
                    }
                )

    for index, item, tier, record in conversions:
        entry = ensure_entry(
            cache,
            item_id=item["id"],
            question=item["question"],
            source_id=str(index),
        )
        existing = entry["answers"].get(tier)
        try:
            stored = store_success(
                cache,
                item_id=item["id"],
                source_id=str(index),
                question=item["question"],
                tier=tier,
                record=record,
            )
        except CacheConflictError:
            raise
        if stored:
            report["imported"] += 1
        else:
            report["already_present"] += 1
            if existing is not None and existing.get("answer") != record["answer"]:
                raise CacheConflictError(
                    f"Conflicting answer for {item['id']} tier {tier}"
                )

    save_cache(output_path, cache)
    report["successfully_available"] = (
        report["imported"] + report["already_present"]
    )
    report["missing_count"] = len(report["missing"])
    report["unmatched_count"] = len(report["unmatched"])
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    source = json.loads(args.source.read_text(encoding="utf-8"))
    dev_items = load_dev_items(args.benchmark, args.data_dir)
    report = import_old_cache(source, dev_items, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
