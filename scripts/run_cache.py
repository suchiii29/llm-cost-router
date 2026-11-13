"""Resumable explicit cache generation for a selected dataset split and tier."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, get_settings  # noqa: E402
from app.providers import get_provider  # noqa: E402
from app.providers.base import ProviderHTTPError  # noqa: E402
from scripts.cache_store import (  # noqa: E402
    has_success,
    load_cache,
    save_cache,
    store_failure,
    store_success,
)


TIERS = {"local": 0, "cheap": 1, "strong": 2}
SPLITS = {"dev", "train_pool", "test"}


def dataset_path(split: str, data_dir: Path = PROJECT_ROOT / "data") -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}")
    return data_dir / f"{split}.jsonl"


def cache_path(split: str, cache_dir: Path = PROJECT_ROOT / "cache") -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}")
    return cache_dir / f"answers_{split}.json"


def load_split(path: Path, split: str) -> list[dict[str, Any]]:
    items = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}, line {line_number}") from error
        if not isinstance(item, dict) or item.get("split") != split:
            raise ValueError(f"Invalid {split} dataset item at line {line_number}")
        if not all(key in item for key in ("id", "question", "answer", "checker")):
            raise ValueError(f"Incomplete dataset item at line {line_number}")
        items.append(item)
    return items



def select_manifest_items(
    *,
    items: list[dict[str, Any]],
    dataset_file: Path,
    manifest_file: Path,
    split: str,
) -> list[dict[str, Any]]:
    """Validate a dataset-bound manifest and restrict collection to its IDs."""
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid manifest JSON in {manifest_file}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest {manifest_file} must contain a JSON object")
    if manifest.get("split") != split:
        raise ValueError(
            f"Manifest split {manifest.get('split')!r} does not match requested split {split!r}"
        )

    source_hash = (
        manifest.get("source_dataset_sha256")
        or manifest.get(f"source_{split}_sha256")
        or manifest.get("source_sha256")
    )
    actual_hash = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    if not isinstance(source_hash, str) or source_hash != actual_hash:
        raise ValueError(
            f"Manifest source dataset hash does not match {dataset_file}"
        )

    if "questions" in manifest:
        listed = manifest["questions"]
        if not isinstance(listed, list):
            raise ValueError("Manifest 'questions' must be a list")
        ids = []
        listed_task_types = {}
        for row in listed:
            if not isinstance(row, dict):
                raise ValueError("Manifest question entries must be objects")
            item_id = row.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("Manifest question entries require a non-empty ID")
            ids.append(item_id)
            listed_task_types[item_id] = row.get("task_type")
    else:
        ids = manifest.get("ids")
        listed_task_types = {}
        if not isinstance(ids, list) or any(
            not isinstance(item_id, str) or not item_id for item_id in ids
        ):
            raise ValueError("Manifest must contain a 'questions' list or an 'ids' list")

    if len(set(ids)) != len(ids):
        raise ValueError("Manifest contains duplicate question IDs")
    dataset_by_id = {item["id"]: item for item in items}
    if len(dataset_by_id) != len(items):
        raise ValueError("Dataset contains duplicate question IDs")
    missing_ids = [item_id for item_id in ids if item_id not in dataset_by_id]
    if missing_ids:
        raise ValueError(
            "Manifest IDs are not present in the dataset: " + ", ".join(missing_ids)
        )
    for item_id, task_type in listed_task_types.items():
        if task_type is not None and task_type != dataset_by_id[item_id].get("task_type"):
            raise ValueError(f"Manifest task type does not match dataset for ID {item_id!r}")

    selected_ids = set(ids)
    selected = [item for item in items if item["id"] in selected_ids]
    if len(selected) != len(ids):
        raise ValueError("Manifest selection did not resolve to every listed dataset ID")
    return selected


def _safe_failure(error: Exception) -> dict[str, Any]:
    status = getattr(error, "status_code", None)
    if isinstance(error, ProviderHTTPError) and status is not None:
        message = f"HTTP {status}"
    else:
        message = f"Request failed ({type(error).__name__})"
    return {
        "error_type": type(error).__name__,
        "error": message,
        "status_code": status if isinstance(status, int) else None,
        "retry_after_seconds": getattr(error, "retry_after_seconds", None),
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }


def _answer_record(response, tier_name: str) -> dict[str, Any]:
    latency_seconds = response.latency_seconds
    return {
        "answer": response.answer,
        "tier": tier_name,
        "model_id": response.model_id,
        # Keep the original key for readers of cache records written by older versions.
        "model": response.model_id,
        "provider": response.provider,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "token_source": response.token_source,
        "latency_ms": response.latency_ms,
        "latency_seconds": latency_seconds,
        "seconds": latency_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "imported": False,
    }


def prompt_for_item(item: dict[str, Any]) -> str:
    question = item["question"]
    checker = item.get("checker")
    if checker == "numeric":
        return (
            question
            + "\n\nSolve the problem, then return only the final number. "
            + "Do not include a label, explanation, or other text."
        )
    if checker == "letter":
        return question + "\n\nReturn only the single answer letter, with no explanation."
    if checker in {"exact", "contains"}:
        return question + "\n\nReturn only the requested output, with no label or extra text."
    return question + "\n\nReturn only the requested output, with no extra text."


def cache_status(
    items: list[dict[str, Any]], cache: dict[str, Any], tier_name: str
) -> dict[str, int]:
    failures = {
        (failure.get("id"), failure.get("tier"))
        for failure in cache.get("failures", [])
    }
    result = {"successful": 0, "failed": 0, "missing": 0}
    for item in items:
        if has_success(cache, item["id"], tier_name):
            result["successful"] += 1
        elif (item["id"], tier_name) in failures:
            result["failed"] += 1
        else:
            result["missing"] += 1
    return result


def run_cache(
    *,
    split: str,
    tier_name: str,
    settings: Optional[Settings] = None,
    data_dir: Path = PROJECT_ROOT / "data",
    cache_dir: Path = PROJECT_ROOT / "cache",
    manifest_path: Optional[Path] = None,
    limit: Optional[int] = None,
    request_interval: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    provider_factory=get_provider,
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}")
    if tier_name not in TIERS:
        raise ValueError(f"Unsupported tier {tier_name!r}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    if request_interval < 0:
        raise ValueError("request_interval must be non-negative")

    settings = settings or get_settings()
    tier = settings.tier(TIERS[tier_name])
    source_path = dataset_path(split, data_dir)
    items = load_split(source_path, split)
    if manifest_path is not None:
        items = select_manifest_items(
            items=items, dataset_file=source_path, manifest_file=manifest_path, split=split
        )
    destination = cache_path(split, cache_dir)
    cache = load_cache(destination, split)
    before = cache_status(items, cache, tier_name)
    pending = [
        item for item in items if not has_success(cache, item["id"], tier_name)
    ]
    selected = pending if limit is None else pending[:limit]
    report = {
        "split": split,
        "tier": tier_name,
        "skipped_successes": before["successful"],
        "failed_before_run": before["failed"],
        "missing_before_run": before["missing"],
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
    }
    if manifest_path is not None:
        report.update({
            "selected_questions": len(items),
            "already_cached_successes": before["successful"],
            "remaining_requests": len(pending),
            "scheduled_requests": len(selected),
        })
        print(
            f"Manifest selection: {len(items)} questions; "
            f"already cached successes: {before['successful']}; "
            f"remaining requests: {len(pending)}; "
            f"scheduled this run: {len(selected)}"
        )

    last_request_at = None
    for item in selected:
        if last_request_at is not None and request_interval:
            remaining = request_interval - (clock() - last_request_at)
            if remaining > 0:
                sleep(remaining)
        report["attempted"] += 1
        request_started = clock()
        retry_delay = None
        try:
            provider = provider_factory(settings, tier)
            prompt = prompt_for_item(item)
            response = provider.generate(prompt, tier.model_id)
            record = _answer_record(response, tier_name)
            record["prompt"] = prompt
            stored = store_success(
                cache,
                item_id=item["id"],
                source_id=item.get("source_id"),
                question=item["question"],
                tier=tier_name,
                record=record,
            )
            if stored:
                report["succeeded"] += 1
            else:
                report["skipped_successes"] += 1
        except Exception as error:
            failure = _safe_failure(error)
            failure.update(
                {
                    "model": tier.model_id,
                    "provider": tier.provider,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "token_source": None,
                    "latency_ms": max(0, round((clock() - request_started) * 1000)),
                }
            )
            failure["latency_seconds"] = failure["latency_ms"] / 1000
            store_failure(
                cache,
                item_id=item["id"],
                source_id=item.get("source_id"),
                question=item["question"],
                tier=tier_name,
                failure=failure,
            )
            report["failed"] += 1
            retry_after = failure["retry_after_seconds"]
            if isinstance(retry_after, (int, float)) and retry_after > 0:
                retry_delay = float(retry_after)
        finally:
            last_request_at = clock()
            save_cache(destination, cache)
        if retry_delay is not None:
            sleep(retry_delay)

    report["status_after_run"] = cache_status(items, cache, tier_name)
    report["still_missing"] = report["status_after_run"]["missing"]
    report["still_failed"] = report["status_after_run"]["failed"]
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=sorted(SPLITS))
    parser.add_argument("--tier", required=True, choices=sorted(TIERS))
    parser.add_argument(
        "--limit",
        type=int,
        help="maximum missing answers to attempt in this run",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="restrict collection to IDs in a manifest bound to the selected dataset hash",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=1.0,
        help="minimum seconds between attempts (default: 1)",
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    args = parser.parse_args(argv)

    try:
        report = run_cache(
            split=args.split,
            tier_name=args.tier,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            manifest_path=args.manifest,
            limit=args.limit,
            request_interval=args.request_interval,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
