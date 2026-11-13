"""Shared cache format and atomic persistence helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class CacheConflictError(ValueError):
    """Raised when a cache already has a different successful answer."""


def new_cache(split: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split": split,
        "entries": {},
        "failures": [],
    }


def load_cache(path: Path, split: str) -> dict[str, Any]:
    if not path.exists():
        return new_cache(split)
    cache = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cache, dict) or cache.get("split") != split:
        raise ValueError(f"Cache {path} is not a valid {split!r} cache")
    if not isinstance(cache.get("entries"), dict):
        raise ValueError(f"Cache {path} has no entries mapping")
    if not isinstance(cache.get("failures", []), list):
        raise ValueError(f"Cache {path} has an invalid failures list")
    cache.setdefault("schema_version", 1)
    cache.setdefault("failures", [])
    return cache


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(cache, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
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


def ensure_entry(
    cache: dict[str, Any],
    *,
    item_id: str,
    question: str,
    source_id: str | None = None,
) -> dict[str, Any]:
    entries = cache["entries"]
    entry = entries.get(item_id)
    if entry is None:
        entry = {
            "id": item_id,
            "question": question,
            "answers": {},
        }
        if source_id is not None:
            entry["source_id"] = source_id
            entry["question_id"] = source_id
        entries[item_id] = entry
        return entry

    if entry.get("question") != question:
        raise CacheConflictError(
            f"Question text for cache ID {item_id!r} does not match"
        )
    if source_id is not None and entry.get("source_id", source_id) != source_id:
        raise CacheConflictError(f"Source ID for cache ID {item_id!r} does not match")
    if source_id is not None:
        entry.setdefault("source_id", source_id)
        entry.setdefault("question_id", source_id)
    answers = entry.setdefault("answers", {})
    if not isinstance(answers, dict):
        raise ValueError(f"Answers for cache ID {item_id!r} are invalid")
    return entry


def has_success(cache: dict[str, Any], item_id: str, tier: str) -> bool:
    entry = cache["entries"].get(item_id, {})
    record = entry.get("answers", {}).get(tier, {})
    return isinstance(record, dict) and isinstance(record.get("answer"), str)


def store_success(
    cache: dict[str, Any],
    *,
    item_id: str,
    source_id: str | None,
    question: str,
    tier: str,
    record: dict[str, Any],
) -> bool:
    entry = ensure_entry(
        cache, item_id=item_id, question=question, source_id=source_id
    )
    existing = entry["answers"].get(tier)
    if existing is not None:
        if existing.get("answer") != record.get("answer"):
            raise CacheConflictError(
                f"Refusing to replace successful {tier!r} answer for {item_id!r}"
            )
        if record.get("imported") and not existing.get("imported"):
            existing["imported"] = True
        return False
    entry["answers"][tier] = record
    return True


def store_failure(
    cache: dict[str, Any],
    *,
    item_id: str,
    source_id: str | None,
    question: str,
    tier: str,
    failure: dict[str, Any],
) -> None:
    ensure_entry(cache, item_id=item_id, question=question, source_id=source_id)
    cache["failures"].append(
        {"id": item_id, "tier": tier, **failure}
    )
