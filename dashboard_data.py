"""SQLite reads and explicit dashboard submission helpers."""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from sqlalchemy import inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.config import Settings, TierSettings, get_settings
from app.routing.router import route_question
from app.storage.db import RequestRecord, get_engine


def _row_dict(record: RequestRecord) -> dict[str, Any]:
    return {
        "request_id": record.id,
        "created_at": record.created_at,
        "question_hash": record.question_hash,
        "question_text": record.question_text,
        "score": record.score,
        "reasons": json.loads(record.reasons_json),
        "routed_to": record.routed_to,
        "model": record.model,
        "fallback": record.fallback,
        "fallback_reason": record.fallback_reason,
        "overridden": record.overridden,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "token_source": record.token_source,
        "latency_ms": record.latency_ms,
        "cost_usd": record.cost_usd,
        "baseline_cost_usd": record.baseline_cost_usd,
        "cache_hit": record.cache_hit,
    }


def load_request_history(
    settings: Optional[Settings] = None,
    *,
    limit: int = 1000,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Read recent request records without creating tables or calling providers."""
    settings = settings or get_settings()
    if not settings.db_url.startswith("sqlite"):
        raise ValueError("Dashboard history requires a SQLite database")

    database = make_url(settings.db_url).database
    if database not in (None, "", ":memory:"):
        database_path = Path(database)
        if not database_path.is_absolute():
            database_path = Path.cwd() / database_path
        if not database_path.exists():
            return []

    try:
        engine = get_engine(settings.db_url)
        with engine.connect() as connection:
            if not inspect(connection).has_table(RequestRecord.__tablename__):
                return []
        statement = (
            select(RequestRecord)
            .order_by(RequestRecord.created_at.desc(), RequestRecord.id.desc())
            .limit(limit)
            .offset(offset)
        )
        with Session(engine) as session:
            return [_row_dict(row) for row in session.scalars(statement).all()]
    except SQLAlchemyError as error:
        raise RuntimeError("Could not read dashboard request history from SQLite") from error


def load_api_config(
    *, api_url: str, api_key: str, timeout: float = 15
) -> dict[str, Any]:
    """Read backend tier and rate configuration without contacting a provider."""
    if not api_key:
        raise ValueError("Set ROUTER_API_KEY to read backend request history")
    try:
        response = requests.get(
            f"{api_url.rstrip('/')}/v1/config",
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )
    except requests.RequestException as error:
        raise RuntimeError("Could not reach the router API configuration endpoint") from error
    if response.status_code >= 400:
        raise RuntimeError(f"Router API configuration returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except (ValueError, requests.exceptions.JSONDecodeError) as error:
        raise ValueError("Router API returned invalid configuration JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("tiers"), list):
        raise ValueError("Router API returned invalid tier configuration")
    return payload


def settings_with_api_tiers(
    settings: Settings, api_config: dict[str, Any]
) -> Settings:
    """Use backend tier IDs/models/rates for interpreting its persisted history."""
    remote_tiers = api_config.get("tiers")
    if not isinstance(remote_tiers, list):
        raise ValueError("Router API returned invalid tier configuration")
    try:
        tiers = [TierSettings.model_validate(item) for item in remote_tiers]
    except (TypeError, ValueError) as error:
        raise ValueError("Router API returned invalid tier configuration") from error
    if sorted(tier.id for tier in tiers) != [0, 1, 2]:
        raise ValueError("Router API returned an incomplete tier configuration")
    return settings.model_copy(update={"tiers": tiers})


def load_api_request_history(
    *, api_url: str, api_key: str, limit: int = 1000, page_size: int = 1000,
    timeout: float = 15,
) -> list[dict[str, Any]]:
    """Retrieve persisted backend rows using the authenticated, paginated API."""
    if not api_key:
        raise ValueError("Set ROUTER_API_KEY to read backend request history")
    if limit < 1 or page_size < 1:
        raise ValueError("History limit and page size must be positive")
    items: list[dict[str, Any]] = []
    for offset in range(0, limit, page_size):
        page_limit = min(page_size, limit - offset)
        try:
            response = requests.get(
                f"{api_url.rstrip('/')}/v1/requests",
                headers={"X-API-Key": api_key},
                params={"limit": page_limit, "offset": offset},
                timeout=timeout,
            )
        except requests.RequestException as error:
            raise RuntimeError("Could not reach the router API request history") from error
        if response.status_code >= 400:
            raise RuntimeError(f"Router API history returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except (ValueError, requests.exceptions.JSONDecodeError) as error:
            raise ValueError("Router API returned invalid history JSON") from error
        page = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(page, list) or any(not isinstance(row, dict) for row in page):
            raise ValueError("Router API returned invalid request history")
        items.extend(page)
        if len(page) < page_limit:
            break
    return items


def merge_request_histories(
    *histories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine backend live rows and local replay rows without double counting."""
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for history in histories:
        for row in history:
            created_at = row.get("created_at")
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            identity = (
                row.get("request_id"), created_at, row.get("question_hash"),
                row.get("model"), row.get("score"),
            )
            merged.setdefault(identity, row)

    def sort_key(row: dict[str, Any]) -> str:
        value = row.get("created_at")
        return value.isoformat() if isinstance(value, datetime) else str(value or "")

    return sorted(merged.values(), key=sort_key, reverse=True)


def missing_cost_rate_settings(settings: Settings) -> list[str]:
    """Return backend environment variable names for unset Groq token rates."""
    variables = {
        1: (
            "GROQ_CHEAP_INPUT_USD_PER_MILLION",
            "GROQ_CHEAP_OUTPUT_USD_PER_MILLION",
        ),
        2: (
            "GROQ_STRONG_INPUT_USD_PER_MILLION",
            "GROQ_STRONG_OUTPUT_USD_PER_MILLION",
        ),
    }
    missing = []
    for tier_id, field_names in (
        (1, ("input_usd_per_million", "output_usd_per_million")),
        (2, ("input_usd_per_million", "output_usd_per_million")),
    ):
        tier = settings.tier(tier_id)
        for field_name, env_name in zip(field_names, variables[tier_id]):
            if getattr(tier, field_name) is None:
                missing.append(env_name)
    return missing


def request_tier(row: dict[str, Any], settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    model = row.get("model")
    for tier in settings.tiers:
        if tier.model_id == model:
            return f"tier {tier.id} · {tier.name}"
    return "unknown tier"


def served_tier_label(row: dict[str, Any]) -> str:
    """Format the tier actually reported by the serving API, without inference."""
    tier_id = row.get("tier_id")
    model_tier = row.get("model_tier")
    if tier_id is None or not model_tier:
        return "Unavailable"
    return f"tier {tier_id} · {model_tier}"


def summarize_requests(
    rows: list[dict[str, Any]], settings: Optional[Settings] = None
) -> dict[str, Any]:
    settings = settings or get_settings()
    count = len(rows)
    tiers = {tier.id: 0 for tier in settings.tiers}
    tier_for_model = {tier.model_id: tier.id for tier in settings.tiers}
    latencies = []
    for row in rows:
        tier_id = tier_for_model.get(row.get("model"))
        if tier_id in tiers:
            tiers[tier_id] += 1
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)) and latency >= 0:
            latencies.append(float(latency))

    def total(field: str) -> Optional[float]:
        values = [row.get(field) for row in rows]
        if not values or any(value is None for value in values):
            return None
        return sum(float(value) for value in values)

    def percentile(values: list[float], fraction: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "requests": count,
        "tier_counts": tiers,
        "tier_shares": {
            tier_id: (tier_count / count if count else 0.0)
            for tier_id, tier_count in tiers.items()
        },
        "fallbacks": sum(bool(row.get("fallback")) for row in rows),
        "overrides": sum(bool(row.get("overridden")) for row in rows),
        "estimated_cost_usd": total("cost_usd"),
        "counterfactual_strong_cost_usd": total("baseline_cost_usd"),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
    }


def filter_requests(
    rows: list[dict[str, Any]],
    *,
    tier: str = "All tiers",
    fallback: str = "All",
    overridden: str = "All",
    search: str = "",
    settings: Optional[Settings] = None,
) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    needle = search.strip().casefold()
    filtered = []
    for row in rows:
        row_tier = request_tier(row, settings)
        if tier != "All tiers" and row_tier != tier:
            continue
        if fallback == "Yes" and not row.get("fallback"):
            continue
        if fallback == "No" and row.get("fallback"):
            continue
        if overridden == "Yes" and not row.get("overridden"):
            continue
        if overridden == "No" and row.get("overridden"):
            continue
        searchable = " ".join(
            [
                str(row.get("request_id", "")),
                str(row.get("question_text") or ""),
                str(row.get("model") or ""),
                str(row.get("routed_to") or ""),
                " ".join(row.get("reasons") or []),
            ]
        ).casefold()
        if needle and needle not in searchable:
            continue
        filtered.append(row)
    return filtered


def format_cost(value: Optional[float]) -> str:
    if value is None:
        return "Cost unavailable"
    return f"${value:.8f} (estimated)"


def submit_to_api(
    question: str,
    *,
    api_url: str,
    api_key: str,
    force_model: Optional[str],
    threshold: int,
    timeout: float = 180,
) -> dict[str, Any]:
    """Send one explicit user submission to the configured router API."""
    if not api_key:
        raise ValueError("Set ROUTER_API_KEY to use the router API")
    payload: dict[str, Any] = {"question": question, "threshold": threshold}
    if force_model is not None:
        payload["force_model"] = force_model
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/v1/route",
            headers={"X-API-Key": api_key},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as error:
        raise RuntimeError("Could not reach the router API") from error
    if response.status_code >= 400:
        try:
            error_payload = response.json().get("error", {})
            message = error_payload.get("message")
        except (ValueError, AttributeError):
            message = None
        raise RuntimeError(
            message or f"Router API returned HTTP {response.status_code}"
        )
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("Router API returned an invalid response")
    return result


def submit_in_replay_mode(
    question: str,
    *,
    settings: Optional[Settings] = None,
    force_model: Optional[str] = None,
    threshold: Optional[int] = None,
) -> dict[str, Any]:
    """Route through the local deterministic fake provider without networking."""
    settings = settings or get_settings()
    fake_settings = settings.model_copy(update={"provider_mode": "fake"})
    result = route_question(
        question,
        force_model=force_model,
        threshold=threshold,
        display=False,
        settings=fake_settings,
    )
    usage = result["provider_usage"]
    return {
        "answer": result["answer"],
        "model": result["model"],
        "model_tier": result["model_tier"],
        "tier_id": result["tier_id"],
        "provider": result["provider"],
        "routed_to": result["routed_to"],
        "score": result["score"],
        "reasons": result["reasons"],
        "fallback": result["fallback"],
        "fallback_reason": result["fallback_reason"],
        "overridden": result["overridden"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "token_source": result["token_source"],
        "latency_ms": result["latency_ms"],
        "cost_usd": result["cost_usd"],
        "baseline_cost_usd": result["baseline_cost_usd"],
        "request_id": result["request_id"],
    }


def api_url_from_env() -> str:
    return os.getenv("API_URL", "http://localhost:8000")
