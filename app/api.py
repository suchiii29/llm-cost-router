"""Authenticated HTTP API for the cost router."""

from collections import defaultdict, deque
from datetime import datetime
import hashlib
import hmac
import json
import logging
import math
import threading
import time
from typing import Literal, Optional

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.providers.base import (
    ProviderConfigurationError,
    ProviderError,
    ProviderTransportError,
)
from app.routing.router import RoutingConfigurationError, route_question
from app.storage.db import RequestRecord, get_engine, initialize_database


class RouteRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    force_model: Optional[Literal["local", "cheap", "strong"]] = None
    threshold: Optional[int] = None


app = FastAPI(title="LLM Cost Router API", version="1.0.0")
logger = logging.getLogger(__name__)
_RATE_WINDOW_SECONDS = 60.0
_rate_lock = threading.Lock()
_rate_events: dict[str, deque[float]] = defaultdict(deque)


def _error(status_code: int, code: str, message: str, headers=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, _error_value: RequestValidationError):
    return _error(422, "validation_error", "Request validation failed")


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Request, exc: HTTPException):
    code = {
        401: "unauthorized",
        404: "not_found",
        429: "rate_limited",
        503: "service_unavailable",
    }.get(exc.status_code, "http_error")
    message = {
        401: "A valid API key is required",
        404: "Resource not found",
        429: "Rate limit exceeded",
        503: "Service is unavailable",
    }.get(exc.status_code, "Request failed")
    return _error(exc.status_code, code, message, exc.headers)


@app.exception_handler(ProviderError)
async def provider_error_handler(_request: Request, exc: ProviderError):
    logger.warning(
        "Provider request failed: error_type=%s status_code=%s",
        type(exc).__name__, getattr(exc, "status_code", None),
    )
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        message = "The provider rejected its credentials; check the configured provider key."
    elif status == 404:
        message = "The configured provider model was not found; check the model ID."
    elif status == 429:
        message = "The provider rate limit or quota was reached; retry later."
    elif isinstance(exc, ProviderConfigurationError):
        message = "Provider credentials are not configured; check the backend provider settings."
    elif isinstance(exc, ProviderTransportError):
        message = "The provider could not be reached; check provider availability and timeout settings."
    elif status is not None and status >= 500:
        message = "The provider is temporarily unavailable; retry later."
    else:
        message = "The provider returned an unusable response; check provider configuration."
    return _error(503, "provider_error", message)


@app.exception_handler(RoutingConfigurationError)
async def routing_configuration_error_handler(_request: Request, exc: RoutingConfigurationError):
    return _error(400, "tier_disabled", str(exc))


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, exc: Exception):
    logger.error("Unexpected API error: error_type=%s", type(exc).__name__)
    return _error(500, "internal_error", "An internal error occurred")


def _take_rate_slot(api_key: str, limit: int) -> Optional[int]:
    now = time.monotonic()
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    with _rate_lock:
        events = _rate_events[key_hash]
        cutoff = now - _RATE_WINDOW_SECONDS
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= limit:
            retry_after = max(1, math.ceil(_RATE_WINDOW_SECONDS - (now - events[0])))
            return retry_after
        events.append(now)
    return None


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> str:
    expected = (
        settings.router_api_key.get_secret_value().encode("utf-8")
        if settings.router_api_key is not None
        else b""
    )
    supplied = (x_api_key or "").encode("utf-8")
    matches = hmac.compare_digest(supplied, expected)
    if not expected or not x_api_key or not matches:
        raise HTTPException(status_code=401)
    retry_after = _take_rate_slot(x_api_key, settings.api_rate_limit_per_minute)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    return x_api_key


@app.post("/v1/route", dependencies=[Depends(require_api_key)])
def route_api(body: RouteRequest, settings: Settings = Depends(get_settings)):
    result = route_question(
        body.question,
        force_model=body.force_model,
        threshold=body.threshold,
        display=False,
        settings=settings,
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


@app.get("/v1/requests", dependencies=[Depends(require_api_key)])
def request_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    settings: Settings = Depends(get_settings),
):
    if not settings.db_url.startswith("sqlite"):
        raise HTTPException(status_code=503)
    initialize_database(settings)
    statement = (
        select(RequestRecord)
        .order_by(RequestRecord.created_at.desc(), RequestRecord.id.desc())
        .limit(limit)
        .offset(offset)
    )
    with Session(get_engine(settings.db_url)) as session:
        records = session.scalars(statement).all()
    return {
        "items": [
            {
                "request_id": record.id,
                "created_at": record.created_at.isoformat()
                if isinstance(record.created_at, datetime)
                else record.created_at,
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
            for record in records
        ],
        "limit": limit,
        "offset": offset,
    }


@app.get("/v1/config", dependencies=[Depends(require_api_key)])
def public_config(settings: Settings = Depends(get_settings)):
    return {"tiers": [{"id": tier.id} for tier in settings.tiers]}


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    try:
        response = requests.get(
            f"{settings.ollama_base_url}/api/tags", timeout=1.0
        )
        ollama_reachable = 200 <= response.status_code < 400
    except requests.RequestException:
        ollama_reachable = False
    tiers = [
        {"id": tier.id, "name": tier.name, "provider": tier.provider,
         "model_id": tier.model_id, "enabled": tier.id in settings.enabled_tiers}
        for tier in settings.tiers
    ]
    groq_enabled = any(t.provider == "groq" and t.id in settings.enabled_tiers for t in settings.tiers)
    groq_configured = (
        settings.groq_api_key is not None
        and bool(settings.groq_api_key.get_secret_value().strip())
    )
    return {
        "status": "ok",
        "api_status": "ok",
        "deployment_profile": settings.deployment_profile,
        "provider_mode": settings.provider_mode,
        "configured_tiers": tiers,
        "enabled_routing_tiers": sorted(settings.enabled_tiers),
        "providers": {
            "groq": {"enabled": groq_enabled, "credentials_configured": groq_configured,
                     "configured": (not groq_enabled or groq_configured), "available": None},
            "ollama": {"enabled": any(t.provider == "ollama" and t.id in settings.enabled_tiers for t in settings.tiers),
                       "reachable": ollama_reachable},
        },
        "ollama_reachable": ollama_reachable,
        "groq_key_configured": groq_configured,
    }
