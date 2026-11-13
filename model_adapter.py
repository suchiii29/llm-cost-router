"""Compatibility facade over the packaged provider implementations."""

import os
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv

from app.config import get_settings
from app.providers.base import (
    METRICS_FILE,
    ProviderConfigurationError,
    ProviderHTTPError,
    ProviderResponse,
    ProviderTransportError,
    record_metrics as _record_metrics,
)
from app.providers.groq import API_URL, GroqProvider
from app.providers.ollama import OllamaProvider
from model_config import get_model_config


__all__ = [
    "API_URL",
    "METRICS_FILE",
    "GroqAPIError",
    "GroqConfigurationError",
    "ModelResponse",
    "OllamaAPIError",
    "ask_cheap_model",
    "ask_groq_model",
    "ask_ollama_model",
    "ask_strong_model",
    "ask_tier",
    "get_groq_api_key",
    "get_model_config",
    "record_metrics",
    "requests",
]


GroqAPIError = ProviderHTTPError
GroqConfigurationError = ProviderConfigurationError
OllamaAPIError = ProviderTransportError


@dataclass(frozen=True)
class ModelResponse:
    answer: str
    model_id: str
    provider: str
    latency_seconds: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]


def _legacy_response(response: ProviderResponse) -> ModelResponse:
    return ModelResponse(
        answer=response.answer,
        model_id=response.model_id,
        provider=response.provider,
        latency_seconds=response.latency_seconds,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        total_tokens=response.total_tokens,
    )


def get_groq_api_key() -> str:
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise GroqConfigurationError("GROQ_API_KEY is missing")
    return key


def record_metrics(response: ModelResponse | ProviderResponse) -> None:
    if isinstance(response, ModelResponse):
        from app.providers.base import make_response
        import time

        converted = make_response(
            answer=response.answer,
            question="",
            model_id=response.model_id,
            provider=response.provider,
            started_at=time.perf_counter() - response.latency_seconds,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        _record_metrics(converted)
    else:
        _record_metrics(response)


def ask_groq_model(question: str, model_id: str) -> ModelResponse:
    settings = get_settings()
    key = get_groq_api_key()
    provider = GroqProvider(
        key,
        timeout=settings.request_timeout_seconds,
        retry_attempts=settings.retry_attempts,
    )
    return _legacy_response(provider.generate(question, model_id))


def ask_ollama_model(
    question: str,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 120,
) -> ModelResponse:
    settings = get_settings()
    provider = OllamaProvider(
        base_url or settings.ollama_base_url,
        timeout=timeout,
        retry_attempts=settings.retry_attempts,
    )
    return _legacy_response(
        provider.generate(question, model_id or settings.tier(0).model_id)
    )


def ask_tier(question: str, tier) -> ModelResponse:
    if tier.provider == "ollama":
        return ask_ollama_model(question, tier.model_id)
    return ask_groq_model(question, tier.model_id)


def ask_cheap_model(question: str) -> ModelResponse:
    return ask_groq_model(question, get_settings().tier(1).model_id)


def ask_strong_model(question: str) -> ModelResponse:
    return ask_groq_model(question, get_settings().tier(2).model_id)
