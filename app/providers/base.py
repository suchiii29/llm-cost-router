import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Literal, Optional

import requests
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential


TOKEN_SOURCE = Literal["provider", "estimated", "unavailable"]
METRICS_FILE = "model_metrics_groq.jsonl"


class ProviderError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ProviderHTTPError(ProviderError):
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        recoverable: Optional[bool] = None,
        retry_after_seconds: Optional[float] = None,
        usage: Optional[dict] = None,
        model_id: Optional[str] = None,
    ) -> None:
        super().__init__(message, status_code)
        self._recoverable_override = recoverable
        self.retry_after_seconds = retry_after_seconds
        self.usage = usage or {}
        self.model_id = model_id

    @property
    def retryable(self) -> bool:
        return self.status_code == 429 or (
            self.status_code is not None and 500 <= self.status_code <= 599
        )

    @property
    def recoverable(self) -> bool:
        if self._recoverable_override is not None:
            return self._recoverable_override
        return self.retryable


class ProviderTransportError(ProviderError):
    recoverable = True


class ProviderConfigurationError(ProviderError):
    pass


class ProviderResponseError(ProviderError):
    pass


@dataclass(frozen=True)
class ProviderResponse:
    answer: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    token_source: TOKEN_SOURCE
    latency_ms: int
    model_id: str
    provider: str
    reported_total_tokens: Optional[int] = None
    input_token_source: Optional[str] = None
    output_token_source: Optional[str] = None
    cached_input_tokens: Optional[int] = None

    @property
    def total_tokens(self) -> Optional[int]:
        if self.reported_total_tokens is not None:
            return self.reported_total_tokens
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    @property
    def latency_seconds(self) -> float:
        return self.latency_ms / 1000


def _token_count(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def make_response(
    *,
    answer: str,
    question: str,
    model_id: str,
    provider: str,
    started_at: float,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int] = None,
) -> ProviderResponse:
    actual_input = _token_count(input_tokens)
    actual_output = _token_count(output_tokens)
    estimated_input = max(1, len(question) // 4) if question else 0
    estimated_output = max(1, len(answer) // 4) if answer else 0
    final_input = actual_input if actual_input is not None else estimated_input
    final_output = actual_output if actual_output is not None else estimated_output
    input_source = "provider" if actual_input is not None else "estimated"
    output_source = "provider" if actual_output is not None else "estimated"
    token_source = (
        "provider"
        if input_source == "provider" and output_source == "provider"
        else "estimated"
        if input_source == "estimated" or output_source == "estimated"
        else "unavailable"
    )
    return ProviderResponse(
        answer=answer,
        input_tokens=final_input,
        output_tokens=final_output,
        token_source=token_source,
        latency_ms=max(0, round((time.perf_counter() - started_at) * 1000)),
        model_id=model_id,
        provider=provider,
        reported_total_tokens=_token_count(total_tokens),
        input_token_source=input_source,
        output_token_source=output_source,
    )


def record_metrics(response: ProviderResponse) -> None:
    """Append successful live provider-call metadata without prompt/answer text."""
    metrics = {
        "model_id": response.model_id,
        "provider": response.provider,
        "latency_seconds": response.latency_seconds,
        "latency_ms": response.latency_ms,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "token_source": response.token_source,
        "estimated_cost_usd": 0.0 if response.provider == "ollama" else None,
    }
    with open(METRICS_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(metrics) + "\n")


def _parse_retry_after(value: object) -> Optional[float]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


class BaseProvider(ABC):
    def __init__(
        self,
        *,
        timeout: float = 120,
        retry_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.sleep = sleep

    @staticmethod
    def _is_retryable(error: BaseException) -> bool:
        return isinstance(error, ProviderHTTPError) and error.retryable

    @staticmethod
    def _retry_wait(retry_state) -> float:
        error = retry_state.outcome.exception()
        if isinstance(error, ProviderHTTPError) and error.retry_after_seconds is not None:
            return max(0.0, error.retry_after_seconds)
        return wait_exponential(multiplier=0.5, min=0.5, max=4)(retry_state)

    def post_json(
        self,
        url: str,
        payload: dict,
        headers: Optional[dict[str, str]] = None,
    ) -> dict:
        retryer = Retrying(
            retry=retry_if_exception(self._is_retryable),
            wait=self._retry_wait,
            stop=stop_after_attempt(self.retry_attempts),
            sleep=self.sleep,
            reraise=True,
        )
        for attempt in retryer:
            with attempt:
                try:
                    response = requests.post(
                        url, json=payload, headers=headers, timeout=self.timeout
                    )
                except (requests.Timeout, requests.ConnectionError) as error:
                    raise ProviderTransportError(
                        f"Provider connection failed ({type(error).__name__})"
                    ) from error
                except requests.RequestException as error:
                    raise ProviderError(
                        f"Provider request failed ({type(error).__name__})"
                    ) from error
                status_code = response.status_code
                if status_code >= 400:
                    response_headers = getattr(response, "headers", {})
                    retry_after = response_headers.get("Retry-After")
                    try:
                        error_data = response.json()
                    except (requests.exceptions.JSONDecodeError, ValueError):
                        error_data = {}
                    error_usage = error_data.get("usage", {}) if isinstance(error_data, dict) else {}
                    if not isinstance(error_usage, dict):
                        error_usage = {}
                    raise ProviderHTTPError(
                        f"HTTP {status_code}",
                        status_code=status_code,
                        retry_after_seconds=_parse_retry_after(retry_after),
                        usage=error_usage,
                        model_id=error_data.get("model") if isinstance(error_data, dict) else None,
                    )
                try:
                    data = response.json()
                except (requests.exceptions.JSONDecodeError, ValueError) as error:
                    raise ProviderResponseError(
                        "Provider returned invalid JSON"
                    ) from error
                if not isinstance(data, dict):
                    raise ProviderResponseError("Provider returned unexpected JSON")
                return data
        raise RuntimeError("Provider request retry loop ended unexpectedly")

    @abstractmethod
    def generate(self, question: str, model_id: str) -> ProviderResponse:
        raise NotImplementedError
