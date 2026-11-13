import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Literal, Optional

import requests
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential


TOKEN_SOURCE = Literal["provider", "estimated"]
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
    ) -> None:
        super().__init__(message, status_code)
        self._recoverable_override = recoverable
        self.retry_after_seconds = retry_after_seconds

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
    input_tokens: int
    output_tokens: int
    token_source: TOKEN_SOURCE
    latency_ms: int
    model_id: str
    provider: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def latency_seconds(self) -> float:
        return self.latency_ms / 1000


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def make_response(
    *,
    answer: str,
    question: str,
    model_id: str,
    provider: str,
    started_at: float,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> ProviderResponse:
    from_provider = input_tokens is not None and output_tokens is not None
    return ProviderResponse(
        answer=answer,
        input_tokens=(
            input_tokens if input_tokens is not None else estimate_tokens(question)
        ),
        output_tokens=(
            output_tokens if output_tokens is not None else estimate_tokens(answer)
        ),
        token_source="provider" if from_provider else "estimated",
        latency_ms=max(0, round((time.perf_counter() - started_at) * 1000)),
        model_id=model_id,
        provider=provider,
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
                    raise ProviderHTTPError(
                        f"HTTP {status_code}",
                        status_code=status_code,
                        retry_after_seconds=_parse_retry_after(retry_after),
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
