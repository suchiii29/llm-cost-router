import time
from typing import Optional

from app.config import Settings, get_settings
from app.cost import estimate_request_cost
from app.providers import get_provider
from app.providers.base import ProviderError, ProviderResponse
from app.routing.complexity import calculate_complexity
from app.routing.policy import choose_tier
from app.storage.db import log_request


OVERRIDE_TIERS = {
    "local": 0,
    "cheap": 1,
    "strong": 2,
    "groq": 2,
}


def ask_tier(question: str, tier, settings: Settings) -> ProviderResponse:
    return get_provider(settings, tier).generate(question, tier.model_id)


def _attempt_success(tier_id: int, response: ProviderResponse) -> dict:
    return {
        "tier": tier_id,
        "model_id": response.model_id,
        "provider": response.provider,
        "outcome": "success",
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "token_source": response.token_source,
        "latency_ms": response.latency_ms,
        "latency_seconds": response.latency_seconds,
    }


def _attempt_failure(tier_id: int, model_id: str, provider: str, error) -> dict:
    return {
        "tier": tier_id,
        "model_id": model_id,
        "provider": provider,
        "outcome": "failure",
        "error_type": type(error).__name__,
        "status_code": getattr(error, "status_code", None),
        "error": str(error),
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "token_source": None,
        "latency_ms": None,
        "latency_seconds": None,
    }


def route_question(
    question: str,
    force_model: Optional[str] = None,
    display: bool = True,
    threshold: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> dict:
    settings = settings or get_settings()
    if force_model not in (None, *OVERRIDE_TIERS):
        raise ValueError(
            "force_model must be local, cheap, strong, groq, or None"
        )
    complexity, score, reasons = calculate_complexity(question)
    active_threshold = (
        settings.complexity_threshold if threshold is None else threshold
    )
    automatic_tier_id = choose_tier(score, active_threshold)
    selected_tier_id = (
        automatic_tier_id if force_model is None else OVERRIDE_TIERS[force_model]
    )
    candidate_ids = list(range(selected_tier_id, -1, -1))
    attempts = []
    fallback_reasons = []
    started_at = time.perf_counter()
    response = None
    last_error = None

    if display:
        print("\n" + "=" * 60)
        print("QUERY ROUTING")
        print("=" * 60)
        print(f"Question   : {question}")
        print(f"Complexity : {complexity}")
        print(f"Score      : {score}")
        print("Reasons    : " + (
            "; ".join(reasons)
            if reasons else "No strong complexity indicators detected"
        ))
        if force_model is not None:
            print("[MANUAL OVERRIDE]")

    for index, tier_id in enumerate(candidate_ids):
        tier = settings.tier(tier_id)
        try:
            response = ask_tier(question, tier, settings)
            attempts.append(_attempt_success(tier_id, response))
            break
        except ProviderError as error:
            last_error = error
            attempts.append(
                _attempt_failure(tier_id, tier.model_id, tier.provider, error)
            )
            if not getattr(error, "recoverable", False):
                error.api_attempts = list(attempts)
                _log_failure(
                    question=question,
                    score=score,
                    reasons=reasons,
                    routed_to=settings.tier(automatic_tier_id).name,
                    model=tier.model_id,
                    fallback=len(attempts) > 1,
                    fallback_reason="; ".join(fallback_reasons) or str(error),
                    overridden=force_model is not None,
                    latency_ms=round((time.perf_counter() - started_at) * 1000),
                    settings=settings,
                )
                raise
            fallback_reasons.append(str(error))
            if index == len(candidate_ids) - 1:
                error.api_attempts = list(attempts)
                _log_failure(
                    question=question,
                    score=score,
                    reasons=reasons,
                    routed_to=settings.tier(automatic_tier_id).name,
                    model=tier.model_id,
                    fallback=len(attempts) > 1,
                    fallback_reason="; ".join(fallback_reasons),
                    overridden=force_model is not None,
                    latency_ms=round((time.perf_counter() - started_at) * 1000),
                    settings=settings,
                )
                raise
            if display:
                next_tier = settings.tier(candidate_ids[index + 1])
                print(
                    f"[FALLBACK: {error}; trying tier {next_tier.id} "
                    f"({next_tier.model_id})]"
                )

    if response is None:
        raise last_error or RuntimeError("No model tier was available")

    actual_tier_id = attempts[-1]["tier"]
    fallback = actual_tier_id != selected_tier_id
    fallback_reason = "; ".join(fallback_reasons) or None
    estimate = estimate_request_cost(
        settings,
        actual_tier_id,
        response.input_tokens,
        response.output_tokens,
    )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    result = {
        "question": question,
        "complexity": complexity,
        "score": score,
        "routed_to": settings.tier(automatic_tier_id).name,
        "model": response.model_id,
        "provider": response.provider,
        "model_tier": settings.tier(actual_tier_id).name,
        "tier_id": actual_tier_id,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "api_attempts": attempts,
        "answer": response.answer,
        "reasons": reasons,
        "token_source": response.token_source,
        "provider_usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        },
        "latency_ms": elapsed_ms,
        "provider_latency_ms": response.latency_ms,
        "latency_seconds": response.latency_seconds,
        "cost_usd": estimate.cost_usd,
        "baseline_cost_usd": estimate.baseline_cost_usd,
        "baseline_label": estimate.baseline_label,
    }
    if force_model is not None:
        result["overridden"] = True
    else:
        result["overridden"] = False

    request_id = log_request(
        question=question,
        score=score,
        reasons=reasons,
        routed_to=result["routed_to"],
        model=response.model_id,
        fallback=fallback,
        fallback_reason=fallback_reason,
        overridden=result["overridden"],
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        token_source=response.token_source,
        latency_ms=elapsed_ms,
        cost_usd=estimate.cost_usd,
        baseline_cost_usd=estimate.baseline_cost_usd,
        cache_hit=False,
        settings=settings,
    )
    result["request_id"] = request_id

    if display:
        print(f"Model      : tier {actual_tier_id} {response.provider} ({response.model_id})")
        print("\nANSWER")
        print("-" * 60)
        print(response.answer)
        print("=" * 60)
    return result


def _log_failure(
    *,
    question: str,
    score: int,
    reasons: list[str],
    routed_to: str,
    model: Optional[str],
    fallback: bool,
    fallback_reason: Optional[str],
    overridden: bool,
    latency_ms: int,
    settings: Settings,
) -> None:
    log_request(
        question=question,
        score=score,
        reasons=reasons,
        routed_to=routed_to,
        model=model,
        fallback=fallback,
        fallback_reason=fallback_reason,
        overridden=overridden,
        input_tokens=None,
        output_tokens=None,
        token_source="estimated",
        latency_ms=latency_ms,
        cost_usd=None,
        baseline_cost_usd=None,
        settings=settings,
    )


def main() -> None:
    print("=" * 60)
    print("LLM COST ROUTER")
    print("Type 'exit' to stop.")
    print("=" * 60)
    while True:
        question = input("\nAsk a question: ").strip()
        force_model = None
        for prefix, tier in (
            ("force cheap:", "cheap"),
            ("force strong:", "strong"),
            ("force local:", "local"),
            ("force groq:", "strong"),
        ):
            if question.lower().startswith(prefix):
                force_model = tier
                question = question[len(prefix):].strip()
                break
        if question.lower() == "exit":
            print("\nExiting router.")
            break
        if not question:
            print("Please enter a question.")
            continue
        try:
            route_question(question, force_model=force_model)
        except Exception as error:
            print("\nERROR:")
            print(error)
