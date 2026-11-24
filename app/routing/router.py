import json
import time
import logging
from typing import Optional

from app.config import Settings, get_settings
from app.cost import attempt_cost_parts, estimate_request_cost
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
logger = logging.getLogger(__name__)


class RoutingConfigurationError(ValueError):
    """The requested tier is disabled by deployment configuration."""


def _automatic_tier(score: int, threshold: int, settings: Settings) -> int:
    enabled = sorted(settings.enabled_tiers)
    preferred = choose_tier(score, threshold)
    if preferred in enabled:
        return preferred
    if preferred == 0:
        return min(enabled)
    return max(enabled)


def ask_tier(question: str, tier, settings: Settings) -> ProviderResponse:
    return get_provider(settings, tier).generate(question, tier.model_id)


def _attempt_success(
    tier_id: int, response: ProviderResponse, provider: Optional[str] = None
) -> dict:
    return {
        "tier": tier_id,
        "model_id": response.model_id,
        "provider": provider or response.provider,
        "simulated": response.provider == "fake",
        "outcome": "success",
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "token_source": response.token_source,
        "input_token_source": getattr(response, "input_token_source", None),
        "output_token_source": getattr(response, "output_token_source", None),
        "cached_input_tokens": getattr(response, "cached_input_tokens", None),
        "latency_ms": response.latency_ms,
        "latency_seconds": response.latency_seconds,
        "cost_usd": None,
    }


def _attempt_failure(tier_id: int, model_id: str, provider: str, error) -> dict:
    usage = getattr(error, "usage", {})
    input_tokens = usage.get("prompt_tokens", usage.get("prompt_eval_count"))
    output_tokens = usage.get("completion_tokens", usage.get("eval_count"))
    total_tokens = usage.get("total_tokens")
    if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0:
        input_tokens = None
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 0:
        output_tokens = None
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens < 0:
        total_tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
    return {
        "tier": tier_id,
        "model_id": getattr(error, "model_id", None) or model_id,
        "provider": provider,
        "outcome": "failure",
        "error_type": type(error).__name__,
        "status_code": getattr(error, "status_code", None),
        "error": str(error),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "token_source": "provider" if input_tokens is not None and output_tokens is not None else None,
        "input_token_source": "provider" if input_tokens is not None else None,
        "output_token_source": "provider" if output_tokens is not None else None,
        "cached_input_tokens": None,
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
    automatic_tier_id = _automatic_tier(score, active_threshold, settings)
    selected_tier_id = (
        automatic_tier_id if force_model is None else OVERRIDE_TIERS[force_model]
    )
    if selected_tier_id not in settings.enabled_tiers:
        tier_name = settings.tier(selected_tier_id).name
        raise RoutingConfigurationError(
            f"The {tier_name} tier is disabled in this deployment. "
            "Choose Automatic, Cheap, or Strong, or enable tier "
            f"{selected_tier_id} with ROUTER_ENABLED_TIERS."
        )
    if selected_tier_id == 0:
        # A local outage may escalate to a configured cloud provider.
        candidate_ids = [
            tier_id for tier_id in sorted(settings.enabled_tiers)
            if tier_id >= selected_tier_id
        ]
    else:
        candidate_ids = [
            tier_id for tier_id in sorted(settings.enabled_tiers, reverse=True)
            if tier_id <= selected_tier_id
        ]
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
            attempts.append(_attempt_success(tier_id, response, tier.provider))
            break
        except ProviderError as error:
            logger.warning(
                "Provider attempt failed: tier=%s provider=%s model=%s error_type=%s status_code=%s",
                tier_id, tier.provider, tier.model_id, type(error).__name__,
                getattr(error, "status_code", None),
            )
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
                    attempts=attempts,
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
                    attempts=attempts,
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
    for attempt in attempts:
        input_cost, output_cost = attempt_cost_parts(settings, attempt)
        attempt["input_cost_usd"] = input_cost
        attempt["output_cost_usd"] = output_cost
        attempt["cost_usd"] = (
            input_cost + output_cost
            if input_cost is not None and output_cost is not None else None
        )
    successful_attempts = [item for item in attempts if item["outcome"] == "success"]
    tokens_known = all(
        item.get("input_tokens") is not None and item.get("output_tokens") is not None
        for item in attempts
    )
    has_unknown_attempt = any(
        item["outcome"] != "success"
        and (item.get("input_tokens") is None or item.get("output_tokens") is None)
        for item in attempts
    )
    has_missing_usage = any(
        item["provider"] != "ollama"
        and (item.get("input_tokens") is None or item.get("output_tokens") is None)
        for item in attempts
    )
    has_unpriced_attempt = any(
        item.get("cost_usd") is None
        and item.get("input_tokens") is not None
        and item.get("output_tokens") is not None
        for item in attempts
    )
    actual_cost = (
        sum(item["cost_usd"] for item in attempts)
        if not has_unpriced_attempt and not has_unknown_attempt
        and not has_missing_usage
        else None
    )
    total_input_tokens = (
        sum(item["input_tokens"] for item in attempts)
        if tokens_known else None
    )
    total_output_tokens = (
        sum(item["output_tokens"] for item in attempts)
        if tokens_known else None
    )
    total_tokens = (
        sum(item["total_tokens"] for item in attempts)
        if all(item.get("total_tokens") is not None for item in attempts)
        else None
    )
    # The baseline uses known usage from every provider attempt.
    baseline = estimate_request_cost(
        settings, 2, total_input_tokens, total_output_tokens
    ).baseline_cost_usd if tokens_known else None
    cost_status = (
        "unknown" if has_unknown_attempt
        else "unavailable_usage" if has_missing_usage
        else "unavailable_pricing" if has_unpriced_attempt
        else "simulated" if successful_attempts and all(item.get("simulated") for item in successful_attempts)
        else "local" if successful_attempts and all(
            item["provider"] == "ollama"
            or settings.tier(item["tier"]).provider == "ollama"
            for item in successful_attempts
        )
        else "estimated" if any(
            item.get("token_source") == "estimated"
            for item in successful_attempts
        )
        else "actual"
    )
    pricing_version = json.dumps({
        item.model_id: {
            "input_usd_per_million": item.input_usd_per_million,
            "output_usd_per_million": item.output_usd_per_million,
        }
        for item in settings.tiers
    }, sort_keys=True)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    result = {
        "question": question,
        "complexity": complexity,
        "score": score,
        "routed_to": settings.tier(automatic_tier_id).name,
        "model": response.model_id,
        "provider": settings.tier(actual_tier_id).provider,
        "model_tier": settings.tier(actual_tier_id).name,
        "tier_id": actual_tier_id,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "api_attempts": attempts,
        "answer": response.answer,
        "reasons": reasons,
        "token_source": response.token_source,
        "provider_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
        },
        "latency_ms": elapsed_ms,
        "provider_latency_ms": response.latency_ms,
        "latency_seconds": response.latency_seconds,
        "cost_usd": actual_cost,
        "baseline_cost_usd": baseline,
        "cost_status": cost_status,
        "pricing_version": pricing_version,
        "baseline_label": "estimated counterfactual always-strong baseline using the same known token counts",
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
        provider=result["provider"],
        fallback=fallback,
        fallback_reason=fallback_reason,
        overridden=result["overridden"],
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        token_source=response.token_source,
        latency_ms=elapsed_ms,
        cost_usd=actual_cost,
        baseline_cost_usd=baseline,
        actual_model=response.model_id,
        total_tokens=total_tokens,
        input_cost_usd=(
            sum(item["input_cost_usd"] for item in attempts)
            if actual_cost is not None
            and all(item.get("input_cost_usd") is not None for item in attempts)
            else None
        ),
        output_cost_usd=(
            sum(item["output_cost_usd"] for item in attempts)
            if actual_cost is not None
            and all(item.get("output_cost_usd") is not None for item in attempts)
            else None
        ),
        cost_currency="USD",
        pricing_version=pricing_version,
        cost_status=cost_status,
        attempts_json=json.dumps(attempts),
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
    attempts: Optional[list[dict]] = None,
    settings: Settings,
) -> None:
    attempts = attempts or []
    for attempt in attempts:
        input_cost, output_cost = attempt_cost_parts(settings, attempt)
        attempt["input_cost_usd"] = input_cost
        attempt["output_cost_usd"] = output_cost
        attempt["cost_usd"] = input_cost + output_cost if input_cost is not None and output_cost is not None else None
    usage_known = bool(attempts) and all(
        item.get("input_tokens") is not None and item.get("output_tokens") is not None
        for item in attempts
    )
    unknown_cost_attempt = any(item.get("cost_usd") is None for item in attempts)
    missing_billable_usage = any(
        item.get("provider") != "ollama"
        and (item.get("input_tokens") is None or item.get("output_tokens") is None)
        for item in attempts
    )
    measured_cost = (
        sum(item["cost_usd"] for item in attempts)
        if attempts and not unknown_cost_attempt and not missing_billable_usage
        else None
    )
    input_tokens = sum(item["input_tokens"] for item in attempts) if usage_known else None
    output_tokens = sum(item["output_tokens"] for item in attempts) if usage_known else None
    total_tokens = sum(item["total_tokens"] for item in attempts) if attempts and all(item.get("total_tokens") is not None for item in attempts) else None
    baseline = estimate_request_cost(settings, 2, input_tokens, output_tokens).baseline_cost_usd if usage_known else None
    cost_status = (
        "unknown" if missing_billable_usage
        else "unavailable_pricing" if unknown_cost_attempt
        else "local" if attempts and all(
            item["provider"] == "ollama"
            or item.get("provider") == "ollama"
            for item in attempts
        )
        else "estimated" if any(item.get("token_source") == "estimated" for item in attempts)
        else "actual"
    )
    pricing_version = json.dumps({
        item.model_id: {
            "input_usd_per_million": item.input_usd_per_million,
            "output_usd_per_million": item.output_usd_per_million,
        }
        for item in settings.tiers
    }, sort_keys=True)
    log_request(
        question=question,
        score=score,
        reasons=reasons,
        routed_to=routed_to,
        model=model,
        provider=attempts[-1].get("provider") if attempts else None,
        fallback=fallback,
        fallback_reason=fallback_reason,
        overridden=overridden,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_source="provider" if usage_known else "unavailable",
        latency_ms=latency_ms,
        cost_usd=measured_cost,
        baseline_cost_usd=baseline,
        total_tokens=total_tokens,
        input_cost_usd=sum(item["input_cost_usd"] for item in attempts) if measured_cost is not None else None,
        output_cost_usd=sum(item["output_cost_usd"] for item in attempts) if measured_cost is not None else None,
        cost_currency="USD",
        actual_model=None,
        pricing_version=pricing_version,
        cost_status=cost_status,
        attempts_json=json.dumps(attempts),
        request_status="failure",
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
