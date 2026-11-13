import csv
import os
import time
from datetime import datetime
from typing import Optional

from app.config import get_settings
from app.cost import estimate_request_cost
from app.routing.router import route_question


LOG_FILE = "routing_log_groq.csv"
LOG_FIELDS = [
    "timestamp",
    "question",
    "complexity",
    "score",
    "automatic_tier",
    "model_tier",
    "model_id",
    "fallback",
    "fallback_reason",
    "provider_input_tokens",
    "provider_output_tokens",
    "estimated_input_tokens",
    "estimated_output_tokens",
    "token_source",
    "input_token_source",
    "output_token_source",
    "estimated_cost_usd",
    "estimated_strong_baseline_usd",
    "estimated_savings_usd",
    "latency_seconds",
    "provider_latency_seconds",
]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def estimate_cost(model_id, question, answer, provider_usage=None):
    settings = get_settings()
    tier_id = next(
        (tier.id for tier in settings.tiers if tier.model_id == model_id),
        None,
    )
    usage = provider_usage or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    input_cost_tokens = (
        input_tokens if input_tokens is not None else estimate_tokens(question)
    )
    output_cost_tokens = (
        output_tokens if output_tokens is not None else estimate_tokens(answer)
    )
    if tier_id is None:
        cost_usd = baseline_cost_usd = None
    else:
        estimate = estimate_request_cost(
            settings, tier_id, input_cost_tokens, output_cost_tokens
        )
        cost_usd = estimate.cost_usd
        baseline_cost_usd = estimate.baseline_cost_usd
    token_source = (
        "provider-reported"
        if input_tokens is not None and output_tokens is not None
        else "estimated (characters / 4)"
    )
    return {
        "provider_input_tokens": input_tokens,
        "provider_output_tokens": output_tokens,
        "estimated_input_tokens": estimate_tokens(question),
        "estimated_output_tokens": estimate_tokens(answer),
        "cost_input_tokens": input_cost_tokens,
        "cost_output_tokens": output_cost_tokens,
        "input_token_source": (
            "provider-reported" if input_tokens is not None
            else "estimated (characters / 4)"
        ),
        "output_token_source": (
            "provider-reported" if output_tokens is not None
            else "estimated (characters / 4)"
        ),
        "token_source": token_source,
        "estimated_cost_usd": cost_usd,
        "estimated_strong_baseline_usd": baseline_cost_usd,
        "estimated_savings_usd": (
            baseline_cost_usd - cost_usd
            if baseline_cost_usd is not None and cost_usd is not None
            else None
        ),
    }


def _format_cost(cost):
    return round(cost, 8) if cost is not None else "unavailable"


def _display_cost(cost):
    return (
        f"${cost:.8f}"
        if isinstance(cost, (int, float))
        else "unavailable (configure model rates in .env)"
    )


def save_log(result):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def track_question(
    question: str,
    force_model: Optional[str] = None,
    display: bool = True,
    threshold: Optional[int] = None,
) -> dict:
    started_at = time.perf_counter()
    result = route_question(
        question,
        force_model=force_model,
        display=display,
        threshold=threshold,
    )
    elapsed = time.perf_counter() - started_at
    cost = estimate_cost(
        result["model"],
        question,
        result["answer"],
        result["provider_usage"],
    )
    settings = get_settings()
    log_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question if settings.log_prompts else "",
        "complexity": result["complexity"],
        "score": result["score"],
        "automatic_tier": result["routed_to"],
        "model_tier": result["model_tier"],
        "model_id": result["model"],
        "fallback": result["fallback"],
        "fallback_reason": result["fallback_reason"],
        "provider_input_tokens": cost["provider_input_tokens"],
        "provider_output_tokens": cost["provider_output_tokens"],
        "estimated_input_tokens": cost["estimated_input_tokens"],
        "estimated_output_tokens": cost["estimated_output_tokens"],
        "token_source": result["token_source"],
        "input_token_source": cost["input_token_source"],
        "output_token_source": cost["output_token_source"],
        "estimated_cost_usd": _format_cost(cost["estimated_cost_usd"]),
        "estimated_strong_baseline_usd": _format_cost(
            cost["estimated_strong_baseline_usd"]
        ),
        "estimated_savings_usd": _format_cost(cost["estimated_savings_usd"]),
        "latency_seconds": round(elapsed, 3),
        "provider_latency_seconds": result["latency_seconds"],
    }
    save_log(log_data)
    if display:
        print("\n" + "=" * 60)
        print("TRACKING INFORMATION")
        print("=" * 60)
        print(f"Model             : {result['model']}")
        print(f"Complexity        : {result['complexity']}")
        print(f"Token source       : {result['token_source']}")
        print(f"Estimated cost     : {_display_cost(cost['estimated_cost_usd'])}")
        print(
            "Counterfactual tier 2 baseline : "
            f"{_display_cost(cost['estimated_strong_baseline_usd'])}"
        )
        print(f"Latency            : {elapsed:.3f} seconds")
        print(f"Log file           : {LOG_FILE}")
        print("=" * 60)
    return result


def main() -> None:
    print("=" * 60)
    print("LLM COST TRACKER")
    print("Type 'exit' to stop.")
    print("=" * 60)
    while True:
        question = input("\nAsk a question: ").strip()
        force_model = None
        for prefix, tier in (
            ("force local:", "local"),
            ("force cheap:", "cheap"),
            ("force strong:", "strong"),
            ("force groq:", "groq"),
        ):
            if question.lower().startswith(prefix):
                force_model = tier
                question = question[len(prefix):].strip()
                break
        if question.lower() == "exit":
            print("\nExiting tracker.")
            break
        if not question:
            print("Please enter a question.")
            continue
        try:
            track_question(question, force_model=force_model)
        except Exception as error:
            print("\nERROR:")
            print(error)
