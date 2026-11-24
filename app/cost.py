from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from app.config import Settings, TierSettings


@dataclass(frozen=True)
class CostEstimate:
    cost_usd: Optional[float]
    baseline_cost_usd: Optional[float]
    baseline_label: str = "estimated counterfactual always-strong baseline at the same token counts"


def cost_for_tier(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    tier: TierSettings,
) -> Optional[float]:
    if tier.provider == "ollama":
        return 0.0
    if input_tokens is None or output_tokens is None:
        return None
    if tier.input_usd_per_million is None or tier.output_usd_per_million is None:
        return None
    input_cost = Decimal(input_tokens) * Decimal(str(tier.input_usd_per_million))
    output_cost = Decimal(output_tokens) * Decimal(str(tier.output_usd_per_million))
    return float((input_cost + output_cost) / Decimal(1_000_000))


def estimate_request_cost(
    settings: Settings,
    tier_id: int,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> CostEstimate:
    return CostEstimate(
        cost_usd=cost_for_tier(input_tokens, output_tokens, settings.tier(tier_id)),
        baseline_cost_usd=cost_for_tier(
            input_tokens, output_tokens, settings.tier(2)
        ),
    )


def estimate_attempt_cost(settings: Settings, attempt: dict) -> Optional[float]:
    """Price a single successful provider attempt using its actual model ID."""
    tier = next(
        (item for item in settings.tiers if item.model_id == attempt.get("model_id")),
        None,
    )
    if tier is None:
        return None
    return cost_for_tier(
        attempt.get("input_tokens"), attempt.get("output_tokens"), tier
    )


def attempt_cost_parts(settings: Settings, attempt: dict) -> tuple[Optional[float], Optional[float]]:
    tier = next(
        (item for item in settings.tiers if item.model_id == attempt.get("model_id")),
        None,
    )
    if tier is None:
        return None, None
    if tier.provider == "ollama":
        return 0.0, 0.0
    input_tokens = attempt.get("input_tokens")
    output_tokens = attempt.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None, None
    if tier.input_usd_per_million is None or tier.output_usd_per_million is None:
        return None, None
    million = Decimal(1_000_000)
    return (
        float(Decimal(input_tokens) * Decimal(str(tier.input_usd_per_million)) / million),
        float(Decimal(output_tokens) * Decimal(str(tier.output_usd_per_million)) / million),
    )
