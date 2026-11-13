from dataclasses import dataclass
from typing import Optional

from app.config import Settings, TierSettings


@dataclass(frozen=True)
class CostEstimate:
    cost_usd: Optional[float]
    baseline_cost_usd: Optional[float]
    baseline_label: str = "counterfactual: tier 2 at the same token counts"


def cost_for_tier(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    tier: TierSettings,
) -> Optional[float]:
    if tier.provider == "ollama":
        return 0.0
    if (
        input_tokens is None
        or output_tokens is None
        or tier.input_usd_per_million is None
        or tier.output_usd_per_million is None
    ):
        return None
    return (
        input_tokens * tier.input_usd_per_million
        + output_tokens * tier.output_usd_per_million
    ) / 1_000_000


def estimate_request_cost(
    settings: Settings,
    tier_id: int,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> CostEstimate:
    return CostEstimate(
        cost_usd=cost_for_tier(
            input_tokens, output_tokens, settings.tier(tier_id)
        ),
        baseline_cost_usd=cost_for_tier(
            input_tokens, output_tokens, settings.tier(2)
        ),
    )
