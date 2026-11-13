"""Backwards-compatible view of the packaged settings."""

from dataclasses import dataclass
from typing import Optional

from app.config import get_settings


@dataclass(frozen=True)
class TierConfig:
    tier_id: int
    name: str
    model_id: str
    provider: str
    input_usd_per_million: Optional[float]
    output_usd_per_million: Optional[float]

    @property
    def id(self):
        return self.tier_id


@dataclass(frozen=True)
class ModelConfig:
    cheap_model_id: str
    strong_model_id: str
    cheap_input_usd_per_million: Optional[float]
    cheap_output_usd_per_million: Optional[float]
    strong_input_usd_per_million: Optional[float]
    strong_output_usd_per_million: Optional[float]
    ollama_model_id: str = "qwen2.5:1.5b"
    ollama_base_url: str = "http://localhost:11434"

    @property
    def tiers(self):
        return (
            TierConfig(0, "local", self.ollama_model_id, "ollama", 0.0, 0.0),
            TierConfig(
                1,
                "cheap",
                self.cheap_model_id,
                "groq",
                self.cheap_input_usd_per_million,
                self.cheap_output_usd_per_million,
            ),
            TierConfig(
                2,
                "strong",
                self.strong_model_id,
                "groq",
                self.strong_input_usd_per_million,
                self.strong_output_usd_per_million,
            ),
        )

    def tier(self, tier_id):
        return self.tiers[tier_id]


def get_model_config() -> ModelConfig:
    settings = get_settings()
    cheap = settings.tier(1)
    strong = settings.tier(2)
    local = settings.tier(0)
    return ModelConfig(
        cheap_model_id=cheap.model_id,
        strong_model_id=strong.model_id,
        cheap_input_usd_per_million=cheap.input_usd_per_million,
        cheap_output_usd_per_million=cheap.output_usd_per_million,
        strong_input_usd_per_million=strong.input_usd_per_million,
        strong_output_usd_per_million=strong.output_usd_per_million,
        ollama_model_id=local.model_id,
        ollama_base_url=settings.ollama_base_url,
    )
