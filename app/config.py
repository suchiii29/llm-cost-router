from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class TierSettings(BaseModel):
    id: int
    name: str
    provider: Literal["ollama", "groq"]
    model_id: str
    input_usd_per_million: Optional[float] = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    output_usd_per_million: Optional[float] = Field(
        default=None, ge=0, allow_inf_nan=False
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
        yaml_file=CONFIG_PATH,
        yaml_file_encoding="utf-8",
    )

    provider_mode: Literal["live", "fake"] = "live"
    db_url: str = "sqlite:///router.db"
    log_prompts: bool = False
    complexity_threshold: int = 3
    request_timeout_seconds: float = 120
    retry_attempts: int = 3
    api_rate_limit_per_minute: int = Field(
        default=60, ge=1, validation_alias="API_RATE_LIMIT_PER_MINUTE"
    )
    router_api_key: Optional[SecretStr] = Field(
        default=None, validation_alias="ROUTER_API_KEY"
    )
    groq_api_key: Optional[SecretStr] = Field(
        default=None, validation_alias="GROQ_API_KEY"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )
    tiers: list[TierSettings] = Field(default_factory=list)

    ollama_model_override: Optional[str] = Field(
        default=None, validation_alias="OLLAMA_MODEL"
    )
    groq_cheap_model_override: Optional[str] = Field(
        default=None, validation_alias="GROQ_CHEAP_MODEL"
    )
    groq_strong_model_override: Optional[str] = Field(
        default=None, validation_alias="GROQ_STRONG_MODEL"
    )
    groq_cheap_input_rate_override: Optional[float] = Field(
        default=None, validation_alias="GROQ_CHEAP_INPUT_USD_PER_MILLION"
    )
    groq_cheap_output_rate_override: Optional[float] = Field(
        default=None, validation_alias="GROQ_CHEAP_OUTPUT_USD_PER_MILLION"
    )
    groq_strong_input_rate_override: Optional[float] = Field(
        default=None, validation_alias="GROQ_STRONG_INPUT_USD_PER_MILLION"
    )
    groq_strong_output_rate_override: Optional[float] = Field(
        default=None, validation_alias="GROQ_STRONG_OUTPUT_USD_PER_MILLION"
    )

    @field_validator(
        "groq_cheap_input_rate_override",
        "groq_cheap_output_rate_override",
        "groq_strong_input_rate_override",
        "groq_strong_output_rate_override",
        mode="before",
    )
    @classmethod
    def blank_rate_is_unset(cls, value):
        return None if value == "" else value

    @field_validator("ollama_base_url")
    @classmethod
    def trim_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            raise ValueError("OLLAMA_BASE_URL must be non-empty")
        return value

    @model_validator(mode="after")
    def apply_environment_overrides(self):
        overrides = {
            0: {
                "model_id": self.ollama_model_override,
            },
            1: {
                "model_id": self.groq_cheap_model_override,
                "input_usd_per_million": self.groq_cheap_input_rate_override,
                "output_usd_per_million": self.groq_cheap_output_rate_override,
            },
            2: {
                "model_id": self.groq_strong_model_override,
                "input_usd_per_million": self.groq_strong_input_rate_override,
                "output_usd_per_million": self.groq_strong_output_rate_override,
            },
        }
        self.tiers = [
            tier.model_copy(update={
                key: value for key, value in overrides.get(tier.id, {}).items()
                if value is not None
            })
            for tier in self.tiers
        ]
        tier_ids = [tier.id for tier in self.tiers]
        if sorted(tier_ids) != [0, 1, 2]:
            raise ValueError("tiers must define exactly IDs 0, 1, and 2")
        if self.complexity_threshold < 0:
            raise ValueError("complexity_threshold must be non-negative")
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be at least one")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    def tier(self, tier_id: int) -> TierSettings:
        for tier in self.tiers:
            if tier.id == tier_id:
                return tier
        raise KeyError(f"Tier {tier_id} is not configured")


def get_settings() -> Settings:
    return Settings()
