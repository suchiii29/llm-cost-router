from app.providers.fake import FakeProvider
from app.providers.groq import GroqProvider
from app.providers.ollama import OllamaProvider


def get_provider(settings, tier):
    kwargs = {
        "timeout": settings.request_timeout_seconds,
        "retry_attempts": settings.retry_attempts,
    }
    if settings.provider_mode == "fake":
        return FakeProvider()
    if tier.provider == "ollama":
        return OllamaProvider(settings.ollama_base_url, **kwargs)
    if tier.provider == "groq":
        api_key = (
            settings.groq_api_key.get_secret_value()
            if settings.groq_api_key is not None
            else ""
        )
        return GroqProvider(api_key, **kwargs)
    raise ValueError(f"Unsupported provider: {tier.provider}")
