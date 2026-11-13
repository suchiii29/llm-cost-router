import time

from app.providers.base import (
    BaseProvider,
    ProviderConfigurationError,
    ProviderResponse,
    ProviderResponseError,
    make_response,
    record_metrics,
)


API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs) -> None:
        super().__init__(**kwargs)
        if not api_key:
            raise ProviderConfigurationError("GROQ_API_KEY is not configured")
        self.api_key = api_key

    def generate(self, question: str, model_id: str) -> ProviderResponse:
        started_at = time.perf_counter()
        data = self.post_json(
            API_URL,
            {
                "model": model_id,
                "messages": [{"role": "user", "content": question}],
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderResponseError(
                "Groq response did not contain a chat completion"
            ) from error
        if not isinstance(answer, str):
            raise ProviderResponseError("Groq response content was not text")
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            raise ProviderResponseError("Groq returned invalid usage metadata")
        result = make_response(
            answer=answer,
            question=question,
            model_id=data.get("model", model_id),
            provider="groq",
            started_at=started_at,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
        record_metrics(result)
        return result
