import time

from app.providers.base import (
    BaseProvider,
    ProviderResponse,
    ProviderResponseError,
    make_response,
    record_metrics,
)


class OllamaProvider(BaseProvider):
    def __init__(self, base_url: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")

    def generate(self, question: str, model_id: str) -> ProviderResponse:
        started_at = time.perf_counter()
        data = self.post_json(
            f"{self.base_url}/api/generate",
            {"model": model_id, "prompt": question, "stream": False},
        )
        answer = data.get("response")
        if not isinstance(answer, str):
            raise ProviderResponseError("Ollama response did not contain generated text")
        result = make_response(
            answer=answer,
            question=question,
            model_id=data.get("model", model_id),
            provider="ollama",
            started_at=started_at,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )
        record_metrics(result)
        return result
