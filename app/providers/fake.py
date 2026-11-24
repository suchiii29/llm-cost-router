import time

from app.providers.base import ProviderResponse, make_response


class FakeProvider:
    """Offline provider with deterministic canned responses."""

    ANSWER = "This is a deterministic fake provider response."

    def generate(self, question: str, model_id: str) -> ProviderResponse:
        started_at = time.perf_counter()
        return make_response(
            answer=self.ANSWER,
            question=question,
            model_id=model_id,
            provider="fake",
            started_at=started_at,
            input_tokens=7,
            output_tokens=9,
        )
