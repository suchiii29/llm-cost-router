"""Legacy filename; smoke-tests both Groq models and never calls Ollama."""

from model_adapter import ask_cheap_model, ask_strong_model
from model_config import get_model_config


def main():
    config = get_model_config()
    question = "What is 2+2? Answer in one word."

    for label, model_id, request_model in (
        ("Cheap Groq", config.cheap_model_id, ask_cheap_model),
        ("Strong Groq", config.strong_model_id, ask_strong_model),
    ):
        response = request_model(question)
        print(f"{label} ({model_id}) said:", response.answer)
        print(
            "Provider tokens:",
            response.input_tokens,
            "input,",
            response.output_tokens,
            "output",
        )


if __name__ == "__main__":
    main()
