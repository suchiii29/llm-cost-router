"""Legacy filename; this script benchmarks the two Groq models, not local models."""

import statistics
import time

from model_adapter import ask_cheap_model, ask_strong_model
from model_config import get_model_config


QUESTIONS = [
    "What is 15 multiplied by 4?",
    "If a notebook costs 40 rupees, how much do 5 notebooks cost?",
    "A train travels 60 km in 2 hours. What is its average speed?",
    "A shop has 24 apples and sells 9. How many apples remain?",
    "What is the difference between a list and a tuple in Python?",
]


def main():
    config = get_model_config()
    models = (
        ("Cheap Groq", config.cheap_model_id, ask_cheap_model),
        ("Strong Groq", config.strong_model_id, ask_strong_model),
    )

    for label, model_id, request_model in models:
        print(f"\n{'=' * 60}")
        print(f"{label}: {model_id}")
        print("=" * 60)
        latencies = []

        for question in QUESTIONS:
            started = time.perf_counter()
            response = request_model(question)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)

            print(f"\nQuestion: {question}")
            print(f"Answer: {response.answer.strip()}")
            print(f"Latency: {elapsed:.2f} seconds")
            print(f"Provider input tokens: {response.input_tokens}")
            print(f"Provider output tokens: {response.output_tokens}")

        print(f"\nAverage latency: {statistics.mean(latencies):.2f} seconds")
        print(f"Median latency: {statistics.median(latencies):.2f} seconds")


if __name__ == "__main__":
    main()
