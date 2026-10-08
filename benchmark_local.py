import time
import statistics
import requests

MODELS = [
    "qwen2.5:1.5b",
    "qwen2.5:3b",
]

QUESTIONS = [
    "What is 15 multiplied by 4?",
    "If a notebook costs 40 rupees, how much do 5 notebooks cost?",
    "A train travels 60 km in 2 hours. What is its average speed?",
    "A shop has 24 apples and sells 9. How many apples remain?",
    "What is the difference between a list and a tuple in Python?",
]


def ask_ollama(model, question):
    start = time.perf_counter()

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": question,
            "stream": False,
        },
        timeout=300,
    )

    response.raise_for_status()

    elapsed = time.perf_counter() - start
    data = response.json()

    return data["response"], elapsed, data


for model in MODELS:
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model}")
    print("=" * 60)

    times = []

    for question in QUESTIONS:
        try:
            answer, elapsed, data = ask_ollama(model, question)
            times.append(elapsed)

            print(f"\nQuestion: {question}")
            print(f"Answer: {answer.strip()}")
            print(f"Latency: {elapsed:.2f} seconds")
            print(f"Input tokens: {data.get('prompt_eval_count')}")
            print(f"Output tokens: {data.get('eval_count')}")

        except requests.RequestException as e:
            print(f"\nRequest failed: {e}")

    if times:
        print(f"\nAverage latency: {statistics.mean(times):.2f} seconds")
        print(f"Median latency: {statistics.median(times):.2f} seconds")
