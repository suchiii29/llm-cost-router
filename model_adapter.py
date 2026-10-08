import os
import json
import time
import requests

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
METRICS_FILE = "model_metrics.jsonl"


def record_metrics(metrics):
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def ask_local_model(question):
    """Send a question to Ollama and record performance metrics."""

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": question,
        "stream": False
    }

    start = time.perf_counter()

    response = requests.post(
        url,
        json=payload,
        timeout=300
    )

    elapsed = time.perf_counter() - start
    response.raise_for_status()

    data = response.json()
    answer = data["response"]

    record_metrics({
        "model": "qwen2.5:1.5b",
        "provider": "ollama",
        "latency_seconds": round(elapsed, 3),
        "input_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
        "estimated_api_cost_usd": 0.0
    })

    return answer


def ask_strong_model(question):
    """Send a question to Groq and record performance metrics."""

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing from your .env file")

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "user", "content": question}
        ]
    }

    start = time.perf_counter()

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    elapsed = time.perf_counter() - start
    response.raise_for_status()

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    usage = data.get("usage", {})

    record_metrics({
        "model": "openai/gpt-oss-120b",
        "provider": "groq",
        "latency_seconds": round(elapsed, 3),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "estimated_api_cost_usd": None
    })

    return answer


if __name__ == "__main__":
    question = input("Ask me anything: ")

    print("\n--- Local Ollama answer ---")
    print(ask_local_model(question))

    print("\n--- Strong Groq API answer ---")
    print(ask_strong_model(question))