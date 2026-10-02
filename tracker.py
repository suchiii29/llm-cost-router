import csv
import os
import time
from datetime import datetime

from router import route_question


# ---------------------------------------------------------
# Pricing
# ---------------------------------------------------------
# Local Ollama:
# Cost = 0 because the model runs locally — no per-token charge.
#
# Groq (openai/gpt-oss-120b):
# FIX: these were placeholders set to 0.0, which made every
# estimated_cost come out as $0.00, even for Groq-routed requests.
# Real published rate (console.groq.com/docs/pricing, checked Oct 2026):
#   $0.15 per 1M input tokens  -> $0.00015 per 1K tokens
#   $0.75 per 1M output tokens -> $0.00075 per 1K tokens
# ---------------------------------------------------------

LOCAL_COST_PER_1K_TOKENS = 0.0

GROQ_INPUT_COST_PER_1K = 0.00015
GROQ_OUTPUT_COST_PER_1K = 0.00075


LOG_FILE = "routing_log.csv"


def estimate_tokens(text):
    """
    Rough token estimate.

    This is only an approximation.
    A simple rule of thumb is around 4 characters per token
    for English text.
    """

    if not text:
        return 0

    return max(1, len(text) // 4)


def estimate_cost(model, question, answer):
    """
    Estimate the cost of one routed request, AND what it would have
    cost if the strong (Groq) model had answered it instead. The
    second number is the baseline your project compares against to
    show a measurable cost reduction.
    """

    input_tokens = estimate_tokens(question)
    output_tokens = estimate_tokens(answer)

    if model == "local":
        input_cost = (
            input_tokens / 1000
        ) * LOCAL_COST_PER_1K_TOKENS
        output_cost = 0.0
    else:
        input_cost = (
            input_tokens / 1000
        ) * GROQ_INPUT_COST_PER_1K
        output_cost = (
            output_tokens / 1000
        ) * GROQ_OUTPUT_COST_PER_1K

    actual_cost = input_cost + output_cost

    # Baseline: what this SAME request would have cost if it had
    # been sent to Groq regardless of which model actually answered.
    baseline_cost = (
        (input_tokens / 1000) * GROQ_INPUT_COST_PER_1K
        + (output_tokens / 1000) * GROQ_OUTPUT_COST_PER_1K
    )

    saved = baseline_cost - actual_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": actual_cost,
        "baseline_cost": baseline_cost,
        "saved": saved,
    }


def save_log(result):
    """
    Saves one routing result to routing_log.csv.
    """

    file_exists = os.path.exists(LOG_FILE)

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "question",
                "complexity",
                "score",
                "model",
                "input_tokens",
                "output_tokens",
                "estimated_cost",
                "baseline_cost",
                "saved",
                "latency_seconds",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def track_question(question):
    """
    Routes one question and measures its latency, estimated cost,
    and how much that cost saved compared to always using Groq.
    """

    start_time = time.perf_counter()

    routing_result = route_question(question)

    end_time = time.perf_counter()

    latency = end_time - start_time

    cost_data = estimate_cost(
        routing_result["model"],
        routing_result["question"],
        routing_result["answer"],
    )

    log_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": routing_result["question"],
        "complexity": routing_result["complexity"],
        "score": routing_result["score"],
        "model": routing_result["model"],
        "input_tokens": cost_data["input_tokens"],
        "output_tokens": cost_data["output_tokens"],
        "estimated_cost": round(
            cost_data["estimated_cost"],
            6
        ),
        "baseline_cost": round(
            cost_data["baseline_cost"],
            6
        ),
        "saved": round(
            cost_data["saved"],
            6
        ),
        "latency_seconds": round(latency, 3),
    }

    save_log(log_data)

    print("\n" + "=" * 60)
    print("TRACKING INFORMATION")
    print("=" * 60)

    print(f"Model             : {log_data['model']}")
    print(f"Complexity        : {log_data['complexity']}")
    print(f"Estimated input tokens  : {log_data['input_tokens']}")
    print(f"Estimated output tokens : {log_data['output_tokens']}")
    print(f"Estimated cost     : ${log_data['estimated_cost']:.6f}")
    print(f"Baseline cost (Groq) : ${log_data['baseline_cost']:.6f}")
    print(f"Saved              : ${log_data['saved']:.6f}")
    print(f"Latency            : {log_data['latency_seconds']} seconds")
    print(f"Log file           : {LOG_FILE}")

    print("=" * 60)


def main():

    print("=" * 60)
    print("LLM COST TRACKER")
    print("=" * 60)
    print("The router will select the model automatically.")
    print("Tracking will record model, latency and estimated cost.")
    print("Type 'exit' to stop.")
    print("=" * 60)

    while True:

        question = input("\nAsk a question: ").strip()

        if question.lower() == "exit":
            print("\nExiting tracker.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            track_question(question)

        except Exception as error:
            print("\nERROR:")
            print(error)


if __name__ == "__main__":
    main()