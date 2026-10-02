import csv
import os
import time
from datetime import datetime

from router import route_question


# ---------------------------------------------------------
# Estimated pricing
# ---------------------------------------------------------
# These are configurable estimates for benchmarking.
# They are NOT actual billing amounts.
#
# Local Ollama:
# Estimated cost = 0 because the model runs locally.
#
# Groq:
# Set an estimated input/output token price here if your
# project wants to compare approximate API costs.
# ---------------------------------------------------------

LOCAL_COST_PER_1K_TOKENS = 0.0

# Keep these as configurable values for now.
# We are using estimated values, not claiming they are
# Groq's current official billing rates.
GROQ_INPUT_COST_PER_1K = 0.0
GROQ_OUTPUT_COST_PER_1K = 0.0


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
    Estimate the cost of one routed request.

    NOTE:
    This is an estimated benchmarking value.
    It is not actual provider billing information.
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

    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": total_cost,
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
                "latency_seconds",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def track_question(question):
    """
    Routes one question and measures its latency and
    estimated cost.
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
        "latency_seconds": round(latency, 3),
    }

    save_log(log_data)

    print("\n" + "=" * 60)
    print("TRACKING INFORMATION")
    print("=" * 60)

    print(
        f"Model             : {log_data['model']}"
    )

    print(
        f"Complexity        : {log_data['complexity']}"
    )

    print(
        f"Estimated input tokens  : "
        f"{log_data['input_tokens']}"
    )

    print(
        f"Estimated output tokens : "
        f"{log_data['output_tokens']}"
    )

    print(
        f"Estimated cost     : "
        f"${log_data['estimated_cost']:.6f}"
    )

    print(
        f"Latency            : "
        f"{log_data['latency_seconds']} seconds"
    )

    print(
        f"Log file           : {LOG_FILE}"
    )

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