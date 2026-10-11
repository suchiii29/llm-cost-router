import argparse
import json
import time
from datetime import datetime

from checker import is_correct
from model_adapter import (
    GroqAPIError,
    GroqConfigurationError,
    ask_cheap_model,
    ask_strong_model,
)
from model_config import get_model_config
from router import calculate_complexity, route_question


MODES = ("always_cheap", "always_strong", "router")


def select_sample_indices(total, sample_size=5):
    if sample_size <= 0 or sample_size > total:
        raise ValueError("sample_size must be between 1 and the dataset size")
    if sample_size == 1:
        return [0]
    return [
        round(index * (total - 1) / (sample_size - 1))
        for index in range(sample_size)
    ]


def _model_rates(model_id, config):
    if model_id == config.cheap_model_id:
        return (
            config.cheap_input_usd_per_million,
            config.cheap_output_usd_per_million,
        )
    if model_id == config.strong_model_id:
        return (
            config.strong_input_usd_per_million,
            config.strong_output_usd_per_million,
        )
    return None, None


def _attempt_cost(attempt, config):
    input_tokens = attempt.get("input_tokens")
    output_tokens = attempt.get("output_tokens")
    input_rate, output_rate = _model_rates(attempt.get("model_id"), config)
    if (
        attempt.get("outcome") != "success"
        or input_tokens is None
        or output_tokens is None
        or input_rate is None
        or output_rate is None
    ):
        return None
    return (
        input_tokens * input_rate + output_tokens * output_rate
    ) / 1_000_000


def _response_attempt(response):
    return {
        "model_id": response.model_id,
        "outcome": "success",
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "latency_seconds": response.latency_seconds,
        "error_type": None,
        "status_code": None,
        "error": None,
    }


def _failure_attempt(model_id, error):
    return {
        "model_id": model_id,
        "outcome": "failure",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "latency_seconds": None,
        "error_type": type(error).__name__,
        "status_code": getattr(error, "status_code", None),
        "error": str(error),
    }


def run_mode(questions, mode, config=None):
    if mode not in MODES:
        raise ValueError(f"Unknown benchmark mode: {mode}")
    config = config or get_model_config()
    question_results = []

    for item in questions:
        question = item["question"]
        correct_answer = item["correct_answer"]
        complexity, score, reasons = calculate_complexity(question)
        routed_to = "cheap" if complexity == "LOW" else "strong"
        started = time.perf_counter()
        answer = None
        attempts = []
        final_model_id = None
        fallback = False
        error_message = None

        try:
            if mode == "always_cheap":
                response = ask_cheap_model(question)
                attempts.append(_response_attempt(response))
                answer = response.answer
                final_model_id = response.model_id
            elif mode == "always_strong":
                response = ask_strong_model(question)
                attempts.append(_response_attempt(response))
                answer = response.answer
                final_model_id = response.model_id
            else:
                result = route_question(question, display=False)
                attempts = result["api_attempts"]
                answer = result["answer"]
                final_model_id = result["model"]
                fallback = result["fallback"]
                routed_to = result["routed_to"]
                complexity = result["complexity"]
                score = result["score"]
                reasons = result["reasons"]
        except (GroqAPIError, GroqConfigurationError) as error:
            attempts = getattr(error, "api_attempts", attempts)
            if not attempts:
                attempted_model = (
                    config.cheap_model_id
                    if mode == "always_cheap" or routed_to == "cheap"
                    else config.strong_model_id
                )
                attempts = [_failure_attempt(attempted_model, error)]
            error_message = str(error)
            if len(attempts) > 1:
                fallback = True

        latency = time.perf_counter() - started
        for attempt in attempts:
            attempt["estimated_cost_usd"] = _attempt_cost(attempt, config)

        successful_attempts = [
            attempt for attempt in attempts if attempt["outcome"] == "success"
        ]
        usage_complete = bool(successful_attempts) and all(
            attempt["input_tokens"] is not None
            and attempt["output_tokens"] is not None
            for attempt in successful_attempts
        )
        costs_complete = bool(attempts) and all(
            attempt["estimated_cost_usd"] is not None for attempt in attempts
        )
        question_results.append({
            "question_index": item["question_index"],
            "complexity": complexity,
            "complexity_score": score,
            "routing_reasons": reasons,
            "routed_to": routed_to,
            "model_id": final_model_id,
            "success": answer is not None,
            "correct": (
                is_correct(answer, correct_answer)
                if answer is not None
                else None
            ),
            "fallback": fallback,
            "error": error_message,
            "latency_seconds": round(latency, 3),
            "input_tokens": (
                sum(attempt["input_tokens"] for attempt in successful_attempts)
                if usage_complete
                else None
            ),
            "output_tokens": (
                sum(attempt["output_tokens"] for attempt in successful_attempts)
                if usage_complete
                else None
            ),
            "attempts": attempts,
            "estimated_cost_usd": (
                round(sum(attempt["estimated_cost_usd"] for attempt in attempts), 10)
                if costs_complete
                else None
            ),
        })

    successful = [row for row in question_results if row["success"]]
    costs = [row["estimated_cost_usd"] for row in question_results]
    total_cost = (
        round(sum(costs), 10)
        if question_results and all(cost is not None for cost in costs)
        else None
    )
    total_input_tokens = [
        row["input_tokens"] for row in question_results
    ]
    total_output_tokens = [
        row["output_tokens"] for row in question_results
    ]
    return {
        "mode": mode,
        "questions_attempted": len(question_results),
        "successful_answers": len(successful),
        "failed_questions": len(question_results) - len(successful),
        "correct": sum(row["correct"] is True for row in successful),
        "accuracy_percent": (
            round(
                100 * sum(row["correct"] is True for row in successful)
                / len(successful),
                1,
            )
            if successful
            else None
        ),
        "input_tokens": (
            sum(total_input_tokens)
            if all(value is not None for value in total_input_tokens)
            else None
        ),
        "output_tokens": (
            sum(total_output_tokens)
            if all(value is not None for value in total_output_tokens)
            else None
        ),
        "api_attempt_count": sum(len(row["attempts"]) for row in question_results),
        "failed_api_attempts": sum(
            attempt["outcome"] == "failure"
            for row in question_results
            for attempt in row["attempts"]
        ),
        "model_ids": sorted({
            attempt["model_id"]
            for row in question_results
            for attempt in row["attempts"]
        }),
        "latency_seconds": round(
            sum(row["latency_seconds"] for row in question_results),
            3,
        ),
        "estimated_cost_usd": total_cost,
        "questions": question_results,
    }


def _paired_accuracy(result, paired_indices):
    answers = {
        row["question_index"]: row["correct"]
        for row in result["questions"]
        if row["success"]
    }
    if not paired_indices:
        return None
    return round(
        100 * sum(answers[index] is True for index in paired_indices)
        / len(paired_indices),
        1,
    )


def run_benchmark(questions, config=None, sample_size=5):
    config = config or get_model_config()
    indices = select_sample_indices(len(questions), sample_size)
    sample = [
        dict(questions[index], question_index=index)
        for index in indices
    ]
    results = {
        mode: run_mode(sample, mode, config)
        for mode in MODES
    }
    by_mode = list(results.values())
    paired_indices = [
        sample_item["question_index"]
        for sample_item in sample
        if all(
            any(
                row["question_index"] == sample_item["question_index"]
                and row["success"]
                for row in result["questions"]
            )
            for result in by_mode
        )
    ]
    for result in by_mode:
        result["paired_question_count"] = len(paired_indices)
        result["paired_accuracy_percent"] = _paired_accuracy(
            result,
            paired_indices,
        )

    strong_cost = results["always_strong"]["estimated_cost_usd"]
    router_cost = results["router"]["estimated_cost_usd"]
    savings_percent = None
    if (
        strong_cost is not None
        and router_cost is not None
        and strong_cost > 0
    ):
        savings_percent = round(
            100 * (strong_cost - router_cost) / strong_cost,
            2,
        )

    return {
        "dataset_total": len(questions),
        "sample_indices": indices,
        "sample_size": len(sample),
        "paired_question_indices": paired_indices,
        "model_ids": {
            "cheap": config.cheap_model_id,
            "strong": config.strong_model_id,
        },
        "pricing_basis": "configured per-model USD-per-million token rates",
        "cost_savings_percent": savings_percent,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare three Groq routing strategies on a small sample."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full 60-question dataset (up to 180 API attempts plus fallbacks).",
    )
    args = parser.parse_args()

    with open("benchmark_questions.json", encoding="utf-8") as file:
        questions = json.load(file)
    config = get_model_config()
    sample_size = len(questions) if args.full else min(5, len(questions))
    print(
        f"Benchmarking {sample_size} of {len(questions)} questions. "
        f"Expected planned requests: {sample_size * 3}; "
        f"maximum with router fallback: {sample_size * 4}."
    )
    evaluation = run_benchmark(questions, config, sample_size)

    for mode, result in evaluation["results"].items():
        print(
            f"{mode}: {result['successful_answers']}/{result['questions_attempted']} "
            f"answers, {result['correct']} correct, "
            f"accuracy={result['accuracy_percent']}, "
            f"tokens={result['input_tokens']}/{result['output_tokens']}, "
            f"latency={result['latency_seconds']}s, "
            f"cost={result['estimated_cost_usd']}, "
            f"API attempts={result['api_attempt_count']} "
            f"(failed={result['failed_api_attempts']})"
        )
    if evaluation["cost_savings_percent"] is None:
        print("Cost saved (%): unavailable")
    else:
        print(f"Cost saved (%): {evaluation['cost_savings_percent']}")
    print(
        "Paired accuracy uses only questions with successful answers from all "
        f"three strategies: {len(evaluation['paired_question_indices'])}."
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results_path = f"benchmark_results_groq_live_{timestamp}.json"
    evaluation["created_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2)
    print(f"Saved benchmark summary to {results_path}")


if __name__ == "__main__":
    main()
