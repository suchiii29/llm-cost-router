import json
from datetime import datetime

from checker import is_correct
from model_config import get_model_config
from router import calculate_complexity
from tracker import estimate_cost


def evaluate_cache(questions, cache, config):
    rows = []
    missing_question_indices = []

    for index, item in enumerate(questions):
        cheap_key = f"{index}|{config.cheap_model_id}"
        strong_key = f"{index}|{config.strong_model_id}"
        if cheap_key not in cache or strong_key not in cache:
            missing_question_indices.append(index)
            continue

        cheap_entry = cache[cheap_key]
        strong_entry = cache[strong_key]
        complexity, score, _ = calculate_complexity(item["question"])
        rows.append({
            "index": index,
            "question": item["question"],
            "score": score,
            "rules_says_strong": complexity == "HIGH",
            "cheap_model_id": cheap_entry["model_id"],
            "strong_model_id": strong_entry["model_id"],
            "cheap_ok": is_correct(
                cheap_entry["answer"],
                item["correct_answer"],
            ),
            "strong_ok": is_correct(
                strong_entry["answer"],
                item["correct_answer"],
            ),
            "cheap_cost": estimate_cost(
                cheap_entry["model_id"],
                cheap_entry["prompt"],
                cheap_entry["answer"],
                cheap_entry["provider_usage"],
            )["estimated_cost_usd"],
            "strong_cost": estimate_cost(
                strong_entry["model_id"],
                strong_entry["prompt"],
                strong_entry["answer"],
                strong_entry["provider_usage"],
            )["estimated_cost_usd"],
        })

    policies = {
        "Always cheap": lambda row: False,
        "Always strong": lambda row: True,
        "Rules router": lambda row: row["rules_says_strong"],
        "Oracle (theoretical upper bound)": lambda row: not row["cheap_ok"],
    }
    results = []

    for name, use_strong in policies.items():
        correct = 0
        strong_calls = 0
        total_cost = 0.0
        cost_available = True
        model_ids = set()

        for row in rows:
            choose_strong = use_strong(row)
            correct += row["strong_ok"] if choose_strong else row["cheap_ok"]
            strong_calls += choose_strong
            model_ids.add(
                row["strong_model_id"] if choose_strong else row["cheap_model_id"]
            )
            cost = row["strong_cost"] if choose_strong else row["cheap_cost"]
            if cost is None:
                cost_available = False
            else:
                total_cost += cost

        results.append({
            "policy": name,
            "evaluated": len(rows),
            "correct": correct,
            "accuracy_percent": round(100 * correct / len(rows), 1) if rows else None,
            "strong_calls": strong_calls,
            "model_ids": sorted(model_ids),
            "question_indices": [row["index"] for row in rows],
            "estimated_cost_usd": (
                round(total_cost, 8) if rows and cost_available else None
            ),
        })

    return {
        "dataset_total": len(questions),
        "evaluated": len(rows),
        "missing_question_indices": missing_question_indices,
        "model_ids": {
            "cheap": config.cheap_model_id,
            "strong": config.strong_model_id,
        },
        "rows": rows,
        "results": results,
    }


def main():
    config = get_model_config()
    with open("benchmark_questions.json", encoding="utf-8") as file:
        questions = json.load(file)
    try:
        with open("answers_cache_groq.json", encoding="utf-8") as cache_file:
            cache = json.load(cache_file)
    except FileNotFoundError:
        print(
            f"No Groq model cache exists yet. Evaluated 0 of {len(questions)} "
            "questions; no question has a complete cheap/strong cache pair. "
            "Run `python3 run_cache.py` to collect it."
        )
        return

    evaluation = evaluate_cache(questions, cache, config)
    evaluated = evaluation["evaluated"]
    total = evaluation["dataset_total"]
    print(f"Evaluated {evaluated} of {total} questions\n")
    if evaluation["missing_question_indices"]:
        missing = ", ".join(
            str(index) for index in evaluation["missing_question_indices"]
        )
        print(f"Missing complete cheap/strong cache pairs for question indices: {missing}")
    if not evaluated:
        print(
            "No complete Groq model pairs are cached yet. "
            "Run `python3 run_cache.py` to collect them."
        )
        return

    print(f"{'Policy':<36}{'Accuracy':<12}{'Strong calls':<15}{'Estimated cost'}")
    for result in evaluation["results"]:
        accuracy = result["accuracy_percent"]
        print(
            f"{result['policy']:<36}{accuracy:>6.1f}%"
            f"{'':<5}{result['strong_calls']}/{evaluated:<10}"
            f"{result['estimated_cost_usd'] if result['estimated_cost_usd'] is not None else 'unavailable'}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    labels_path = f"labels_groq_{timestamp}.json"
    results_path = f"benchmark_results_groq_{timestamp}.json"
    with open(labels_path, "w", encoding="utf-8") as file:
        json.dump(evaluation["rows"], file, indent=2)
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump({
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset_total": total,
            "evaluated": evaluated,
            "missing_question_indices": evaluation["missing_question_indices"],
            "model_ids": evaluation["model_ids"],
            "results": evaluation["results"],
        }, file, indent=2)
    print(f"\nSaved {labels_path} and {results_path}")


if __name__ == "__main__":
    main()
