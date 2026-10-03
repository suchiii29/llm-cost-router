import json
import re
import time
from model_adapter import ask_local_model, ask_strong_model
from router import route_question
from tracker import estimate_cost


def extract_number(text):
    """
    Pulls the last number out of a model's answer text, so we can
    compare it to the known correct answer. Models often explain
    their reasoning before giving the final number, so we grab the
    LAST number mentioned, which is usually the final answer.
    """
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if not numbers:
        return None
    return numbers[-1]


def is_correct(model_answer_text, correct_answer):
    """
    Checks if the model's extracted number matches the known
    correct answer. Converts both to float so '18' and '18.0'
    both count as correct.
    """
    extracted = extract_number(model_answer_text)
    if extracted is None:
        return False
    try:
        return float(extracted) == float(correct_answer)
    except ValueError:
        return False


def run_mode(questions, mode):
    """
    Runs every question through ONE specific mode:
    'always_local', 'always_strong', or 'router'.
    Returns accuracy and total cost for that mode.
    """
    correct_count = 0
    total_cost = 0

    for item in questions:
        question = item["question"]
        correct_answer = item["correct_answer"]

        if mode == "always_local":
            answer = ask_local_model(question)
            model_used = "local"
        elif mode == "always_strong":
            answer = ask_strong_model(question)
            model_used = "groq"
        elif mode == "router":
            result = route_question(question)
            answer = result["answer"]
            model_used = result["model"]

        cost_info = estimate_cost(model_used, question, answer)
        total_cost += cost_info["estimated_cost"]

        if is_correct(answer, correct_answer):
            correct_count += 1

    accuracy = (correct_count / len(questions)) * 100
    return {
        "mode": mode,
        "accuracy": round(accuracy, 1),
        "correct": correct_count,
        "total": len(questions),
        "total_cost": round(total_cost, 6)
    }


if __name__ == "__main__":
    with open("benchmark_questions.json") as f:
        questions = json.load(f)

    print(f"Running benchmark on {len(questions)} questions...\n")

    print("=" * 60)
    print("MODE 1: Always Local (cheap model only)")
    print("=" * 60)
    result_local = run_mode(questions, "always_local")
    print(result_local)

    print("\n" + "=" * 60)
    print("MODE 2: Always Strong (Groq only)")
    print("=" * 60)
    result_strong = run_mode(questions, "always_strong")
    print(result_strong)

    print("\n" + "=" * 60)
    print("MODE 3: Router (our system)")
    print("=" * 60)
    result_router = run_mode(questions, "router")
    print(result_router)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Mode':<15} {'Accuracy':<12} {'Cost':<12}")
    print(f"{'Always Local':<15} {result_local['accuracy']}%{'':<7} ${result_local['total_cost']}")
    print(f"{'Always Strong':<15} {result_strong['accuracy']}%{'':<7} ${result_strong['total_cost']}")
    print(f"{'Router':<15} {result_router['accuracy']}%{'':<7} ${result_router['total_cost']}")

    with open("benchmark_results.json", "w") as f:
        json.dump({
            "always_local": result_local,
            "always_strong": result_strong,
            "router": result_router
        }, f, indent=2)

    print("\nResults saved to benchmark_results.json")