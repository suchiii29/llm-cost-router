def main():
    import json
    import pickle

    import pandas as pd

    from checker import is_correct
    from router import calculate_complexity
    from train_classifier import extract_features

    with open("benchmark_questions.json") as f:
        questions = json.load(f)

    with open("answers_cache.json") as f:
        cache = json.load(f)

    with open("classifier.pkl", "rb") as f:
        model = pickle.load(f)

    rows = []

    for i, item in enumerate(questions):
        local_key = f"{i}|local"
        strong_key = f"{i}|strong"

        if local_key not in cache or strong_key not in cache:
            continue

        question = item["question"]

        rules_hard = (
            calculate_complexity(question)[0] == "HIGH"
        )

        features = pd.DataFrame([
            extract_features(question)
        ])

        classifier_hard = model.predict(features)[0] == 1

        local_ok = is_correct(
            cache[local_key]["answer"],
            item["correct_answer"]
        )

        strong_ok = is_correct(
            cache[strong_key]["answer"],
            item["correct_answer"]
        )

        rows.append({
            "local_ok": local_ok,
            "strong_ok": strong_ok,
            "rules": rules_hard,
            "classifier": classifier_hard,
        })

    policies = {
        "Legacy Qwen only": lambda r: False,
        "Legacy rules router": lambda r: r["rules"],
        "Legacy classifier": lambda r: r["classifier"],
        "Legacy hybrid OR": lambda r: (
            r["rules"] or r["classifier"]
        ),
        "Legacy hybrid AND": lambda r: (
            r["rules"] and r["classifier"]
        ),
        "Legacy strong only": lambda r: True,
    }

    n = len(rows)

    print(f"\nEvaluated {n} questions\n")
    print(
        f"{'Policy':<22}"
        f"{'Accuracy':<13}"
        f"{'Strong calls':<15}"
    )

    for name, route in policies.items():
        correct = 0
        strong_calls = 0

        for row in rows:
            use_strong = route(row)

            if use_strong:
                strong_calls += 1
                correct += row["strong_ok"]
            else:
                correct += row["local_ok"]

        accuracy = correct / n * 100 if n else 0

        print(
            f"{name:<22}"
            f"{accuracy:>6.1f}%"
            f"{'':<7}"
            f"{strong_calls}/{n}"
        )


if __name__ == "__main__":
    main()
