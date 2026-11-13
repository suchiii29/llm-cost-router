def main():
    import json
    import pickle

    import pandas as pd

    from checker import is_correct
    from router import calculate_complexity
    from train_classifier import extract_features

    with open("classifier.pkl", "rb") as f:
        model = pickle.load(f)

    with open("benchmark_questions.json") as f:
        questions = json.load(f)

    with open("answers_cache.json") as f:
        cache = json.load(f)

    rows = []

    for i, item in enumerate(questions):
        local_key = f"{i}|local"
        strong_key = f"{i}|strong"

        if local_key not in cache or strong_key not in cache:
            continue

        question = item["question"]

        features = pd.DataFrame([
            extract_features(question)
        ])

        prediction = model.predict(features)[0]

        local_ok = is_correct(
            cache[local_key]["answer"],
            item["correct_answer"]
        )

        strong_ok = is_correct(
            cache[strong_key]["answer"],
            item["correct_answer"]
        )

        rows.append({
            "question": question,
            "predict_incorrect": prediction == 1,
            "local_ok": local_ok,
            "strong_ok": strong_ok,
        })

    policies = {
        "Legacy Qwen only": lambda r: False,
        "Legacy strong only": lambda r: True,
        "Legacy rules router": lambda r: (
            calculate_complexity(r["question"])[0] == "HIGH"
        ),
        "Legacy classifier": lambda r: r["predict_incorrect"],
    }

    n = len(rows)

    print(f"Evaluated {n} questions\n")
    print(f"{'Policy':<22}{'Accuracy':<14}{'Strong calls'}")

    for name, use_strong in policies.items():
        correct = 0
        strong_calls = 0

        for row in rows:
            use = use_strong(row)

            if use:
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
