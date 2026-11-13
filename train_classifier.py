import re

from router import calculate_complexity


def extract_features(question):
    q = question.lower()
    words = re.findall(r"\b\w+\b", q)
    numbers = re.findall(r"\d+(?:\.\d+)?", q)

    complexity, score, reasons = calculate_complexity(question)

    reasoning_words = [
        "explain", "analyze", "analyse", "compare",
        "calculate", "solve", "derive", "prove",
        "evaluate", "optimize", "difference"
    ]

    math_words = [
        "total", "altogether", "remaining", "left",
        "twice", "half", "each", "every", "per",
        "more", "less", "before", "after"
    ]

    return {
        "question": question,
        "word_count": len(words),
        "character_count": len(question),
        "number_count": len(numbers),
        "unique_number_count": len(set(numbers)),
        "reasoning_word_count": sum(
            word in words for word in reasoning_words
        ),
        "math_word_count": sum(
            word in words for word in math_words
        ),
        "and_count": words.count("and"),
        "then_count": words.count("then"),
        "question_mark_count": question.count("?"),
        "complexity_score": score,
        "is_high_complexity": int(complexity == "HIGH"),
        "reason_count": len(reasons),
    }


def main():
    import json
    import pickle

    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    with open("train_labels.json") as f:
        data = json.load(f)

    records = list(data.values())

    X = pd.DataFrame([
        extract_features(item["question"])
        for item in records
    ])

    y = np.array([
        0 if item["local_ok"] else 1
        for item in records
    ])

    numeric_features = [
        "word_count",
        "character_count",
        "number_count",
        "unique_number_count",
        "reasoning_word_count",
        "math_word_count",
        "and_count",
        "then_count",
        "question_mark_count",
        "complexity_score",
        "is_high_complexity",
        "reason_count",
    ]

    preprocessor = ColumnTransformer([
        (
            "text",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=3000,
                sublinear_tf=True
            ),
            "question"
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ])

    model = Pipeline([
        ("features", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        )
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )

    print("Training examples:", len(X))
    print("Local answers correct:", sum(y == 0))
    print("Local answers incorrect:", sum(y == 1))

    print("\nCross-validation classification report:")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Local correct", "Local incorrect"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y, predictions, labels=[0, 1]))

    model.fit(X, y)

    with open("classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\nSaved improved classifier to classifier.pkl")


if __name__ == "__main__":
    main()
