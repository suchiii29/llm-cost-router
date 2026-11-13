"""Train and evaluate a question-only local-answer correctness classifier offline."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRAIN_LABELS_PATH = PROJECT_ROOT / "data" / "train_labels.jsonl"
DEV_PATH = PROJECT_ROOT / "data" / "dev.jsonl"
DEV_CACHE_PATH = PROJECT_ROOT / "cache" / "answers_dev.json"
MODEL_PATH = PROJECT_ROOT / "models" / "local_correctness_tfidf_logreg.joblib"
RESULTS_PATH = PROJECT_ROOT / "results" / "local_correctness_research.json"
RANDOM_SEED = 506
CV_FOLDS = 5

# Explicitly whitelist the two permitted feature fields. Labels and cache metadata
# are never passed through this function to the vectorizer.
def feature_text(record: dict[str, Any]) -> str:
    question = record.get("question")
    task_type = record.get("task_type")
    if not isinstance(question, str) or not isinstance(task_type, str):
        raise ValueError("Each feature row requires text question and task_type fields")
    return f"task_type {task_type} question {question}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED,
            solver="liblinear",
        )),
    ])


def _metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray,
             probabilities: list[float] | np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2 else None
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def _group_metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_type"]].append(row)
    output = {}
    for task_type, task_rows in sorted(grouped.items()):
        y = [row["local_ok"] for row in task_rows]
        pred = [row[prediction_key] for row in task_rows]
        probabilities = [row.get("local_probability", 0.5) for row in task_rows]
        output[task_type] = _metrics(y, pred, probabilities)
    return output


def _derive_dev_rows(dev_items: list[dict[str, Any]], cache: dict[str, Any]) -> list[dict[str, Any]]:
    if cache.get("split") != "dev" or not isinstance(cache.get("entries"), dict):
        raise ValueError("Dev cache must be the split='dev' cache format")
    entries = cache["entries"]
    rows = []
    seen = set()
    for item in dev_items:
        item_id = item.get("id")
        if item_id in seen:
            raise ValueError(f"Duplicate dev ID: {item_id}")
        seen.add(item_id)
        if item.get("split") != "dev":
            raise ValueError(f"Unexpected split in dev dataset for {item_id}")
        entry = entries.get(item_id)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing dev cache entry for {item_id}")
        answers = entry.get("answers", {})
        local = answers.get("local", {})
        strong = answers.get("strong", {})
        if not isinstance(local.get("answer"), str) or not isinstance(strong.get("answer"), str):
            raise ValueError(f"Missing local or strong answer for dev ID {item_id}")
        from checker import check_answer
        rows.append({
            "id": item_id,
            "question": item["question"],
            "task_type": item["task_type"],
            "local_ok": int(check_answer(local["answer"], item["answer"], item["checker"])),
            "strong_ok": int(check_answer(strong["answer"], item["answer"], item["checker"])),
        })
    extra = set(entries) - seen
    if extra:
        raise ValueError(f"Dev cache contains IDs outside dev dataset: {sorted(extra)}")
    return rows


def _routing_results(rows: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    policy_rows = []
    for row, probability in zip(rows, probabilities):
        policy_rows.append({**row, "local_probability": float(probability)})
    result = {}
    for policy in ("always_local", "always_strong", "classifier_selective"):
        strong_calls = 0
        correct = 0
        grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in policy_rows:
            choose_strong = policy == "always_strong" or (
                policy == "classifier_selective" and row["local_probability"] < 0.5
            )
            strong_calls += int(choose_strong)
            outcome = row["strong_ok"] if choose_strong else row["local_ok"]
            correct += outcome
            grouped[row["task_type"]].append((outcome, int(choose_strong)))
        n = len(policy_rows)
        result[policy] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
            "local_calls": n - strong_calls,
            "local_call_share": (n - strong_calls) / n if n else None,
            "strong_calls": strong_calls,
            "strong_call_share": strong_calls / n if n else None,
            "by_task_type": {
                task: {
                    "n": len(values),
                    "accuracy": sum(value[0] for value in values) / len(values),
                    "local_calls": sum(1 - value[1] for value in values),
                    "strong_calls": sum(value[1] for value in values),
                }
                for task, values in sorted(grouped.items())
            },
        }
    return {"threshold": 0.5, "route_to_strong_when_local_probability_below": 0.5,
            "policies": result}


def run_training(train_path: Path = TRAIN_LABELS_PATH, dev_path: Path = DEV_PATH,
                 dev_cache_path: Path = DEV_CACHE_PATH, model_path: Path = MODEL_PATH,
                 results_path: Path = RESULTS_PATH) -> dict[str, Any]:
    train = _read_jsonl(train_path)
    if len(train) != 598:
        raise ValueError(f"Expected 598 training labels, found {len(train)}")
    ids = [item.get("id") for item in train]
    if len(set(ids)) != len(ids):
        raise ValueError("Training labels contain duplicate IDs")
    if any(item.get("local_ok") not in (0, 1) for item in train):
        raise ValueError("Training labels must contain binary local_ok values")
    X = [feature_text(item) for item in train]
    y = np.asarray([int(item["local_ok"]) for item in train], dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Training labels require both classes")

    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_probabilities = cross_val_predict(_model(), X, y, cv=folds, method="predict_proba")[:, 1]
    cv_predictions = (cv_probabilities >= 0.5).astype(int)
    cv_metrics = _metrics(y, cv_predictions, cv_probabilities)
    majority_class = int(Counter(y.tolist()).most_common(1)[0][0])
    baseline_predictions = np.full(len(y), majority_class, dtype=int)
    baseline = _metrics(y, baseline_predictions, baseline_predictions.astype(float))

    model = _model().fit(X, y)
    dev_items = _read_jsonl(dev_path)
    cache = json.loads(dev_cache_path.read_text(encoding="utf-8"))
    dev_rows = _derive_dev_rows(dev_items, cache)
    dev_X = [feature_text(row) for row in dev_rows]
    dev_y = np.asarray([row["local_ok"] for row in dev_rows], dtype=int)
    dev_probs = model.predict_proba(dev_X)[:, 1]
    dev_predictions = (dev_probs >= 0.5).astype(int)
    dev_eval_rows = [{**row, "local_probability": float(prob),
                      "classifier_prediction": int(pred)}
                     for row, prob, pred in zip(dev_rows, dev_probs, dev_predictions)]
    dev_metrics = _metrics(dev_y, dev_predictions, dev_probs)
    dev_metrics["by_task_type"] = _group_metrics(dev_eval_rows, "classifier_prediction")
    routing = _routing_results(dev_rows, dev_probs)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    report = {
        "purpose": "question-only prediction of local Qwen checker correctness",
        "random_seed": RANDOM_SEED,
        "cross_validation": {"strategy": "StratifiedKFold", "folds": CV_FOLDS,
                              "shuffle": True, "metrics": cv_metrics,
                              "majority_class_baseline": {
                                  "class": majority_class, **baseline}},
        "dev": {"metrics": dev_metrics, "routing_comparison": routing},
        "features": {"fields": ["question", "task_type"],
                      "vectorizer": "TfidfVectorizer", "ngram_range": [1, 2],
                      "min_df": 1, "sublinear_tf": True,
                      "classifier": "LogisticRegression", "class_weight": "balanced",
                      "solver": "liblinear", "max_iter": 2000,
                      "probability_threshold": 0.5},
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "joblib": joblib.__version__},
        "data": {"train_labels": {"path": _display_path(train_path),
                                    "count": len(train), "sha256": _sha256(train_path)},
                 "dev": {"path": _display_path(dev_path),
                         "count": len(dev_items), "sha256": _sha256(dev_path)},
                 "dev_cache": {"path": _display_path(dev_cache_path),
                               "sha256": _sha256(dev_cache_path)}},
        "artifacts": {"model": _display_path(model_path)},
    }
    results_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = run_training()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
