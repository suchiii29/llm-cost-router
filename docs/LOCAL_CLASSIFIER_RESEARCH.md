# First local correctness classifier

This is the first offline pre-hoc classifier result for predicting whether the local Qwen answer will pass the dataset checker. The model uses question text and task type only. It does not train on the ground-truth answer, cached answer, correctness rationale, ID, or cache metadata. Its input feature builder explicitly reads only `question` and `task_type`.

## Reproduction

From the repository root, run:

```sh
.venv312/bin/python scripts/train_local_classifier.py
```

This reads `data/train_labels.jsonl`, `data/dev.jsonl`, and `cache/answers_dev.json`, derives dev local and strong outcomes with `checker.py`, fits a TF-IDF plus Logistic Regression pipeline, and writes:

- `models/local_correctness_tfidf_logreg.joblib`
- `results/local_correctness_research.json`

The script performs no model-provider calls and does not read the frozen test dataset. The model artifact contains both TF-IDF preprocessing and Logistic Regression. The JSON report stores input hashes, sample counts, random seed, feature settings, library versions, CV metrics, dev metrics, and policy comparisons.

## Evaluation design

Training uses 5-fold shuffled `StratifiedKFold`, seed 506, and pooled out-of-fold probabilities. The majority baseline always predicts the training majority class. The primary pipeline is `TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)` followed by `LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=506)`. The default selective-routing rule uses local when predicted local correctness is at least 0.5 and strong otherwise. No alternate thresholds were selected.

Dev labels are computed independently from the dev questions, existing local/strong cached raw answers, each row's ground truth, and its checker field. Dev remains outside training and CV. The current dev file contains 60 math examples only, so its results do not establish performance on extract, MCQ, classification, or reformat tasks. Results are exploratory, and random row-wise CV may overstate generalization across repeated task templates.

## Current result

See `results/local_correctness_research.json` for complete metrics and hashes. This is an initial research baseline, not a production routing recommendation. In particular, compare routing accuracy and model call shares together: on this math-only dev set, the classifier threshold routes every example to strong.
