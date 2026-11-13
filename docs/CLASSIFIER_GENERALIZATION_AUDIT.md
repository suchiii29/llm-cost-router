# Classifier Generalization Audit

## Scope and data integrity

This read-only audit compared train and dev question/label/cache records with `checker.py`, reran the existing five-fold prediction procedure, and ran grouped diagnostics. It did not read the frozen test set, write datasets/caches/labels/model artifacts, or make provider/network calls.

Both train math and dev math use `task_type=math` and `checker=numeric`, and both are scored by the same `check_answer()` -> `is_correct()` -> `extract_number()` implementation. The numeric checker removes commas from the extracted answer and ground truth, then compares numeric values with a `1e-6` tolerance. All 200 train math rows and 60 dev rows have this checker. Rechecking the train cache's raw local answer against each source row produced **zero disagreements** with the saved train labels. Dev labels were derived directly from its local cache answers using that same checker: **39/60 correct (65%)**. There is no observed checker-semantic or train-label-generation mismatch explaining the math gap.

The checker does have a general limitation: it takes the last numeric value in an answer when there is no explicit “final answer” marker. That can be ambiguous for verbose reasoning. It does not explain the 7.5% train result: all 200 train math answers yielded a number, and each of the 185 checker-negative cases yielded a value different from its ground truth. The two comma-formatted train ground truths are handled by the corrected checker; their model outputs are also numerically wrong.

## Dataset and cache shift

| Property | Train math | Dev math |
|---|---:|---:|
| Rows / checker | 200 / numeric | 60 / numeric |
| Source indicated by IDs/build path | GSM8K train sample | `benchmark_questions.json` wrapped as `dev-existing-*` |
| Local correctness | 15/200 (7.5%) | 39/60 (65.0%) |
| Mean question words (median) | 45.1 (43) | 44.6 (41) |
| Ground-truth answer characters, median | 2 | 2 |
| Unique ground-truth strings | 110 | 47 |
| Ground truths with commas / decimals / negative sign | 2 / 0 / 0 | 0 / 0 / 0 |

Question length and short integer answer formats are similar, but source and apparent problem mix are not. GSM8K train questions more often contain rough multi-step indicators: 63 include `$` and 21 include `%`, versus 21 and 11 in dev; 55 train questions contain a fraction/multiplier term such as “half”, “third”, “twice”, or “double”, versus 11 in dev. These are descriptive proxies, not a formal difficulty rating. The distinct sources and observed accuracy gap support a real source/difficulty distribution shift; they do not prove difficulty alone caused the gap.

The raw-value differences are visible in individual saved records: train `gsm8k-train-05899` expects `22` but local output is `14`; train `gsm8k-train-02662` expects `72` but output is `48`. A dev negative such as `dev-existing-002` expects `70000` and its imported local reasoning ends at `65000`. These are numeric disagreements under the same checker, not output-format-only failures.


The model-output regimes also differ substantially:

- Every train local cache entry identifies `ollama` / `qwen2.5:1.5b`; its request prompt is saved. Math answers are short: 181/200 are plain numeric strings, 19 are prose/worked responses, and median answer length is 2 characters. Only 15 values match the ground truth.
- Dev local entries are marked `imported: true`, but do not include a model ID, provider, or original prompt. The source/model/prompt therefore cannot be verified as the same Qwen invocation. Their answers are all multiline, have median length 656 characters, and 53/60 contain a “final answer” marker. The checker accepts 39 and rejects 21.
- All 60 cached dev strong answers pass the numeric checker. That makes the observed always-strong result 60/60 on this cache, not a general guarantee.

Thus the 7.5% train result is a genuine low checker-scored success rate for the saved GSM8K/Qwen train outputs, rather than a broad parsing defect. The train/dev discrepancy is consistent with source/problem-mix shift and may also reflect a different generation/model/prompt regime in the imported dev cache. Missing dev provenance prevents separating those causes. Dataset construction is a known confound: training math is exclusively GSM8K, while dev math is exclusively the older benchmark question set. There are no exact normalized train/dev question overlaps.

## Why random CV looked strong

The classifier is not performing well on math in random CV. Its aggregate 83.44% accuracy is dominated by task composition and class prevalence. Out-of-fold predictions at 0.5 threshold show it predicts every math question as incorrect: **185/200 accuracy (92.5%) but zero recall for the 15 correct answers**. That looks good in accuracy only because 92.5% of train math labels are negative. It similarly predicts almost every MCQ positive and most reformat rows negative. Since `task_type` is an allowed feature and task-level label rates differ greatly, aggregate CV mostly rewards task recognition and majority-class behavior; it does not demonstrate robust within-math prediction.

Synthetic simple tasks reuse instruction templates. A deterministic template audit found 198 rows among repeated instruction-prefix groups. In each random CV fold, 13–16 instruction templates appeared in both training and validation portions. These are related generated patterns, though not exact duplicate questions. By contrast, character 3–5-gram similarity at cosine threshold 0.78 found no multirow component among the 200 math questions. This audit found no repeated near-duplicate math wording under that explicit heuristic; it cannot rule out semantic similarity.

A more realistic sensitivity run used `StratifiedGroupKFold(5, shuffle=True, random_state=506)`, grouping synthetic simple rows by task plus instruction prefix and grouping math/MCQ question near-duplicates by connected components of within-task character TF-IDF cosine similarity >= 0.78. It produced **81.44% accuracy, 76.43% precision, 84.70% recall, 80.35% F1, ROC-AUC 0.861**. This only modestly reduces aggregate accuracy from random CV's 83.44%; the math subgroup still predicts every row negative (92.5% accuracy, 0 recall, ROC-AUC 0.559). Grouping removes much of the repeated-template exposure but cannot fix the task/base-rate problem.

A source-held-out Leave-One-Group-Out diagnostic, grouping IDs by `gsm8k`, `arc`, and `simple`, yielded only **26.76% accuracy** overall (precision 19.42%, recall 20.15%, F1 19.78%, ROC-AUC 0.195). It is intentionally severe and confounded because each source maps almost exactly to one task family (GSM8K=math, ARC=MCQ, simple=other tasks); it is not a clean estimate of within-source generalization. It does show that the model is not transportable across these source/task regimes.

## Metrics

Confusion matrices below use true labels as rows and predictions as columns, ordered `[0, 1]`.

### Existing random five-fold CV, pooled out-of-fold predictions

Overall: **accuracy 83.44%, precision 78.84%, recall 86.19%, F1 82.35%, ROC-AUC 0.895**; confusion matrix `[[268, 62], [37, 231]]`. Majority-class baseline is **55.18% accuracy** (always 0).

| Task | n | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| Math | 200 | 92.50% | 0.00% | 0.00% | 0.00% | 0.592 | `[[185, 0], [15, 0]]` |
| MCQ | 200 | 76.00% | 76.77% | 98.70% | 86.36% | 0.511 | `[[0, 46], [2, 152]]` |
| Extract | 101 | 81.19% | 85.54% | 91.03% | 88.20% | 0.887 | `[[11, 12], [7, 71]]` |
| Classify | 33 | 72.73% | 60.00% | 54.55% | 57.14% | 0.764 | `[[18, 4], [5, 6]]` |
| Reformat | 64 | 87.50% | 100.00% | 20.00% | 33.33% | 0.940 | `[[54, 0], [8, 2]]` |

### Dev classifier performance

All dev rows are math. At threshold 0.5 the model predicts every row incorrect: **accuracy 35%, precision/recall/F1 0%**, ROC-AUC **0.377**; math confusion matrix `[[21, 0], [39, 0]]`. There are no other dev task groups to report.

| Policy on cached dev answers | Accuracy | Local call share | Strong call share |
|---|---:|---:|---:|
| Always local | 65% (39/60) | 100% | 0% |
| Always strong | 100% (60/60) | 0% | 100% |
| Classifier selective, threshold 0.5 | 100% (60/60) | 0% | 100% |

The classifier routes every dev row to strong, so it has no call-share advantage over always strong on this data.

### Grouped CV per task

| Task | n | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| Math | 200 | 92.50% | 0.00% | 0.00% | 0.00% | 0.559 | `[[185, 0], [15, 0]]` |
| MCQ | 200 | 77.00% | 77.27% | 99.35% | 86.93% | 0.482 | `[[1, 45], [1, 153]]` |
| Extract | 101 | 69.31% | 75.82% | 88.46% | 81.66% | 0.643 | `[[1, 22], [9, 69]]` |
| Classify | 33 | 72.73% | 62.50% | 45.45% | 52.63% | 0.773 | `[[19, 3], [6, 5]]` |
| Reformat | 64 | 84.38% | 0.00% | 0.00% | 0.00% | 0.865 | `[[54, 0], [10, 0]]` |

## Recommendation

Do **not** use this classifier as the current router. Retain it as an exploratory baseline, but retrain only after collecting a provenance-matched, representative math set and evaluating with source/template-aware splits. The current 0.5 classifier sends every dev math item to strong and saves no calls versus always strong. If correctness alone is the immediate objective on this particular cached dev set, always strong wins; that 60-row result is too narrow to recommend a project-wide policy. Obtain dev examples across the same sources, prompt regime, and task mix as training before choosing a selective policy or threshold.

## Reproducibility and exact commands

Repository root: `/Users/nsuchitra/llm-cost-router/llm-cost-router`. The analysis used Python 3.12 and the installed scikit-learn environment. Read-only inspection and analysis commands:

```sh
sed -n '1,280p' scripts/train_local_classifier.py
sed -n '1,180p' checker.py
cat results/local_correctness_research.json
.venv312/bin/python /tmp/audit_analyze.py
PYTHONPATH=. .venv312/bin/python /tmp/audit_more.py
```

The temporary analysis scripts loaded only the listed train/dev datasets, their labels and caches, plus the saved metrics; they did not load the test dataset. The grouped run used seed 506 and the grouping rules described above. No repository files other than this audit document were written.

## Files inspected

- `data/train_pool.jsonl`
- `data/train_labels.jsonl`
- `data/dev.jsonl`
- `cache/answers_train_pool.json`
- `cache/answers_dev.json`
- `scripts/train_local_classifier.py`
- `checker.py`
- `results/local_correctness_research.json`
- `scripts/build_datasets.py` to confirm source construction for dev and train
