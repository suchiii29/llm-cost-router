# LLM Cost Router: Final Research Project Report

**Project type:** educational prototype and offline research study
**Report date:** 2026-11-13
**Research scope:** rule-based routing prototype, local-answer correctness classifier experiment, and a paired 96-question cache-based model pilot

## Abstract

This project investigates whether a language-model router can reserve stronger hosted models for questions that are less likely to be answered correctly by a small local model. The software prototype combines a deterministic complexity heuristic, three model tiers, retry/fallback handling, request logging, an HTTP API, and a dashboard. The research pipeline separately builds checker-based local-answer labels, trains a question-only TF-IDF plus Logistic Regression classifier, and evaluates cached model answers against a frozen benchmark using each row's checker.

The completed 96-question pilot sampled a frozen 300-question test dataset by task and collected local, cheap, and strong answers for the same questions. Checker accuracy was 43.8% local, 84.4% cheap, and 87.5% strong. Exact paired McNemar tests found many more cheap/strong wins than local wins, while cheap and strong differed on only three questions (0 versus 3; two-sided exact p=0.25). These are results for this fixed pilot, not a measured estimate over the entire frozen test set. A task-proportion-weighted calculation is reported separately as exploratory. The classifier is not ready for live routing: its dev evaluation is limited to 60 math questions from a different source/provenance regime, it predicts every dev case as local-incorrect at the default threshold, and group/source-held-out analyses reveal weak transportability.

The work supports the feasibility of offline, resumable, paired evaluation and identifies a substantial accuracy gap between the local model and hosted tiers on the selected pilot. It does not establish actual cost savings, full-test performance, general semantic quality, or production readiness.

## 2. Problem statement and motivation

Using the strongest available model for every request can consume more provider capacity and cost than necessary. A small local model may be adequate for some requests, while others may benefit from a more capable hosted model. A router could reduce hosted usage if it can identify those requests reliably without causing unacceptable quality loss.

The core research question is whether question features can predict when the local model will pass the project's task-specific correctness checker, and whether that prediction can support selective routing. The system also explores a simpler rule-based complexity score. These are related but distinct: a complexity heuristic estimates apparent difficulty, while the classifier estimates local checker success from labeled examples.

## 3. Project objectives

The project objectives were to:

- Implement a demonstrable multi-tier router with local inference, hosted tiers, fallback behavior, and explicit user overrides.
- Record provider, model, usage, latency, and estimated cost metadata where available.
- Assemble reproducible train, dev, and frozen test datasets with stable IDs and checker definitions.
- Preserve raw answers and use deterministic checkers to derive labels and evaluate cached results offline.
- Train a simple question-only classifier as an initial research baseline, while keeping it separate from live routing.
- Run a small, deterministic, paired pilot before considering a larger provider-budget commitment.

The current implementation remains an educational prototype. Query decomposition, subtask aggregation, and quality-triggered live fallback are not implemented. The classifier is not connected to the runtime router.

## 4. System architecture and workflow

### Runtime routing

The compatibility entry point `router.py` delegates to the packaged implementation under `app/`. `app/routing/complexity.py` assigns points from question length, reasoning and technical keywords, context phrases, math symbols, advanced phrases, and numeric values. `app/routing/policy.py` maps a score below the configured threshold (default 3) to local tier 0 and a score at or above it to strong tier 2. The cheap tier 1 is available for explicit overrides and as a fallback; it is not the automatic LOW destination.

`app/routing/router.py` keeps the decision logic separate from provider access. A recoverable hosted failure can descend to the next cheaper tier: strong to cheap to local, or cheap to local. Authentication/configuration and other non-recoverable errors stop the chain. The result includes the selected model, answer, route score and reasons, fallback history, token usage, latency, and cost fields. Requests can be written to SQLite through `app/storage/`.

### Provider and evaluation workflow

`app/providers/ollama.py` sends non-streaming requests to the local Ollama generate endpoint. `app/providers/groq.py` sends hosted chat-completion requests. Both return a common response shape; provider usage is retained when supplied and missing token counts can be estimated for runtime display. `scripts/run_cache.py` is the explicit dataset-answer collector. It supports selected splits/tiers and a hash-validated manifest, skips successful answers, saves each outcome, and preserves existing successful records through `scripts/cache_store.py`.

Offline research then proceeds independently of live routing:

1. `scripts/build_datasets.py` constructs stable JSONL splits and a manifest.
2. `scripts/build_training_labels.py` matches train questions to raw local answers and computes `local_ok` with `checker.py`.
3. `scripts/train_local_classifier.py` fits TF-IDF and Logistic Regression, evaluates cross-validation, derives dev labels from the existing dev cache, and reports cached-policy comparisons.
4. `scripts/build_pilot_manifest.py` selects the fixed 96-question sample from the frozen test set.
5. `scripts/evaluate_split_cache.py` checks cached answers for all three tiers using each dataset row's checker.
6. `scripts/analyze_pilot_results.py` computes uncertainty intervals, exact paired correctness tests, ID-level comparisons, and latency summaries from those saved artifacts.

The API in `app/api.py` exposes health, route, configuration, and request-history endpoints. The Streamlit interface uses the API for submitted questions and reads stored request history for analysis. Replay mode uses a fake provider. Opening the dashboard does not itself call model providers.

## 5. Technologies used

- **Python 3.12** for routing, data preparation, evaluation, and model training.
- **Ollama** for local inference with `qwen2.5:1.5b`.
- **Groq API** for the hosted cheap and strong tiers.
- **Requests** and **Tenacity** for provider HTTP calls, retry handling, and backoff.
- **FastAPI**, **Pydantic/Pydantic Settings**, and **Uvicorn** for the HTTP API and configuration.
- **SQLAlchemy** with SQLite for request history.
- **Streamlit** for the dashboard.
- **scikit-learn** for TF-IDF, Logistic Regression, and classifier evaluation; **NumPy** and **joblib** support metrics and artifact persistence.
- **PyYAML** and environment-backed settings for configuration. This report did not inspect `.env`; checked-in `config.yaml` was used for configuration facts.
- **unittest**, **Ruff**, and Python compile checks for validation.

The saved classifier report records Python 3.12.16, scikit-learn 1.9.1, NumPy 2.5.4, and joblib 1.6.0 for that experiment. The project requirements pin several application dependencies; scientific packages are listed without exact versions, so reproductions should retain their environment/version records.

## 6. Dataset sources, composition, splits, and safeguards

The dataset builder combines GSM8K math questions, ARC-Challenge multiple-choice questions, and deterministic synthetic extraction, classification, and reformat tasks. The development split preserves 60 older benchmark questions. Every row carries a stable ID, split, task type, reference answer, and checker type.

| Split | Count | Composition |
|---|---:|---|
| `train_pool` | 598 | 200 GSM8K math, 200 ARC-Challenge MCQ, 198 synthetic simple tasks: 101 extract, 64 reformat, 33 classify |
| `dev` | 60 | 60 math questions from the existing benchmark set |
| `test` | 300 | 100 math, 100 MCQ, 51 extract, 33 reformat, 16 classify |

The train/test synthetic overlap audit found two exact normalized question collisions; only those two train rows were removed, with no replacements. Their historical cache answers remain as unmatched/orphaned records rather than being deleted. The frozen test file was preserved. Its SHA-256 is `417e709d72d19f3bb855ef22cb3704bc85b9d16152c091f32ea8e22c80fe9be5`.

The pilot manifest is deterministic with seed 506. It includes 20 math, 20 MCQ, 20 extract, 20 reformat, and all 16 classify questions (96 total), records the source dataset hash, and is checked against that hash before evaluation or manifest-scoped collection. The pilot deliberately overrepresents small task groups; the primary pilot accuracy is a simple sample accuracy, and the report separately gives an exploratory task-proportion-weighted estimate.

Dataset safeguards include unique IDs, split validation, stable source hashes, manifest-to-dataset ID/task checks, frozen test protection in the builder, atomic writes, and cache resume behavior that skips successful answers. Existing raw model responses are retained. Train/dev/test are kept separate for their defined roles; the frozen test set was not used to fit the classifier or choose its threshold.

## 7. Routing strategies and model tiers

| Tier | Provider and checked-in model | Role |
|---|---|---|
| 0 — local | Ollama, `qwen2.5:1.5b` | Local inference; configured model rates are zero |
| 1 — cheap | Groq, `openai/gpt-oss-20b` | Hosted lower tier and fallback from strong |
| 2 — strong | Groq, `openai/gpt-oss-120b` | Hosted high tier and default automatic destination for high-scoring questions |

The existing rule router maps low complexity to local and high complexity to strong. It does not automatically use tier 1 for a medium score because the current policy has only a low/high split. Manual overrides can select local, cheap, or strong. The historical `groq` override remains an alias for strong.

The evaluation baselines are always local, always cheap, and always strong over the same cached questions. Classifier-selective routing is an offline comparison only: use local when its predicted local-correctness probability is at least 0.5, otherwise strong. This classifier policy is not active in the application.

## 8. Experimental methodology

### Checker-based labels

`checker.py` dispatches by the row's checker field:

- **numeric:** extract a number and compare numerically with tolerance; comma-formatted ground truth is normalized.
- **letter:** compare a parsed multiple-choice letter to a single-letter key.
- **exact:** compare normalized Unicode text, case, and whitespace, preserving the primary output-format contract.
- **contains:** in the current implementation, require exactly one allowed classification label (`billing`, `technical`, or `account`) as the complete normalized response.

The checker is deterministic, not a semantic judge. In particular, strict exact checks penalize extra prefixes, punctuation/spacing changes, and Markdown fences; numeric extraction can select a final number from verbose output; and the letter checker depends on its response parsing rules. For JSON reformat tasks, the label builder has a separate semantic-array diagnostic that may identify equivalent parsed arrays after removing a complete Markdown fence. That diagnostic does not change primary correctness.

### Classifier experiment

The training set is `data/train_labels.jsonl`; the input features are question text and task type only. Ground truth, raw answer, ID, explanation, model ID, and cache metadata are excluded. The primary classifier is a pipeline of `TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)` and balanced `LogisticRegression(solver="liblinear", max_iter=2000, random_state=506)`. Five-fold shuffled `StratifiedKFold` uses seed 506, with pooled out-of-fold probabilities. The reference baseline always predicts the majority class. Dev labels are independently derived from dev questions and cached local/strong answers. No test rows are read by the classifier training script.

### Pilot experiment

The pilot is a fixed, stratified subset selected from the frozen 300-question test dataset. All three tiers answered the same 96 IDs. Existing cache answers were evaluated offline; no answer was silently corrected. Missing/failed answers are counted separately from incorrect responses. Accuracy uses successful cached answers. Overall and per-task confidence intervals use the 95% Wilson score method. Pairwise model correctness uses two-sided exact conditional McNemar tests. Latency analysis is descriptive, including same-question cheap-minus-strong differences; no latency significance test was performed. The pilot was not used to select a routing policy or tune a threshold.

## 9. Pilot results: overall and task-wise

All 96 selected questions have successful cached answers for each tier, with no failed or missing answers in the pilot evaluation.

| Tier | Correct | Incorrect | Accuracy | 95% Wilson CI |
|---|---:|---:|---:|---:|
| Local | 42/96 | 54 | 43.75% | 34.26%–53.72% |
| Cheap | 81/96 | 15 | 84.38% | 75.81%–90.30% |
| Strong | 84/96 | 12 | 87.50% | 79.41%–92.70% |

Per-task sample sizes are small (16 or 20), and the intervals are correspondingly wide. Parenthetical values are 95% Wilson confidence intervals.

| Task | n | Local | Cheap | Strong |
|---|---:|---:|---:|---:|
| Math | 20 | 2/20 (10%; 2.8–30.1%) | 18/20 (90%; 69.9–97.2%) | 20/20 (100%; 83.9–100%) |
| MCQ | 20 | 14/20 (70%; 48.1–85.5%) | 19/20 (95%; 76.4–99.1%) | 19/20 (95%; 76.4–99.1%) |
| Extract | 20 | 15/20 (75%; 53.1–88.8%) | 20/20 (100%; 83.9–100%) | 20/20 (100%; 83.9–100%) |
| Reformat | 20 | 4/20 (20%; 8.1–41.6%) | 8/20 (40%; 21.9–61.3%) | 9/20 (45%; 25.8–65.8%) |
| Classify | 16 | 7/16 (43.75%; 23.1–66.8%) | 16/16 (100%; 80.6–100%) | 16/16 (100%; 80.6–100%) |

Reformat accuracy is low for all tiers under the exact-output checker. The checker audit found common format failures such as extra prefixes, omitted punctuation, and fenced or spaced JSON, alongside genuine content errors. The JSON semantic-equivalence result is diagnostic only and must not be substituted for the exact primary metric.

The raw label file currently contains **268 correct of 598 (44.82%)** local training answers. Per task it contains math 15/200 (7.5%), MCQ 154/200 (77.0%), extract 78/101 (77.23%), classify 11/33 (33.33%), and reformat 10/64 (15.63%). The earlier audit's headline of 270/600 reflects the pre-remediation dataset/checker state; it is not the current sidecar count. The current label sidecar and saved classifier report agree on 268 positives.

Task-proportion-weighted exploratory estimates using the full test task counts (math 100, MCQ 100, extract 51, reformat 33, classify 16) are local 43.95%, cheap 88.40%, and strong 92.28%. They extrapolate pilot task accuracies and are **not measured full-test accuracy**.

## 10. Paired statistical analysis and confidence intervals

The exact two-sided McNemar test uses the discordant paired correctness outcomes. P-values below are nominal and were not adjusted for the three comparisons.

| Pair | Paired n | Left only correct | Right only correct | Exact p-value |
|---|---:|---:|---:|---:|
| Local vs cheap | 96 | 4 | 43 | 2.78 × 10⁻⁹ |
| Local vs strong | 96 | 4 | 46 | 4.46 × 10⁻¹⁰ |
| Cheap vs strong | 96 | 0 | 3 | 0.25 |

Within this pilot, the hosted tiers were correct on many questions where local was not. The cheap-versus-strong comparison has only three discordant pairs and does not show a clear difference. Statistical significance within the sampled pilot does not resolve representativeness or generalization beyond these questions.

The overall 95% Wilson intervals are local 34.3%–53.7%, cheap 75.8%–90.3%, and strong 79.4%–92.7%. Task-wise intervals are shown above and are also included in `docs/PILOT_STATISTICAL_ANALYSIS.md` and `results/pilot_statistical_analysis.json`; their small denominators (16–20) make precision limited, even when all sampled answers were correct.

## 11. Latency analysis

Only valid numeric cache latency metadata was used. All tiers had 96 valid values. Latency distributions were:

| Tier | Mean | Median | Q1–Q3 | Range |
|---|---:|---:|---:|---:|
| Local | 0.1558 s | 0.1080 s | — | 0.074–2.046 s |
| Cheap | 1.8005 s | 1.2865 s | 0.5973–2.2405 s | 0.245–6.831 s |
| Strong | 2.0056 s | 1.5855 s | 0.6253–2.7728 s | 0.416–7.560 s |

For the 96 matched cheap/strong observations, cheap-minus-strong latency averaged −0.2051 s and had a median of −0.1475 s; cheap was faster for 53 questions and strong for 43, with no ties. These are observed descriptive differences. No statistical test or general speed claim is made. Latency can vary with model serving, provider load, network conditions, prompt length, and measurement setup.

## 12. Classifier experiments and generalization limitations

The saved question-only classifier achieved 83.44% pooled random-fold accuracy, 78.84% precision, 86.19% recall, 82.35% F1, and 0.895 ROC-AUC. Its majority-class baseline accuracy was 55.18%. These aggregate cross-validation metrics overstate what can be inferred about robust within-task prediction: task type and class prevalence explain much of the apparent performance.

The current train pool contains 200 GSM8K math questions with 15 local positives (7.5%). All 60 dev questions are math from an older benchmark set, with 39 local positives (65%). Both splits use the numeric checker, so this is not a checker-semantic mismatch. It is a source and generation-regime mismatch: training math outputs are short answers from the saved Qwen run, while dev answers were imported historical multiline responses without verifiable model/prompt provenance. The source audit found no exact train/dev math overlap. Training questions also exhibit more multi-step indicators. The observed gap is consistent with distribution shift and is not evidence that the checker created the 7.5% train figure.

At threshold 0.5 the classifier predicted every dev math question as local-incorrect: dev classifier accuracy was 35%, with zero recall for local-correct items. It selected strong for every dev question. The cached comparisons were 39/60 local, 60/60 strong, and 60/60 classifier-selective; the selective rule made no call-share reduction relative to always strong. Dev contains no non-math tasks, so it cannot validate classifier behavior for the rest of the benchmark.

Random-fold evaluation also mixes synthetic task templates between folds. A grouped five-fold diagnostic that kept related templates together reported 81.44% accuracy, 76.43% precision, 84.70% recall, 80.35% F1, and 0.861 ROC-AUC. However, grouped math performance still predicted every row negative, with 92.5% accuracy but zero recall for the 15 positives. A source-held-out diagnostic reported 26.76% accuracy, 19.42% precision, 20.15% recall, 19.78% F1, and 0.195 ROC-AUC; it is confounded because source groups closely correspond to task families, but it underscores weak transportability.

**Conclusion on the classifier:** retain it as an exploratory baseline, not as a live router. Retraining or threshold adjustment on the same mismatched data would not resolve the validation problem. Live use would require representative, provenance-matched labels, task/source-aware validation, calibration assessment, and a locked policy evaluated prospectively on untouched data.

## 13. Discussion and research findings

1. **The prototype demonstrates the end-to-end mechanics.** Rule scoring, provider selection, recoverable fallback, manual overrides, metadata capture, an API, and a dashboard are implemented as separable components.
2. **The pilot shows a large observed local-to-hosted quality gap for this sample.** Local achieved 43.8%, versus 84.4% cheap and 87.5% strong under the project checkers.
3. **The pilot does not establish that strong is better than cheap.** The 3 discordant correctness cases favor strong, but the exact McNemar p-value is 0.25 and overall point estimates are close.
4. **Task contracts affect measured correctness.** Exact formatting requirements are part of the primary reformat label. Semantic equivalence may be informative, but changing the primary label after seeing outcomes would alter the target and should be treated as a separate study.
5. **The classifier's random CV score is not sufficient evidence for routing.** Group/template overlap, skewed task-specific label rates, and source shift make generalization more important than aggregate random CV accuracy.
6. **Cost savings remain unmeasured.** The checked-in Groq input/output price rates are blank. No actual billed cost or savings is claimed. Model-call share and latency are not substitutes for verified cost.

## 14. Limitations and threats to validity

- **Pilot sampling uncertainty:** 96 questions are a small subset, with only 16 classification and 20 questions in each other task. Task-wise intervals are wide.
- **Task-weighting assumptions:** the exploratory weighted estimate assumes each pilot task accuracy applies to every question in that task in the full test distribution. This is unvalidated.
- **Single response per model/question:** generation variability, sampling settings, prompt versions, and repeated-run stability are not measured by this pilot.
- **Checker validity:** numeric parsing, letter extraction, strict exact comparison, and controlled-label compliance are deterministic approximations to the intended task contracts. They are not general semantic evaluation.
- **Source and prompt mismatch:** current train math and dev math originate from different sources and have different answer provenance. Historical dev records lack provider/model/prompt metadata.
- **Synthetic templates:** synthetic simple tasks can share template patterns and are easier/more regular than real user traffic.
- **Classifier evaluation:** random CV can leak template style across folds; grouped and source-held-out estimates are themselves heuristic and may be confounded. The dev set's math-only composition limits external validity.
- **Latency measurement:** a single observed run is sensitive to service and network conditions; no latency inference was performed.
- **Cost configuration:** Groq pricing is not configured in checked-in YAML. Per-account billing, retries, and current rates have not been reconciled to these pilot observations.
- **No full-test comparison:** the pilot covers 96 of the 300 frozen questions. Full-test metrics cannot be inferred as measured results from the pilot.
- **Prototype scope:** no decomposition/aggregation, learned live routing, semantic quality gate, or production-grade traffic study is included.

## 15. Future improvements

- Collect a larger task- and source-stratified evaluation, ideally all 300 frozen questions if budget permits, with repeated runs or documented deterministic generation settings where feasible.
- Build provenance-matched training and validation data across sources and tasks; keep related templates and near-duplicates grouped during validation.
- Define correctness targets explicitly. Keep exact output compliance separate from semantic or task-level correctness diagnostics, and independently review checker edge cases.
- Evaluate calibration, selective risk/coverage, and per-task call shares using a policy locked before held-out evaluation. Compare against simple always-local/always-hosted and rule-based baselines.
- Verify current hosted pricing and provider billing metadata before making any cost claims. Report actual measured usage and uncertainty rather than assumed prices.
- Test routing on realistic, privacy-safe traffic and operational conditions, including provider failures and fallback frequency, before considering production deployment.
- If product scope requires it, explore decomposition and aggregation as a separate research phase with explicit evaluation criteria; those capabilities are not part of the current implementation.

## 16. Conclusion

The project delivered a working educational router and a reproducible offline research workflow. On the fixed 96-question pilot, the two hosted tiers substantially outperformed the local model under the project checkers, while the observed difference between cheap and strong was small and inconclusive. The question-only classifier is not suitable for live routing because its validation regime does not represent the training/test task and source mix, and its dev policy made no savings in hosted call share.

The evidence justifies a larger, preregistered, representative paired experiment; it does not justify claims of full-test accuracy, production routing quality, or dollar savings. The frozen dataset, raw cache answers, checker-based labels, evaluation outputs, and documented limitations provide a traceable basis for that next study.

## 17. Reproducibility instructions

Run from the repository root. These commands reproduce the offline labels, classifier artifacts, manifest, pilot evaluation, statistical report, and test checks from the checked-in data/cache inputs. Label/classifier generation rewrites their respective derived outputs; it does not make provider calls. Dataset building can download source datasets, so it is not needed when the checked-in JSONL datasets are available.

```sh
# Install dependencies in a clean environment (Python 3.12 recommended).
python3 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements.txt

# Rebuild checker-derived labels from existing raw local answers.
python scripts/build_training_labels.py

# Refit the question-only research baseline and regenerate its metrics.
python scripts/train_local_classifier.py

# Recreate the deterministic 96-question manifest (reads the frozen test file).
python scripts/build_pilot_manifest.py

# Evaluate the already collected cache offline; this performs no provider calls.
python scripts/evaluate_split_cache.py
python scripts/analyze_pilot_results.py

# Validate code and tests.
python -m unittest discover -s tests -v
ruff check .
python -m compileall -q .
git diff --check
```

The deterministic pilot uses seed 506 and source test SHA-256 `417e709d72d19f3bb855ef22cb3704bc85b9d16152c091f32ea8e22c80fe9be5`. Its saved artifacts are `data/pilot_manifest.json`, `results/pilot_cache_evaluation.json`, `docs/PILOT_CACHE_EVALUATION.md`, `results/pilot_statistical_analysis.json`, and `docs/PILOT_STATISTICAL_ANALYSIS.md`.

If recollecting answers instead of using the existing cache, the following commands make provider calls and may consume quota; they are not part of offline reproduction:

```sh
python scripts/run_cache.py --split test --tier local --manifest data/pilot_manifest.json
python scripts/run_cache.py --split test --tier cheap --manifest data/pilot_manifest.json
python scripts/run_cache.py --split test --tier strong --manifest data/pilot_manifest.json
```

The manifest contains 96 IDs, implying 96 local Ollama requests and 192 hosted Groq requests if no answers are already cached and no retries occur. Successful cache records are resumable and are not overwritten. Runtime configuration may be overridden by environment variables; `.env` was not read for this report.

## Project records and related documentation

This report consolidates, rather than replaces, the detailed implementation and analysis records:

- `README.md` — setup, runtime interfaces, dataset/cache workflow, and usage notes.
- `docs/STATUS.md` — implementation status and legacy compatibility notes.
- `docs/FINAL_EXPERIMENT_READINESS.md` — split sizes, cache coverage, metadata, cost configuration, and call budgeting before the pilot.
- `docs/LABEL_QUALITY_AUDIT.md` — checker audit, example errors, overlap remediation, and primary-versus-semantic output definitions.
- `docs/LOCAL_CLASSIFIER_RESEARCH.md` — classifier features, training procedure, metrics, hashes, and cached dev comparisons.
- `docs/CLASSIFIER_GENERALIZATION_AUDIT.md` — source/task shift, grouped evaluation, and recommendation against live routing.
- `docs/PILOT_CACHE_EVALUATION.md` — task-level pilot scoring and cache metadata.
- `docs/PILOT_STATISTICAL_ANALYSIS.md` — confidence intervals, exact paired tests, question IDs, and descriptive latency statistics.

The README also contains historical benchmark results and classifier plans written before the current labels and pilot existed; those statements are project history, not the current experimental result. The current classifier and pilot documents listed above describe the later research stages. No `AGENTS.md` file was found in the repository parent search.

No `.env` values are included or required to interpret this report. No new provider calls or answer generation were performed while preparing it.
