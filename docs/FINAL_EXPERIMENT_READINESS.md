# Final Experiment Readiness

## Dataset size and cache coverage

Dataset counts were read from the JSONL files and cross-checked against `data/MANIFEST.json`.

| Split | Questions | Successful local | Failed local | Missing local | Successful cheap | Failed cheap | Missing cheap | Successful strong | Failed strong | Missing strong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train_pool` | 598 | 598 | 0 | 0 | 0 | 0 | 598 | 0 | 0 | 598 |
| `dev` | 60 | 60 | 0 | 0 | 0 | 0 | 60 | 60 | 0 | 0 |
| `test` | 300 | 0 | 0 | 300 | 0 | 0 | 300 | 0 | 0 | 300 |
| **Total question-tier cells** | **958** | **658** | **0** | **300** | **0** | **0** | **958** | **60** | **0** | **898** |

“Failed” means the current tier has a recorded failure and no successful answer for that question. No tier failure records exist in these split caches. The test cache file `cache/answers_test.json` does not exist, so its 900 question-tier cells are all missing.

The train cache has 600 entries although the current train dataset has 598 questions. Its two extra records, `simple-train-0010` and `simple-train-0064`, are historical cache entries without corresponding rows in the current train pool. They are not counted as current successes, failures, or missing answers. Do not remove them as part of experiment preparation.

## Cache-to-dataset association and record metadata

All three datasets have unique IDs and each row's `split` matches its file. Existing caches declare the corresponding split. For every current cache entry, the mapping key equals the embedded entry ID and the stored question text exactly matches its dataset row. Dev has exactly 60 matching entries and no extras. Train has all 598 current IDs and exact question matches, plus the two historical orphans above. There is no test cache to validate.

Metadata below describes successful records for the matching split and tier. Token counts are shown as `input/output` records with usable integer values. Latency is counted where a numeric `seconds`, `latency_seconds`, or `latency_ms` value is present.

| Split / tier | Token counts | Latency | Provider | Model ID | Timestamp | Notes |
|---|---:|---:|---:|---:|---:|---|
| Train / local (598) | 598 / 598 | 598 | 598 | 598 | 597 | Model ID is stored under legacy key `model` rather than canonical `model_id`; all identify `qwen2.5:1.5b`. One row, `simple-train-0058`, has no timestamp. Provider is `ollama`. |
| Train / cheap or strong | 0 | 0 | 0 | 0 | 0 | No successful records. |
| Dev / local (60) | 0 / 0 | 60 | 0 | 0 | 0 | Imported records; latency is retained as `seconds`, but provider/model/token/timestamp provenance is absent. |
| Dev / cheap | 0 | 0 | 0 | 0 | 0 | No successful records. |
| Dev / strong (60) | 0 / 0 | 60 | 0 | 0 | 0 | Imported records; latency is retained as `seconds`, but provider/model/token/timestamp provenance is absent. |
| Test / any tier | 0 | 0 | 0 | 0 | 0 | No cache file or successful records. |

The answer text is present for every successful record counted above. Historical development answers should not be represented as newly measured calls with complete provider metadata.

## Models and pricing configuration

Checked-in `config.yaml` selects local Ollama `qwen2.5:1.5b`, cheap Groq `openai/gpt-oss-20b`, and strong Groq `openai/gpt-oss-120b`. Local input/output rates are configured as zero. Both Groq input and output rate fields are blank/null for both hosted tiers. Therefore, **actual Groq pricing is not configured in the checked-in project configuration, and no cost estimate is supportable from it**.

`app/config.py` supports environment overrides for model IDs and rates. Per instruction, `.env` was not read; runtime environment overrides were not inspected. Runtime pricing cannot be confirmed from this read-only audit. No public price was looked up or substituted.

## Evaluation code readiness

| Script | Reusable for | Gaps for the final standardized experiment |
|---|---|---|
| `scripts/run_cache.py` | Explicit split/tier collection into the new `answers_<split>.json` format; skips existing successful records and saves after each attempt. | It makes live calls when run; it is a collector, not an evaluator. No collection was run for this report. |
| `checker.py` | Offline scoring with the dataset-selected numeric, letter, exact, or contains checker. | The legacy `run_benchmark.py` and `evaluate.py` use numeric-only `is_correct`, not the general dataset checker dispatch. |
| `scripts/build_training_labels.py` | Existing offline train local-label generation and validation. | Only produces local correctness labels for `train_pool`; it does not evaluate cheap/strong or test policy comparisons. |
| `scripts/train_local_classifier.py` | Existing question-only classifier CV plus local/strong comparison on cached dev. | Does not evaluate cheap or frozen-test performance. Its saved dev set is math-only and its classifier currently routes every dev example to strong. |
| `run_benchmark.py` | Live comparison of always-cheap, always-strong, and rule-router strategies on the old `benchmark_questions.json` (60 questions). | Defaults to 5 evenly spaced old benchmark questions; `--full` is still that 60-question file, not `data/test.jsonl`. Calls providers directly rather than using the new split caches and uses the numeric checker. It is not a final frozen-test evaluator. Its printed maximum `sample_size * 4` is below the route's actual worst case: on a high-complexity row the router can try strong, cheap, then local, giving up to 5 total calls across the three modes. |
| `evaluate.py` | Historical evaluation over old model-ID-keyed `answers_cache_groq.json` pairs. | That cache is absent; its input schema and `benchmark_questions.json` are not the new split caches or mixed-checker test dataset. |
| `evaluate_hybrid.py`, `test_classifier.py`, `train_classifier.py` | Historical classifier artifacts and analyses. | Use older labels/cache/model formats; do not constitute evaluation of current split caches or the new classifier on the final test. |

There is currently **no complete offline evaluator for the new split cache format** that applies each row's checker and compares local, cheap, strong, and router policies on the standardized test dataset. The cache collector and checker are reusable building blocks, but a final cache-based comparison still needs an evaluation procedure. No code was added in this readiness audit.

## Additional provider-request counts

Two scopes should be distinguished:

1. **Complete the full train/dev/test cache matrix for all three tiers:** 2,156 logical generation requests for missing question-tier answers: 1,196 train hosted answers, 60 dev cheap answers, and 900 test answers. Provider breakdown: **1,856 Groq requests** (cheap/strong) plus **300 Ollama requests** (local).
2. **Run the final held-out test comparison across all three tiers:** test has no cache, so **900 logical requests**: 300 local Ollama, 300 cheap Groq, and 300 strong Groq. These are output-generation calls before retries. Provider-level retry attempts, if any, increase actual HTTP attempts. Filling train/dev cheap and strong tiers is not required for a test-only final comparison.

These counts assume one successful generation call per requested question/tier, with no provider retries or router fallbacks. They are derived directly from the missing cells in the table, not from estimated costs.

For reference, the older documented `run_benchmark.py --full` run has 60 questions and 180 planned policy requests. It can make up to 120 additional router fallback calls (up to 300 total in the worst case), but still evaluates the old benchmark file rather than the frozen 300-question test set.

## Recommended limited-budget experiment

Use a **predeclared, deterministic, paired sample from `data/test.jsonl`**, scored only after the policy/model choices and threshold are frozen. Current test composition is 100 math, 100 MCQ, 51 extract, 33 reformat, and 16 classify questions.

A practical task-stratified pilot is 96 questions: sample 20 each from math, MCQ, extract, and reformat, and include all 16 classification questions. Generate all three tiers for those same IDs. This costs **288 logical generation calls**: 192 Groq calls and 96 local calls, before retries. Select rows with a fixed documented seed; report per-task results and weight the overall estimate back to the test task proportions because this sample deliberately overrepresents the smaller tasks. Report uncertainty and label the result a pilot; 16 classification cases and 20 cases in other groups do not support narrow per-task confidence intervals.

If the project needs a strong final claim about accuracy, use all 300 test questions at all three tiers (900 logical calls) instead. Do not use test outcomes to tune the classifier, router, or threshold. For either scope, configure and verify current Groq rates outside this audit before making cost claims, then use cached outputs and each dataset row's checker for paired offline evaluation. The currently available evaluation scripts cannot produce that final report as-is.

## Files inspected

- `data/train_pool.jsonl`, `data/dev.jsonl`, `data/test.jsonl`, `data/MANIFEST.json`
- `cache/answers_train_pool.json`, `cache/answers_dev.json` (no `cache/answers_test.json` exists)
- `scripts/cache_store.py`, `scripts/run_cache.py`, `scripts/build_training_labels.py`, `scripts/train_local_classifier.py`
- `checker.py`, `config.yaml`, `app/config.py`, `model_config.py`
- `run_benchmark.py`, `evaluate.py`, `evaluate_hybrid.py`, `train_classifier.py`, `test_classifier.py`
- `README.md`, `docs/STATUS.md`, `docs/LOCAL_CLASSIFIER_RESEARCH.md`, `docs/CLASSIFIER_GENERALIZATION_AUDIT.md`

`.env` was not read. No provider/network calls were made. No dataset, label, cache, model, or existing evaluation artifact was modified.
