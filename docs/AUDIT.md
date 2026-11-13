# Repository audit

Audit performed 2026-11-11. The Git repository root is the nested `llm-cost-router/` directory. There is no `AGENTS.md` in this repository or its parent tree. “Import-time work” means externally visible I/O, network/model calls, or application execution during import; ordinary imports and constant/function definitions are not counted. Status reflects the file's present role in the project, not whether its behavior is bug-free.

## Tracked-file inventory

| Tracked file | Purpose | Status | Work at import time? |
|---|---|---|---|
| `.gitignore` | Excludes local credentials, environments, generated Groq files and Python caches. | complete | No |
| `README.md` | Setup, operation, benchmark interpretation and limitations. | partial | No |
| `__pycache__/checker.cpython-310.pyc` | Compiled historical checker bytecode. | legacy | No |
| `__pycache__/model_adapter.cpython-310.pyc` | Compiled historical adapter bytecode. | legacy | No |
| `__pycache__/router.cpython-310.pyc` | Compiled historical router bytecode. | legacy | No |
| `__pycache__/tracker.cpython-310.pyc` | Compiled historical tracker bytecode. | legacy | No |
| `__pycache__/train_classifier.cpython-310.pyc` | Compiled historical classifier-training bytecode. | legacy | No |
| `answers_cache.json` | Saved historical local/strong model answer pairs for classifier experiments. | legacy | No |
| `benchmark_local.py` | Live Groq cheap/strong timing benchmark despite legacy filename. | legacy | No |
| `benchmark_questions.json` | Benchmark questions and numeric expected answers. | complete | No |
| `benchmark_results.json` | Historical benchmark result artifact; not current Groq evidence. | legacy | No |
| `checker.py` | Extracts a number and compares it with the expected number. | partial | No |
| `classifier.pkl` | Serialized historical classifier. | legacy | No |
| `docs/STATUS.md` | Project status, design and caveats. | partial | No |
| `docs/guide_demo.md` | Demo flow and live-call cautions. | partial | No |
| `download_benchmark.py` | Downloads benchmark data and overwrites `benchmark_questions.json` when run. | legacy | No |
| `evaluate.py` | Offline comparison of complete cached Groq answer pairs; emits result files. | partial | No |
| `evaluate_hybrid.py` | Historical saved-data classifier/router analysis. | legacy | No |
| `labels.json` | Historical benchmark labels. | legacy | No |
| `list_models.py` | Live Groq model-list request when executed. | partial | No |
| `model_adapter.py` | Groq chat-completions API adapter and successful-call metrics append. | complete | No |
| `model_metrics.jsonl` | Historical local-model metrics. | legacy | No |
| `output.txt` | Historical sample output referring to prior providers/models. | legacy | No |
| `requirements.txt` | Runtime and dashboard package requirements. | partial | No |
| `router.py` | Rule score, two-tier Groq routing, override, recoverable fallback and CLI. | partial | No |
| `routing_log.csv` | Historical routing log. | legacy | No |
| `run_benchmark.py` | Live comparison of cheap, strong and router policies; writes result files. | partial | No |
| `run_cache.py` | Resumable live collection of cheap/strong Groq cache pairs. | partial | No |
| `run_train_labels.py` | Resumable live cheap-Groq labeling script. | partial | No |
| `test_classifier.py` | Historical classifier evaluation using saved data. | legacy | No |
| `test_ollama.py` | Live Groq smoke test despite legacy filename; does not call Ollama. | legacy | No |
| `tests/test_core.py` | Mock-based routing, adapter, cost and benchmark tests. | partial | No |
| `tests/test_dashboard.py` | Offline dashboard data and submission tests. | partial | No |
| `tracker.py` | Per-request cost estimates and Groq routing CSV logging/CLI. | partial | No |
| `train_classifier.py` | Historical classifier training and serialization. | legacy | No |
| `train_labels.json` | Historical classifier training labels. | legacy | No |

There are **36 tracked files**. The working tree also contains untracked dashboard/config files and generated Groq artifacts; those are not part of the tracked-file table. The explicitly protected data/model files were inspected only as repository artifacts and were not changed.

## Providers, models and score mapping

The live adapter in `model_adapter.py` sends HTTPS POST requests to Groq's OpenAI-compatible chat-completions endpoint. `model_config.get_model_config()` defaults the configured IDs to `openai/gpt-oss-20b` (cheap) and `openai/gpt-oss-120b` (strong), with environment overrides. The code makes no local Ollama call. Legacy `force local:` and `force groq:` command aliases map to cheap Groq and strong Groq respectively; they do not select a local model. `test_ollama.py` and `benchmark_local.py` also call the Groq adapter.

`router.calculate_complexity()` (in `router.py`) returns a label, numeric score and reasons. Its heuristic adds points for question length, reasoning keywords, technical keywords, context phrases, math symbols, advanced phrases and multiple numbers. The threshold in the function is **score >= 3 → HIGH**, otherwise **LOW**. In `route_question()`, LOW selects the configured cheap Groq model and HIGH selects the configured strong Groq model. Manual overrides can replace this automatic choice. A recoverable strong-tier failure tries the cheap Groq model. The trained classifier is not invoked by the live router.

## Referenced paths that are absent

At audit time, `answers_cache_groq.json` does not exist. `evaluate.py` reads it and `run_cache.py` creates it on an explicit live run. Dynamic output families referenced by code also do not exist yet: `labels_groq_<timestamp>.json`, `benchmark_results_groq_<timestamp>.json`, and `benchmark_results_groq_live_<timestamp>.json`; their writers are `evaluate.main()` and `run_benchmark.main()`. `train_labels_groq.json` is an output of `run_train_labels.py` and is absent. These are expected generated paths, rather than missing source dependencies.

The current worktree has `model_metrics_groq.jsonl` and `routing_log_groq.csv` as untracked outputs; both exist, despite not being tracked. The exact historical live benchmark result named in the README exists. No reference to `answers_cache_groq.json` is a reference to the protected `answers_cache.json`.

## Secrets scan and ignore rules

Scanned tracked text files for token-like strings matching the requested `gsk_`, `AIza`, and `sk-` patterns. The scan produced **no file/line matches**. Secret values were not printed. `.env` was not opened or scanned. `.env.example` is untracked and was not included in the tracked-file scan; it is intended as the blank template. This scan is pattern-based and cannot establish that no other secret format exists.

`.gitignore` covers `.env`, `.env.*` (with an explicit exception for `.env.example`), `__pycache__/`, and the project environment directories `.venv/`, `.venv312/`, and `venv/`. It also ignores generated Groq outputs. `.env.example` is deliberately not ignored.

## Requested unittest run

Ran `python3 -m unittest discover -s tests` from the repository root, with no live network or model calls. Result: **FAILED during discovery; 0 test cases ran**. Both test modules raised `ModuleNotFoundError: No module named 'requests'` under the system Python 3.9.6. Also ran the same discovery command with `.venv/bin/python`; again **0 tests ran**, this time because `python-dotenv` is missing. The project venv also reports Python 3.9.6, below the stated Python 3.10+ project rule. The `.venv312` environment was not needed to establish the requested command's result.

## Requirement matrix

| Requirement | Status | File and function evidence |
|---|---|---|
| API interface | Implemented | `model_adapter.py:ask_groq_model()` calls Groq's chat-completions endpoint; `router.py:main()` and `tracker.py:main()` are terminal interfaces; `streamlit_app.py` provides the dashboard form. |
| Complexity score | Implemented | `router.py:calculate_complexity()` computes heuristic points, labels and reasons. This is not a trained difficulty estimator. |
| Per-request display of model/reason/cost/latency | Partial | `router.py:route_question()` displays model, complexity, score and reasons. `tracker.py:track_question()` displays cost and total latency; its terminal path supplies request details. The Streamlit result in `streamlit_app.py` shows model, reasons, estimated cost and response latency. Cost may be unavailable until rates are configured. |
| Manual override | Implemented | `router.py:route_question(force_model=...)` accepts cheap/strong and local/groq aliases; `router.py:main()` parses force prefixes. |
| Logging | Implemented | `tracker.py:save_log()` appends request history to `routing_log_groq.csv`; `model_adapter.py:record_metrics()` appends successful-call metadata to `model_metrics_groq.jsonl`. |
| Baseline comparison | Partial | `tracker.py:estimate_cost()` computes a counterfactual strong-model cost from the routed response's token counts; this is not a separately measured strong response. `run_benchmark.py:run_benchmark()` directly runs all three strategies on a common sample when explicitly invoked. |
| Availability fallback | Partial | `router.py:route_question()` retries a recoverable strong-model error using cheap Groq. There is no local or second-provider fallback, and same-provider outages/rate limits can affect both attempts. |
| Quality-based fallback | Missing | `checker.py:is_correct()` is only used in evaluation scripts; `router.route_question()` does not evaluate answer quality or retry on a quality signal. |
| Local tier | Missing | No Ollama client/call exists in active routing; both tiers use Groq through `model_adapter.py`. |
| Classifier used in live router | Missing | `train_classifier.py:main()` trains a historical classifier; no classifier load or prediction occurs in `router.py:route_question()`. |
| Query decomposition | Missing | No routing or application function splits a question into subquestions and combines their answers. |

## Ten highest-risk problems

1. **The requested unit-test command currently executes zero tests.** Dependencies are missing in both Python environments tried, and those environments are Python 3.9.6 despite the Python 3.10+ project requirement.
2. **`download_benchmark.py` can overwrite protected `benchmark_questions.json`.** Its `main()` opens that path for writing; this conflicts directly with the project’s protected-file rule.
3. **Several explicitly live scripts can incur provider usage.** `router.py`, `tracker.py`, `run_cache.py`, `run_benchmark.py`, `run_train_labels.py`, `list_models.py`, `benchmark_local.py`, and `test_ollama.py` make live calls when explicitly run; names such as `test_ollama.py` look like tests but are a live smoke test.
4. **Availability fallback stays within one provider/account.** A Groq 429, outage, or connection failure can affect both strong and cheap attempts; there is no local or independent-provider fallback.
5. **Benchmark correctness is numeric-only.** `checker.is_correct()` extracts a number, so it cannot establish semantic answer quality and may mis-score explanations with unrelated numbers.
6. **The available test suite does not cover every critical workflow, and it was not runnable here.** In particular, no live-call path is tested live by design; environment setup currently prevents even mock tests from loading.
7. **The current complexity score is a brittle keyword heuristic.** `router.calculate_complexity()` can correlate with length, symbols or listed terms without reliably measuring difficulty; no validation evidence establishes its routing quality.
8. **Tracker baseline savings are counterfactual estimates.** `tracker.estimate_cost()` reuses the selected response's token counts for the strong baseline, so the baseline response and its usage are not measured.
9. **Cost depends on manually configured rates and fallback token handling.** Missing rates produce unavailable cost, and tracker estimates missing token counts from characters; displayed cost is not provider billing.
10. **Legacy and generated artifacts can be mistaken for current experiment evidence.** Tracked caches, labels, classifier and results are historical; generated Groq caches/results are absent or untracked. The README and status docs do label several of these caveats, but care is still needed when presenting outputs.
