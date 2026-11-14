# LLM Cost Router

## Project overview

LLM Cost Router is an educational prototype and research project for comparing local and hosted language-model tiers, recording routing outcomes, and evaluating answer correctness from cached responses. The runtime router uses a rule-based complexity score; a separate question-only classifier was evaluated as an exploratory experiment and is **not used for live routing**. This is not a production-ready service, and no verified dollar savings are claimed.

The project combines a Python routing/API layer, Ollama and Groq provider adapters, a Streamlit dashboard, reproducible train/dev/frozen-test datasets, and offline label/evaluation scripts. The completed 96-question test pilot found 42/96 local answers correct (43.75%), 81/96 cheap answers correct (84.38%), and 84/96 strong answers correct (87.50%). These are **pilot results, not full-test results**. Cheap versus strong was not statistically significant in the exact paired McNemar test (p=0.25).

For the full methodology and interpretation, see the [final project report](docs/FINAL_PROJECT_REPORT.md), [pilot cache evaluation](docs/PILOT_CACHE_EVALUATION.md), [pilot statistical analysis](docs/PILOT_STATISTICAL_ANALYSIS.md), [classifier research](docs/LOCAL_CLASSIFIER_RESEARCH.md), [classifier generalization audit](docs/CLASSIFIER_GENERALIZATION_AUDIT.md), [label quality audit](docs/LABEL_QUALITY_AUDIT.md), and [final experiment readiness review](docs/FINAL_EXPERIMENT_READINESS.md).

## Problem and motivation

Calling a powerful language model for every question can spend more API quota and money than necessary. Many questions are simple enough for a local model, while questions needing several reasoning steps may benefit from a stronger hosted model. This project demonstrates a rule-based router that chooses among Ollama and two Groq tiers and records what happened.

The project is an educational prototype, not a production routing service. It estimates cost from configured rates and token counts; it does not report Groq's billed amount.

## Architecture and workflow

1. `app/routing/complexity.py` scores the question using word count, keyword groups, phrases, math symbols, and number counts.
2. `app/routing/policy.py` maps scores below the configured threshold (default 3) to local tier 0 and scores at or above it to strong tier 2.
3. `app/routing/router.py` calls the selected provider and records the answer, route, fallback history, usage, latency, and estimated cost fields. Recoverable failures descend to cheaper tiers; non-recoverable errors stop.
4. `app/providers/` implements Ollama and Groq adapters. The API is in `app/api.py`, SQLite request storage is in `app/storage/`, and the Streamlit UI is in `streamlit_app.py`.
5. Separate offline scripts build datasets and labels, train the research classifier, and evaluate the deterministic test pilot from saved cache records. They do not put the classifier in the live route path.

The tiers are **local** (Ollama `qwen2.5:1.5b`), **cheap** (Groq `openai/gpt-oss-20b`), and **strong** (Groq `openai/gpt-oss-120b`). Automatic low-complexity routing selects local; high-complexity routing selects strong. Cheap is available by manual override and as a fallback from strong. On recoverable errors, strong can fall back to cheap and then local; cheap can fall back to local. Model IDs, Ollama URL, Groq credentials, and optional rates can be configured through environment variables. Local model cost is configured as zero; hosted cost remains unavailable without both input and output rates.

### Complexity score

The current rule-based score adds points for:

- Question length: 1 point over 20 words, 2 points over 40 words.
- Reasoning and technical keywords: 2 points for matches in each group.
- Context-heavy terms: 1 point.
- Math symbols: 1 point for one symbol or 2 points for multiple symbols.
- Listed advanced phrases: 2 points.
- Numeric values: 1 point for at least two, or 2 points for at least four.

The rules are simple heuristics, not a trained classifier or a reliable measure of true difficulty. A score of 3 or more routes to the strong tier.

## Setup on macOS

Run commands from the project directory:

```sh
cd /Users/nsuchitra/llm-cost-router/llm-cost-router
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Create `.env` only if it does not already exist, so existing credentials are not overwritten:

```sh
if [ ! -f .env ]; then cp .env.example .env; fi
open -e .env
```

Put your Groq key in the local `.env` file as `GROQ_API_KEY=...`. Do not put a real key in `.env.example`, source files, benchmark outputs, screenshots, or Git. `.env` and other `.env.*` files are ignored by Git; `.env.example` is the safe blank template.

Optional rate settings in `.env` are USD per million tokens:

```text
GROQ_CHEAP_INPUT_USD_PER_MILLION=
GROQ_CHEAP_OUTPUT_USD_PER_MILLION=
GROQ_STRONG_INPUT_USD_PER_MILLION=
GROQ_STRONG_OUTPUT_USD_PER_MILLION=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
```

Set both rates for a Groq model to enable its cost estimate. If either is unset, cost for that model is shown as unavailable; no default price is assumed. Blank model and Ollama URL values use the defaults listed above. Ollama must be running locally with the configured model available.

## Run the application

The interactive router calls Ollama for LOW questions and Groq for HIGH questions or fallback attempts:

```sh
python3 router.py
```

Enter a question for automatic routing, or use `force local: <question>`, `force cheap: <question>`, or `force strong: <question>`. These select Ollama tier 0, Groq tier 1, or Groq tier 2. The legacy `force groq:` prefix remains supported as an alias for tier 2. Local requests require a running Ollama server and do not use Groq quota; Groq requests use provider quota.

For tracking and CSV logging:

```sh
python3 tracker.py
```

It uses the same interactive syntax and writes new records to `routing_log_groq.csv`, separate from the historical `routing_log.csv`.

## Streamlit dashboard

Install the project dependencies (the requirements include Streamlit), then start the dashboard from the project directory:

```sh
python3 -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opening or refreshing the dashboard reads request history from the SQLite `requests` table and makes no model or API calls. The Ask form sends one `POST /v1/route` to `API_URL` (default `http://localhost:8000`) using `ROUTER_API_KEY` in the `X-API-Key` header. In Docker Compose, the dashboard API URL defaults to `http://api:8000`.

The dashboard shows tier shares, fallback and override counts, latency percentiles, estimated cost, and counterfactual strong-only cost. Requests can be filtered by tier, fallback, override, and stored question/model/reason text. Question text is only available if `LOG_PROMPTS=true`. A cost is labeled estimated; missing rates or token counts yield **Cost unavailable**. The **Replay mode** toggle runs the deterministic fake provider inside the dashboard process, writes the replay request to SQLite, and does not require Wi-Fi, an API key, or a model server. The regular Ask path uses the configured API and provider routing.

## HTTP API

Install the pinned dependencies, set a non-empty `ROUTER_API_KEY` in `.env`, then start the API:

```sh
uvicorn app.api:app --reload
```

All routes except `/health` require the same key in the `X-API-Key` header. `POST /v1/route` accepts a question (1–4000 characters), an optional `force_model` (`local`, `cheap`, or `strong`), and an optional threshold. `GET /v1/requests` reads request history from SQLite; `GET /v1/config` returns configured tier models and token rates. A per-key in-memory limit defaults to 60 requests per minute and can be changed with `API_RATE_LIMIT_PER_MINUTE`. `/health` checks Ollama HTTP reachability and whether a Groq key is configured; it does not call a model.

`ROUTER_ENABLED_TIERS` is a comma-separated list of tier IDs and defaults to `0,1,2` for local development. For a cloud service without access to local Ollama, set it to `1,2`; low-complexity Automatic requests then use Cheap, and Strong failures can fall back to Cheap. Set `ROUTER_DEPLOYMENT_PROFILE` to `local` or `cloud` for health diagnostics. The profile is informational; `ROUTER_ENABLED_TIERS` controls routing.

### Docker Compose

Copy `.env.example` to `.env` and fill in the credentials you intend to use, then run `docker compose up --build`. Compose starts the API on port 8000 and the Streamlit dashboard on port 8501. Both services share the `router_data` volume for the SQLite database. Compose sets `OLLAMA_BASE_URL` to `http://host.docker.internal:11434` unless you override it. On Docker Desktop for Mac, that hostname reaches services on the host. On Linux, Compose maps it to the host gateway; configure host Ollama to listen on an interface reachable from Docker (commonly `OLLAMA_HOST=0.0.0.0:11434`) and allow port 11434 through the host firewall. Keep Ollama access limited to trusted networks.

## Tests

Unit tests mock API calls and do not consume Groq credits:

```sh
python3 -m unittest discover -s tests -v
python3 -m py_compile *.py tests/*.py
```

The tests cover tier routing, manual overrides, fallback order, Ollama response parsing, authentication failures, cost estimation, numeric checking, and incomplete cache pairs. Provider calls are mocked.

## Dataset building and labelling

The dataset builder requires the `datasets` package from `requirements.txt`. It downloads source datasets only when explicitly run; importing the script, starting the app, and running tests do not download data or call a model.

```sh
python scripts/build_datasets.py
```

The fixed seed is `506`. The script writes:

- `data/train_pool.jsonl`: 598 items for manual labelling: 200 sampled GSM8K train questions, 200 sampled ARC-Challenge train questions, and 198 generated simple tasks after removing two exact synthetic overlaps with the frozen test set. The builder removes those overlaps without adding replacement questions.
- `data/dev.jsonl`: the existing 60 `benchmark_questions.json` records, preserving each source question and answer content.
- `data/test.jsonl`: 300 frozen items: 100 GSM8K test questions not found in dev, 100 ARC-Challenge multiple-choice questions with letter answers, and 100 generated simple tasks.
- `data/MANIFEST.json`: the seed, source/composition metadata, item counts, and SHA-256 digest for each JSONL file.

The builder removes exact normalized question overlaps from the generated training pool and refuses to replace `data/test.jsonl` when it already exists. To intentionally regenerate the frozen test set, run `python scripts/build_datasets.py --force-test`. Save or compare the old manifest before doing so. Source selection and generated records are deterministic for the fixed seed and the downloaded source dataset contents; the manifest hashes let you verify the exact files produced.

Simple examples cover date, email, and number extraction; JSON-list and uppercase reformatting; and classification into the fixed labels `billing`, `technical`, and `account`. These synthetic simple tasks are easier than real-world queries, so their results may not generalise to real user queries.

### Training labels and classifier experiment

`data/train_labels.jsonl` contains 598 checker-derived local labels keyed by stable IDs; `local_ok` is computed separately from provider success using the dataset's checker. The current sidecar contains 268 correct labels (44.82%). It preserves raw answers and does not rewrite the frozen test data. An older audit reported 270/600 before the final overlap and checker remediation; use current artifacts and the linked audit for the final counts.

The exploratory model is a scikit-learn pipeline with word TF-IDF n-grams and Logistic Regression. Random five-fold CV accuracy was 83.44%, but group-aware and source-held-out diagnostics exposed substantial generalization limitations. On the 60-question math-only dev set it predicted every item local-incorrect at the 0.5 threshold. It therefore has no call-share advantage over always strong on that set. **The classifier remains an offline research artifact and is not used by the live router.** Do not interpret its CV score as production routing performance.

Never tune on `data/test.jsonl`. Classifier features exclude ground-truth answers, raw model answers, IDs, correctness explanations, and cache metadata. See the [classifier research](docs/LOCAL_CLASSIFIER_RESEARCH.md) and [generalization audit](docs/CLASSIFIER_GENERALIZATION_AUDIT.md) before interpreting any classifier metrics.

## Completed test pilot

The deterministic pilot samples 96 questions from the frozen 300-question test set: 20 math, 20 MCQ, 20 extract, 20 reformat, and all 16 classify items. The same IDs were evaluated for all three tiers using each row's checker and the raw cached answer.

| Tier | Correct | Accuracy |
|---|---:|---:|
| Local | 42/96 | 43.75% |
| Cheap | 81/96 | 84.38% |
| Strong | 84/96 | 87.50% |

These are **pilot results, not full-test results**. The exact paired cheap-versus-strong McNemar test had p=0.25 (0 cheap-only wins, 3 strong-only wins), so the pilot does not show a statistically significant accuracy difference between those tiers. Local-versus-cheap and local-versus-strong comparisons favored the hosted tier within this pilot, but the sample does not establish generalization. Full task-wise results, confidence intervals, IDs, and latency summaries are in [PILOT_CACHE_EVALUATION.md](docs/PILOT_CACHE_EVALUATION.md) and [PILOT_STATISTICAL_ANALYSIS.md](docs/PILOT_STATISTICAL_ANALYSIS.md).

No verified dollar savings are available: checked-in Groq rates are blank, and the pilot reports correctness and observed latency rather than billed cost. The router and classifier are research prototypes, not production-ready systems.

## Historical benchmark tools

The commands in this section refer to the earlier `benchmark_questions.json` workflow. They are separate from the completed 96-question frozen-test pilot above.

`evaluate.py` only reads saved files. It does not call a model:

```sh
python3 evaluate.py
```

It reads `answers_cache_groq.json`, evaluates only questions with both model answers cached, names any missing question indices, and writes timestamped Groq result files. All compared policies use the same complete cached question pairs. The `Oracle (theoretical upper bound)` chooses between the two cached answers using knowledge of whether the cheap answer was correct; it is not a deployable policy because a live router does not know answer correctness in advance.

The existing 120-entry `answers_cache.json` contains historical local/strong model answers, not Groq pairs. The Groq evaluator does not treat those answers as current Groq results.

`run_benchmark.py` defaults to five evenly spaced questions from the 60-question dataset and runs all three strategies on that same sample. That is 15 planned requests (five per policy), plus up to ten fallback attempts in the router policy (two per question). The router policy now includes local Ollama tier 0. It reports answer success/failures, model selections, provider-reported input/output tokens, latency, and numeric-checker correctness. Paired accuracy compares only questions answered successfully by all three strategies. The checker compares extracted numbers, so it is not a general semantic-quality measurement.

Benchmark cost sums each successful API attempt using its provider-reported input/output tokens and configured model rates. Successful fallback calls are included as separate attempts. If pricing or token usage is unavailable, or a failed attempt has unknown billing, the total cost and savings percentage are **unavailable**; benchmark cost does not fall back to character-based token estimates. Rates must be verified against current [official Groq pricing](https://groq.com/pricing/) and your account billing. That public pricing page redirected to the site homepage in this environment, so no model prices or savings are assumed.

```sh
python3 run_benchmark.py
```

The explicit `--full` option runs the full dataset (180 planned calls and up to 120 additional router fallback attempts). Do not use it without approval.

## Answer caches

Import the existing local/Groq answers for the 60 development questions without making any provider calls:

```sh
python scripts/import_old_cache.py
```

This reads `answers_cache.json` without modifying it and creates or merges `cache/answers_dev.json`. Its `entries` mapping uses the stable development dataset ID. Each entry keeps the question and original numeric cache ID in `question_id` and `source_id`; its `answers` mapping uses tier names such as `local` and `strong`. Imported answer records keep the original answer and available metadata (currently `seconds`) and add `tier` and `imported: true`. The importer reports imported, already-present, missing, and unmatched records. Re-running it is idempotent; a different successful answer for the same question/tier is a conflict and is not replaced.

Generate a selected tier for a dataset split with `scripts/run_cache.py` (the root `run_cache.py` remains a CLI wrapper). Supported splits are `dev`, `train_pool`, and `test`; supported tiers are `local`, `cheap`, and `strong`:

```sh
python scripts/run_cache.py --split dev --tier local --limit 2
python scripts/run_cache.py --split dev --tier cheap --limit 2
python scripts/run_cache.py --split dev --tier strong --limit 2
python scripts/run_cache.py --split train_pool --tier local --limit 2
python scripts/run_cache.py --split train_pool --tier cheap --limit 2
python scripts/run_cache.py --split train_pool --tier strong --limit 2
python scripts/run_cache.py --split test --tier local --limit 2
python scripts/run_cache.py --split test --tier cheap --limit 2
python scripts/run_cache.py --split test --tier strong --limit 2
```

Use `--limit` for small smoke runs; it limits new missing answers attempted during that invocation, after successful cache entries have been skipped. Cache files are `cache/answers_<split>.json`. A successful record stores answer text, tier, configured model, provider, token usage/source, latency, and whether it was imported. Failed attempts are recorded separately under `failures`, so a failed request is distinguishable from a question/tier that has never been attempted. Each success or failure is atomically saved immediately. Rerunning the command resumes missing or failed entries and never replaces an existing successful answer.

**Live cache generation for `cheap` and `strong` sends requests to Groq, consumes API quota, and may incur costs.** Start with `--limit 1` or `--limit 2`, confirm the selected split, tier, configured model, and cache output, then decide whether to run a larger job. `local` uses the configured Ollama provider. Do not run a large generation job unintentionally.

### Historical five-question benchmark (2026-10-11)

Before the local tier was added, this five-question sample used dataset indices 0, 15, 30, 44, and 59. All three strategies answered all five; all 15 planned calls succeeded and the router did not fall back. Every policy therefore used the same five questions. Results from the then-current numeric checker:

| Strategy | Correct | Checker accuracy | Input tokens | Output tokens | Total elapsed time |
|---|---:|---:|---:|---:|---:|
| Always cheap | 5/5 | 100.0% | 642 | 1,379 | 4.668 s |
| Always strong | 4/5 | 80.0% | 642 | 1,268 | 5.804 s |
| Existing router | 5/5 | 100.0% | 642 | 1,152 | 5.402 s |

The router automatically selected the strong model for four questions and the cheap model for one; no fallback occurred. All three policies attempted the same five questions.

**Measured cost and cost-savings percentage: unavailable.** Both models' rate pairs were unset, and current official model-specific rates could not be verified from the accessible official pricing page. These five math questions and the numeric-only checker are a small, narrow sample; the results do not establish general model quality or an accuracy improvement. The run's detailed metadata is saved locally in the ignored file `benchmark_results_groq_live_20261011_230811_373988.json`.

**This cache command makes live API requests and is not needed for offline evaluation:**

```sh
python3 run_cache.py
```

`run_cache.py` makes up to two Groq calls for each of the 60 questions not already cached (up to 120 calls). It saves Groq results separately from the existing historical results.

## Important Python files

| File | Purpose |
|---|---|
| `router.py` | Rule-based score, tier selection, manual override, fallback, and interactive router. |
| `app/api.py`, `app/routing/`, `app/providers/`, `app/storage/` | Packaged HTTP API, routing policy, Ollama/Groq adapters, and SQLite request history. |
| `model_config.py` | Environment-backed model IDs and optional per-model rates. |
| `model_adapter.py` | Ollama and Groq API requests, response parsing, and provider usage/latency metadata. |
| `checker.py` | Dataset-selected numeric, letter, exact-output, and classification checkers. |
| `tracker.py` | Token-source and cost estimates, interactive tracker, and routing CSV logging. |
| `scripts/run_cache.py`, `scripts/cache_store.py` | Manifest-aware, resumable split/tier cache collection and atomic cache persistence. |
| `scripts/build_pilot_manifest.py`, `scripts/evaluate_split_cache.py`, `scripts/analyze_pilot_results.py` | Deterministic pilot selection, offline cache evaluation, and pilot statistical analysis. |
| `scripts/build_training_labels.py`, `scripts/train_local_classifier.py` | Offline checker-based labels and exploratory TF-IDF/Logistic Regression experiment. |
| `evaluate.py` | Historical evaluation of complete pairs in the old Groq cache format. |
| `run_benchmark.py` | Explicit live comparison of always-cheap, always-strong, and rule-router policies. |
| `streamlit_app.py` | Streamlit dashboard; only a submitted question invokes the existing tracker/router. |
| `dashboard_data.py` | No-network helpers for loading logs, dashboard summaries, and at-most-once submission handling. |
| `tests/test_core.py` | Network-mocked unit tests. |
| `tests/test_dashboard.py` | Offline dashboard data and submission tests. |
| `evaluate_hybrid.py`, `test_classifier.py`, `train_classifier.py` | Historical classifier experiments using earlier labels/cache formats. |

## Limitations and verification status

- `benchmark_questions.json` contains 60 questions with numeric reference answers. The existing Groq answer cache is separate from the historical 120-entry local/strong cache. Any live benchmark result reports its actual sample size and does not imply that all 60 questions were run.
- Local Ollama model cost is zero. Groq costs remain unavailable unless both token rates for the selected model are configured; no default Groq price is assumed.
- `checker.is_correct()` uses numeric extraction and numeric equality. It is not a general semantic grader and can misjudge explanations or answers containing unrelated numbers.
- Token counts are provider-reported when present; otherwise the tracker estimates each side as roughly characters divided by four. Cost is an estimate based on those counts and user-configured rates.
- The “always strong” cost comparison in single-request tracking reuses that request's token counts and rates; it is a counterfactual estimate, not a separately measured strong-model call.
- The rules may route questions incorrectly. No routing accuracy improvement is claimed.
- The current TF-IDF classifier is exploratory, has not been validated for live routing, and is not used by the router. The older classifier artifacts remain historical.
- The completed frozen-test pilot covers 96 of 300 questions; its pilot accuracies must not be presented as full-test results. The detailed report documents task-level uncertainty and source/template generalization risks.
- Unit tests verify mocked paths, not live Groq connectivity, account quota, current model availability, or billing.
- Failed requests may not return token usage even if a timeout leaves provider-side billing uncertain. The benchmark records failed attempts and refuses to claim a complete total cost in that case.
- A Groq key was found in the previous `.env.example` contents and has been removed from that template. Treat the exposed credential as compromised: revoke/rotate it with Groq before using the project. Do not copy it from old terminal output or history.

For a short, safe-to-follow live demonstration plan and guide Q&A, see [`docs/guide_demo.md`](docs/guide_demo.md).
