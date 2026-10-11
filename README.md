# LLM Cost Router

## Problem and motivation

Calling a powerful language model for every question can spend more API quota and money than necessary. Many questions are simple enough for a smaller model, while questions needing several reasoning steps may benefit from a stronger model. This project demonstrates a rule-based router that chooses between two Groq-hosted models and records what happened.

The project is an educational prototype, not a production routing service. It estimates cost from configured rates and token counts; it does not report Groq's billed amount.

## Architecture and workflow

1. `router.calculate_complexity()` scores the question using existing word-count, keyword, phrase, math-symbol, and number-count rules.
2. `router.route_question()` uses the existing threshold: scores below 3 choose the cheap model; scores of 3 or more choose the strong model. The routing rules and threshold have not been changed for this demonstration.
3. `model_adapter.ask_groq_model()` sends the selected model ID and question to Groq's chat-completions API. The adapter reads the API key from `GROQ_API_KEY`, returns the answer, provider-reported token usage, and latency.
4. If the strong-model request fails with a recoverable error (HTTP 429, HTTP 5xx, timeout, or connection error), the router tries the cheap Groq model. Authentication/configuration errors are not retried with the same credential. The cheap-model failure is reported.
5. `tracker.track_question()` can log the selected model, fallback status, token sources, latency, and estimated cost to a new Groq-specific CSV file.

The cheap model is `openai/gpt-oss-20b`; the strong model is `openai/gpt-oss-120b`. IDs can be changed through environment variables. The router decision logic is separate from API calls.

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
```

Set both rates for a model to enable its cost estimate. If either is unset, cost for that model is shown as unavailable; no default price is assumed. `GROQ_CHEAP_MODEL` and `GROQ_STRONG_MODEL` optionally override the model IDs.

## Run the application

The interactive router makes a live Groq request for each submitted question:

```sh
python3 router.py
```

Enter a question for automatic routing, or use `force cheap: <question>` / `force strong: <question>`. The legacy prefixes `force local:` and `force groq:` remain supported (`local` means the cheap Groq model, not an installed local model). Each live request uses provider quota.

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

Opening or refreshing the dashboard does not call a model. The only dashboard model request is made after entering a question and clicking **Ask Router**. It calls the existing `tracker.track_question()` once, which invokes the existing router once and records the successful response in `routing_log_groq.csv`. The existing recoverable strong-to-cheap fallback remains part of that one routing attempt. Re-running the page does not resubmit a handled form event.

The dashboard reads `routing_log_groq.csv` for user-submitted routed-query history. It uses `model_metrics_groq.jsonl` for successful API-call counts and latency only when routing history is empty. API-call counts are explicitly labeled separately from queries; the metrics file has no question text or request IDs, so it cannot supply routing history. If no pricing rates are configured, the dashboard displays **Cost unavailable**. Its benchmark panel remains **Not yet verified** until the current Groq models have been evaluated on the complete benchmark; historical local-model results are not shown as current.

## Tests

Unit tests mock API calls and do not consume Groq credits:

```sh
python3 -m unittest discover -s tests -v
python3 -m py_compile *.py tests/*.py
```

The tests cover routing, manual overrides, recoverable and authentication failures, missing credentials, cost estimation, numeric checking, and incomplete cache pairs.

## Benchmark without unnecessary API calls

`evaluate.py` only reads saved files. It does not call a model:

```sh
python3 evaluate.py
```

It reads `answers_cache_groq.json`, evaluates only questions with both model answers cached, names any missing question indices, and writes timestamped Groq result files. All compared policies use the same complete cached question pairs. The `Oracle (theoretical upper bound)` chooses between the two cached answers using knowledge of whether the cheap answer was correct; it is not a deployable policy because a live router does not know answer correctness in advance.

The existing 120-entry `answers_cache.json` contains historical local/strong model answers, not Groq pairs. The Groq evaluator does not treat those answers as current Groq results.

`run_benchmark.py` defaults to five evenly spaced questions from the 60-question dataset and runs all three strategies on that same sample. That is 15 planned requests (five per policy), plus at most five cheap-model fallback requests from the router. It reports answer success/failures, model selections, provider-reported input/output tokens, latency, and numeric-checker correctness. Paired accuracy compares only questions answered successfully by all three strategies. The checker compares extracted numbers, so it is not a general semantic-quality measurement.

Benchmark cost sums each successful API attempt using its provider-reported input/output tokens and configured model rates. Successful fallback calls are included as separate attempts. If pricing or token usage is unavailable, or a failed attempt has unknown billing, the total cost and savings percentage are **unavailable**; benchmark cost does not fall back to character-based token estimates. Rates must be verified against current [official Groq pricing](https://groq.com/pricing/) and your account billing. That public pricing page redirected to the site homepage in this environment, so no model prices or savings are assumed.

```sh
python3 run_benchmark.py
```

The explicit `--full` option runs the full dataset (180 planned calls and up to 60 additional fallback calls). Do not use it without approval.

### Latest measured sample

On 2026-10-11, the five-question sample used dataset indices 0, 15, 30, 44, and 59. All three strategies answered all five; all 15 planned calls succeeded and the router did not fall back. Every policy therefore used the same five questions. Results from the existing numeric checker:

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
| `model_config.py` | Environment-backed model IDs and optional per-model rates. |
| `model_adapter.py` | Groq API request, response parsing, and provider usage/latency metadata. |
| `checker.py` | Numeric answer extraction and equality check for benchmark scoring. |
| `tracker.py` | Token-source and cost estimates, interactive tracker, and Groq-specific CSV logging. |
| `run_cache.py` | Explicit live collection of cheap/strong answers into the separate Groq cache. |
| `evaluate.py` | Offline evaluation of complete pairs in the Groq cache. |
| `run_benchmark.py` | Explicit live comparison of always-cheap, always-strong, and rule-router policies. |
| `streamlit_app.py` | Streamlit dashboard; only a submitted question invokes the existing tracker/router. |
| `dashboard_data.py` | No-network helpers for loading logs, dashboard summaries, and at-most-once submission handling. |
| `tests/test_core.py` | Network-mocked unit tests. |
| `tests/test_dashboard.py` | Offline dashboard data and submission tests. |
| `evaluate_hybrid.py`, `test_classifier.py`, `train_classifier.py` | Historical classifier experiments based on existing labels/cache, not the current Groq router. |

## Limitations and verification status

- `benchmark_questions.json` contains 60 questions with numeric reference answers. The existing Groq answer cache is separate from the historical 120-entry local/strong cache. Any live benchmark result reports its actual sample size and does not imply that all 60 questions were run.
- Neither Groq model currently has both input and output rates configured, and official model-specific rates could not be verified in this environment, so the project currently reports costs as unavailable.
- `checker.is_correct()` uses numeric extraction and numeric equality. It is not a general semantic grader and can misjudge explanations or answers containing unrelated numbers.
- Token counts are provider-reported when present; otherwise the tracker estimates each side as roughly characters divided by four. Cost is an estimate based on those counts and user-configured rates.
- The “always strong” cost comparison in single-request tracking reuses that request's token counts and rates; it is a counterfactual estimate, not a separately measured strong-model call.
- The rules may route questions incorrectly. No routing accuracy improvement is claimed.
- The old classifier data/results are retained for historical reproducibility and do not demonstrate classifier performance on GPT-OSS 20B.
- Unit tests verify mocked paths, not live Groq connectivity, account quota, current model availability, or billing.
- Failed requests may not return token usage even if a timeout leaves provider-side billing uncertain. The benchmark records failed attempts and refuses to claim a complete total cost in that case.
- A Groq key was found in the previous `.env.example` contents and has been removed from that template. Treat the exposed credential as compromised: revoke/rotate it with Groq before using the project. Do not copy it from old terminal output or history.

For a short, safe-to-follow live demonstration plan and guide Q&A, see [`docs/guide_demo.md`](docs/guide_demo.md).
