# LLM Cost Router status

## Current design

- `model_config.get_model_config()` reads model IDs and optional price rates from the environment (including `.env`). Defaults are `openai/gpt-oss-20b` for cheap and `openai/gpt-oss-120b` for strong.
- `model_adapter.ask_groq_model()` is the shared Groq API adapter. It reads `GROQ_API_KEY`, sends the configured model ID, and returns the answer, actual response model ID, provider-reported usage, and request latency.
- `router.calculate_complexity()` and `router.route_question()` keep rule scoring and provider calls separate. The existing score threshold and keyword rules are unchanged. A recoverable strong-model error can try the cheap model. Authentication/configuration errors are not retryable using the same key.
- `tracker.estimate_cost()` uses provider-reported token counts when available and otherwise uses a character-count approximation. Costs are estimated from configured per-model rates; they are reported as unavailable until the relevant input and output rates are configured.

## Requirements status

| Requirement | Status | Evidence |
|---|---|---|
| Web UI | Planned | No web app is present; `router.main()` is a terminal interface. |
| Complexity score | Partial | `router.calculate_complexity()` scores using existing rules; it is not an ML-derived difficulty estimate. |
| Query decomposition | Planned | No subtask-splitting implementation exists. |
| Model routing | Partial | `router.route_question()` maps LOW to the configured cheap Groq model and HIGH to the configured strong Groq model. It is a two-tier rule, not general capability/price optimization. |
| Subtask aggregation | Planned | No subtask outputs are created or combined. |
| Per-request display | Partial | `router.route_question()` displays tier/model/reasons; `tracker.track_question()` displays provider usage, estimated token counts, estimated costs, and latency. Cost is unavailable until rates are set. |
| Manual override | Implemented | `router.route_question()` accepts cheap/strong and legacy local/groq aliases; the interactive loops accept `force cheap:` and `force strong:` plus the old command prefixes for compatibility. |
| Decision logging and strongest baseline | Partial | `tracker.track_question()` writes a new, explicit-schema `routing_log_groq.csv` row including fallback details. Baseline cost remains an estimate based on the response token counts, not a separately measured strong-model response. The old `routing_log.csv` is historical and unchanged. |
| API failure handling | Partial | `model_adapter.ask_groq_model()` distinguishes configuration, HTTP, timeout, connection, and malformed-response errors. `router.route_question()` retries recoverable strong-tier errors with the cheap model. An invalid/missing key, invalid model ID, provider-wide outage, and cheap-tier failure cannot be recovered by retrying the same provider; errors are surfaced. |
| Quality-based fallback | Planned | `checker.is_correct()` is an offline numeric checker, not a live answer-quality gate. |
| Classifier experiment | Partial | `train_classifier.main()` trains from `train_labels.json`, which records the historical Qwen local-model outcome. It is not trained on Groq 20B data and is not used in live routing. |

## Historical data and experiment compatibility

- `benchmark_questions.json`, `answers_cache.json`, `labels.json`, `train_labels.json`, `classifier.pkl`, `benchmark_results.json`, `routing_log.csv`, and `model_metrics.jsonl` are existing Qwen-era artifacts. They are not rewritten or relabeled as GPT-OSS results.
- New Groq API-call metadata is appended to `model_metrics_groq.jsonl`; prompt and answer text are not written to that metrics file.
- `output.txt` is a historical sample that mentions a prior local model and Gemini; neither is part of the current Groq architecture.
- New Groq cache collection writes model-ID-keyed records to `answers_cache_groq.json`; it does not overwrite the old cache.
- `evaluate.py` evaluates only complete pairs in the Groq cache, reports missing question indices, and writes timestamped labels/results that include model IDs and evaluated counts. Before new results are collected, it reports that there are no current Groq pairs.
- `evaluate_hybrid.py` and `test_classifier.py` remain historical classifier analyses over the old `local`/`strong` cache entries. Do not interpret their classifier results as predictions for the new Groq cheap model.
- `run_benchmark.py` defaults to five evenly spaced questions and runs all three policies on the same sample (15 planned API calls and up to five router fallback calls). `--full` explicitly selects all 60 questions. It records each successful/failed attempt, usage, per-attempt cost when available, and result correctness in a timestamped `benchmark_results_groq_live_*.json`; `evaluate.py` writes timestamped saved-cache results.
- The legacy-named `benchmark_local.py` and `test_ollama.py` have been repurposed to call the configured Groq models only. They make API calls when run directly.
- `run_train_labels.py` writes new cheap-Groq labels to `train_labels_groq.json`; the old `train_labels.json` remains unchanged.
- Existing saved `benchmark_results.json` reports 8 questions, not the full 60-question benchmark. It is not a current Groq evaluation.

## Cost and usage interpretation

- Groq's returned prompt/completion token counts are provider-reported usage; the adapter stores them separately from estimated costs.
- When both provider token counts are available, `tracker.estimate_cost()` uses them. Missing counts are filled with `max(1, characters // 4)` and explicitly marked estimated.
- Cost is **not provider-billed cost**. It is calculated from token counts and optional rates in `.env`. No rate defaults are assumed; without both input/output rates for a model, cost is unavailable.
- The live benchmark uses provider-reported usage only and includes successful fallback attempts as additional billable usage. If a failed attempt has no provider usage, total benchmark cost is unavailable because any provider-side charge cannot be established from the response.
- On 2026-10-11, the approved five-question sample answered all five questions under each policy with no failed attempts or fallback. The numeric checker scored 5/5 cheap, 4/5 strong, and 5/5 router. Costs and cost savings remained unavailable because both model-rate pairs were unset; this small math-only sample does not verify full-benchmark results.
- The always-strong baseline is an estimate using the same request's input/output token counts. A real counterfactual strong-model response would need a separate strong call.
- Historical Qwen cost results used earlier hardcoded rates and should not be compared directly with the new configurable-rate results.

## Groq API errors and fallback

- Missing credentials and HTTP 401/403 are not retried with another model: both models use the same Groq credential, so switching model IDs does not fix invalid authentication or authorization.
- HTTP 429, HTTP 5xx, timeouts, and connection errors from a strong-model call may try the cheap model. Because both are on the same provider/account, account-wide rate limits or outages may also affect the retry.
- Invalid model IDs and malformed provider responses are reported as non-recoverable errors. Failures in the cheap-model request are surfaced; there is no local-model fallback.
- The API key is read from `GROQ_API_KEY`; do not put a real value in `.env.example` or source code. `.env` is ignored by Git.

## Reproduce

Run commands from the repository directory. No model is installed locally.

```sh
cd /Users/nsuchitra/llm-cost-router/llm-cost-router
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If `.env` does not already exist, create it from the example without overwriting existing credentials:

```sh
if [ ! -f .env ]; then cp .env.example .env; fi
```

Edit `.env` locally to add `GROQ_API_KEY`. Optional per-model rates are USD per million input/output tokens; leave them empty to show costs as unavailable.

```sh
python3 -m unittest discover -s tests -v
python3 -m py_compile *.py tests/*.py
python3 router.py
# At the prompt: force cheap: What is 2+2?
# Or: force strong: Explain how this algorithm works.
python3 tracker.py
```

The unit tests mock all API calls. `router.py` and `tracker.py` make real Groq requests after a question is entered and therefore require a valid key and may incur provider usage.

To populate the new cache (makes up to two API calls per uncached benchmark question), then evaluate only the saved responses:

```sh
python3 run_cache.py
python3 evaluate.py
```

`python3 run_benchmark.py` makes 15 live requests for its five-question sample, plus up to five fallback calls. `python3 run_benchmark.py --full` makes 180 planned calls across the full dataset, plus up to 60 fallback calls; do not run that command without explicit approval. Both may incur usage. The historical classifier experiments use existing saved Qwen results when run with:

```sh
python3 evaluate_hybrid.py
python3 test_classifier.py
```
