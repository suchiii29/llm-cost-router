# LLM Cost Router status

## Current design

- `model_config.get_model_config()` reads model IDs, Ollama base URL, and optional price rates from the environment (including `.env`). Tiers are 0: Ollama `qwen2.5:1.5b`, 1: Groq `openai/gpt-oss-20b`, and 2: Groq `openai/gpt-oss-120b`.
- `model_adapter.ask_ollama_model()` and `ask_groq_model()` return a common response containing answer, model ID, provider, usage and latency. Ollama uses the local `/api/generate` endpoint; Groq reads `GROQ_API_KEY`.
- `router.calculate_complexity()` and `router.route_question()` keep rule scoring and provider calls separate. The score threshold and keyword rules are unchanged: LOW selects tier 0 and HIGH selects tier 2. Recoverable failures descend through cheaper tiers; authentication/configuration errors stop the chain.
- `tracker.estimate_cost()` uses provider-reported token counts when available and otherwise uses a character-count approximation. Costs are estimated from configured per-model rates; they are reported as unavailable until the relevant input and output rates are configured.

## Requirements status

| Requirement | Status | Evidence |
|---|---|---|
| Web UI | Implemented | `streamlit_app.py` provides the dashboard and invokes the router only for a submitted question. |
| Complexity score | Partial | `router.calculate_complexity()` scores using existing rules; it is not an ML-derived difficulty estimate. |
| Query decomposition | Planned | No subtask-splitting implementation exists. |
| Model routing | Implemented | `router.route_question()` maps LOW to tier 0 Ollama and HIGH to tier 2 Groq. Tier 1 Groq is currently a fallback tier; this is a rule-based policy, not general capability/price optimization. |
| Subtask aggregation | Planned | No subtask outputs are created or combined. |
| Per-request display | Partial | `router.route_question()` displays tier/model/reasons; `tracker.track_question()` displays provider usage, estimated token counts, estimated costs, and latency. Local cost is zero; Groq cost is unavailable until rates are set. |
| Manual override | Implemented | `router.route_question()` accepts local/cheap/strong and legacy groq aliases; both interactive loops parse the corresponding force prefixes. |
| Decision logging and strongest baseline | Partial | `tracker.track_question()` writes a new, explicit-schema `routing_log_groq.csv` row including fallback details. Baseline cost remains an estimate based on the response token counts, not a separately measured strong-model response. The old `routing_log.csv` is historical and unchanged. |
| API failure handling | Partial | Groq and Ollama adapters distinguish recoverable HTTP, timeout and connection errors from non-recoverable errors. `router.route_question()` tries lower tiers after recoverable failures; there is no fallback below tier 0. Groq authentication errors stop without retry. |
| Local tier | Implemented | `model_adapter.ask_ollama_model()` calls Ollama with non-streaming generation; tracker and benchmark cost calculations assign local usage zero model cost. |
| Quality-based fallback | Planned | `checker.is_correct()` is an offline numeric checker, not a live answer-quality gate. |
| Classifier experiment | Partial | `train_classifier.main()` trains from `train_labels.json`, which records the historical Qwen local-model outcome. It is not trained on Groq 20B data and is not used in live routing. |

## Historical data and experiment compatibility

- `benchmark_questions.json`, `answers_cache.json`, `labels.json`, `train_labels.json`, `classifier.pkl`, `benchmark_results.json`, `routing_log.csv`, and `model_metrics.jsonl` are existing Qwen-era artifacts. They are not rewritten or relabeled as GPT-OSS results.
- New successful provider-call metadata is appended to `model_metrics_groq.jsonl`; prompt and answer text are not written to that metrics file.
- `legacy/output.txt` is a historical sample that mentions prior providers and models.
- New Groq cache collection writes model-ID-keyed records to `answers_cache_groq.json`; it does not overwrite the old cache.
- `evaluate.py` evaluates only complete pairs in the Groq cache, reports missing question indices, and writes timestamped labels/results that include model IDs and evaluated counts. Before new results are collected, it reports that there are no current Groq pairs.
- `evaluate_hybrid.py` and `test_classifier.py` remain historical classifier analyses over the old `local`/`strong` cache entries. Do not interpret their classifier results as predictions for the new Groq cheap model.
- `run_benchmark.py` defaults to five evenly spaced questions and runs all three policies on the same sample (15 planned calls and up to ten router fallback attempts). `--full` explicitly selects all 60 questions and may add up to 120 fallback attempts. It records each successful/failed attempt, usage, per-attempt cost and result correctness in timestamped result files.
- `legacy/benchmark_local.py`, `legacy/test_ollama.py`, and `legacy/list_models.py` are retained historical Groq scripts; their names do not describe the current Ollama provider.
- `run_train_labels.py` writes new cheap-Groq labels to `train_labels_groq.json`; the old `train_labels.json` remains unchanged.
- `legacy/benchmark_results.json` reports 8 historical questions, not the full 60-question benchmark. It is not a current evaluation.

## Cost and usage interpretation

- Groq's returned prompt/completion token counts are provider-reported usage; the adapter stores them separately from estimated costs.
- When both provider token counts are available, `tracker.estimate_cost()` uses them. Missing counts are filled with `max(1, characters // 4)` and explicitly marked estimated.
- Cost is **not provider-billed cost**. It is calculated from token counts and optional rates in `.env`. No rate defaults are assumed; without both input/output rates for a model, cost is unavailable.
- The live benchmark uses provider-reported usage only and includes successful fallback attempts as additional billable usage. If a failed attempt has no provider usage, total benchmark cost is unavailable because any provider-side charge cannot be established from the response.
- On 2026-10-11, the approved five-question sample answered all five questions under each policy with no failed attempts or fallback. The numeric checker scored 5/5 cheap, 4/5 strong, and 5/5 router. Costs and cost savings remained unavailable because both model-rate pairs were unset; this small math-only sample does not verify full-benchmark results.
- The always-strong baseline is an estimate using the same request's input/output token counts. A real counterfactual strong-model response would need a separate strong call.
- Historical Qwen cost results used earlier hardcoded rates and should not be compared directly with the new configurable-rate results.

## Provider errors and fallback

- Missing credentials and HTTP 401/403 are not retried with another Groq model. Recoverable tier 2 failures try tier 1 then tier 0; recoverable tier 1 failures try tier 0. Local tier failures have no cheaper tier.
- HTTP 429, HTTP 5xx, timeouts, and connection errors are recoverable. Groq account-wide rate limits or outages may affect both hosted tiers, but tier 0 may still work if the local Ollama server is reachable.
- Invalid model IDs and malformed provider responses are non-recoverable and stop fallback.
- The API key is read from `GROQ_API_KEY`; do not put a real value in `.env.example` or source code. `.env` is ignored by Git.

## Reproduce

Run commands from the repository directory. Install/run Ollama locally and make the configured model available before using the local tier.

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

The unit tests mock all provider calls. LOW questions use local Ollama; HIGH questions and fallback attempts can make real Groq requests and may incur provider usage.

To populate the new cache (makes up to two API calls per uncached benchmark question), then evaluate only the saved responses:

```sh
python3 run_cache.py
python3 evaluate.py
```

`python3 run_benchmark.py` makes 15 planned requests for its five-question sample, plus up to ten router fallback attempts. `python3 run_benchmark.py --full` makes 180 planned calls across the full dataset, plus up to 120 fallback attempts; do not run that command without explicit approval. Groq attempts may incur provider usage. The historical classifier experiments use existing saved Qwen results when run with:

```sh
python3 evaluate_hybrid.py
python3 test_classifier.py
```
