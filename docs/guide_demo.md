# LLM Cost Router dashboard demo

## 3–5 minute walkthrough

### 1. Open the dashboard

From the project directory:

```sh
cd /Users/nsuchitra/llm-cost-router/llm-cost-router
source .venv/bin/activate
streamlit run streamlit_app.py
```

Opening the page, refreshing it, and navigating the dashboard do not call a model. LOW questions use local Ollama; HIGH questions or fallback attempts use Groq and require a valid `GROQ_API_KEY`.

### 2. Explain the overview (30–45 seconds)

The overview reads `routing_log_groq.csv` for recorded query count, selected local/cheap/strong tier, average request latency, and estimated cost. If the log does not exist, recorded queries remain 0. `model_metrics_groq.jsonl` records successful calls with provider names and latency; those records do not contain question text or request IDs and are not user-query counts. Local model cost is zero; Groq cost remains unavailable until valid input and output rates are configured.

Performance charts use routing-history rows where available. If there is no routing CSV but API metrics exist, charts are labeled as successful API calls, not routed queries. Routing history does not attempt to reconstruct questions from API metrics.

### 3. Ask a simple question (30–45 seconds)

Enter:

```text
What is 2 + 2?
```

Click **Ask Router** once. The existing router calculates the score and chooses a model. The dashboard shows the answer, complexity, score, selected model ID, routing reason, latency, provider token usage, cost and fallback status. A LOW question calls local Ollama; a HIGH question calls Groq and may fall back through Groq tier 1 to local Ollama. No manual model choice is added to the dashboard.

### 4. Ask a more complex question (45–60 seconds)

Enter:

```text
Prove that the sum of the first n positive integers is n(n+1)/2 using mathematical induction.
```

Click **Ask Router**. Compare the score and selected tier with the simple question. Explain that this is the existing rule-based heuristic, not an ML classifier and not a guarantee of actual question difficulty.

### 5. Compare history and metrics (30–45 seconds)

Show how each successful dashboard or terminal-tracker query appears in routing history because both use the same `routing_log_groq.csv`. The charts summarize saved request metrics. The CSV is re-read after a successful question so the new row is visible immediately.

### 6. Explain benchmark status (20–30 seconds)

The benchmark panel says **Not yet verified** because current Groq results have not been validated against all 60 benchmark questions. The five-question sample shown in the README predates the local tier and is not a full benchmark. Do not run `run_cache.py` during this short demo; it can make up to 120 live API calls. `run_benchmark.py` defaults to 15 planned requests plus up to ten fallback attempts, so do not run it during the demo unless that usage is approved.

## Architecture in brief

```text
Dashboard form submission
        |
        v
tracker.track_question(display=False) -- one call
        |
        v
router.route_question() -- existing scoring, tier selection and fallback
        |
        v
model_adapter -> Ollama or Groq -> answer and provider token/latency data
        |
        +--> tracker writes routing_log_groq.csv
        +--> adapter appends model_metrics_groq.jsonl
        |
        v
Dashboard displays result and refreshes saved history
```

The dashboard does not implement its own routing strategy and does not invoke both tracker and router separately. Streamlit stores the handled form submission ID in session state to prevent a rerun from repeating that same request.

## Likely guide questions

**Does opening or refreshing the dashboard use the API?**
No. The page reads saved CSV/JSONL data and configuration. A model request happens only after **Ask Router** is clicked with a non-empty question.

**How does the dashboard choose a model?**
It delegates to the existing `router.route_question()`, through the existing tracking entry point. The rule-based complexity score below 3 selects local tier 0; score 3 or more selects strong tier 2. Tier 1 is the first fallback for tier 2. Thresholds and keywords were not changed for the dashboard.

**Does it send a question twice to track it?**
No. `tracker.track_question()` routes once and then logs that returned result. A recoverable tier 2 failure can make up to two fallback calls (tier 1, then local tier 0).

**Why is cost unavailable?**
Local Ollama cost is zero. Groq cost is unavailable until valid rates are configured. For Groq calls, the amount is an estimate based on logged token counts and is not the provider invoice.

**Why is the benchmark not verified?**
The current Groq cache does not yet contain a validated set of pairs for all 60 questions. The old cache belongs to an earlier local-model experiment and is kept separate.

**What if the strong model hits a rate limit?**
The existing router can retry with a cheaper tier for recoverable errors such as HTTP 429, HTTP 5xx, timeout, or connection failure. Missing/invalid authentication is not retried. The result indicates whether fallback occurred.

## Keep the demo safe

- Do not show the `.env` file or API key in screenshots or screen sharing.
- The question examples each make a live request and may consume API quota.
- Run `python3 -m unittest discover -s tests -v` to demonstrate offline tests; requests are mocked.
- Avoid `run_cache.py` and `run_benchmark.py` during the demo unless their API usage has been approved.
- State clearly: routing is implemented; a 5-question numeric-checker sample was measured but full-benchmark accuracy is not verified; costs are unavailable without verified rates; no accuracy or savings improvement is claimed.
