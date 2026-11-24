"""SQLite-backed dashboard. Provider calls happen only on Ask submission."""

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

from app.config import get_settings
from dashboard_data import (
    api_url_from_env,
    filter_requests,
    format_cost,
    load_api_config,
    load_api_request_history,
    load_request_history,
    merge_request_histories,
    missing_cost_rate_settings,
    request_tier,
    served_tier_label,
    settings_with_api_tiers,
    submit_in_replay_mode,
    submit_to_api,
    summarize_requests,
    unavailable_cost_explanation,
)


st.set_page_config(
    page_title="LLM Cost Router",
    page_icon="↗",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; color: #172033; }
    [data-testid="stHeader"] { background: rgba(245, 247, 251, 0.92); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e5eaf2;
        padding: 1rem; border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _display_row(row: dict[str, Any], settings) -> dict[str, Any]:
    created = row.get("created_at")
    return {
        "Request": row.get("request_id"),
        "Created": (
            created.isoformat(timespec="seconds")
            if hasattr(created, "isoformat")
            else str(created or "").replace("T", " ")[:19]
        ),
        "Tier": request_tier(row, settings),
        "Routed to": row.get("routed_to"),
        "Actual model": row.get("actual_model") or (
            "No successful response" if row.get("request_status") == "failure"
            else row.get("model")
        ),
        "Score": row.get("score"),
        "Fallback": row.get("fallback"),
        "Override": row.get("overridden"),
        "Tokens": (
            f"{row.get('input_tokens')} in / {row.get('output_tokens')} out"
            if row.get("input_tokens") is not None
            and row.get("output_tokens") is not None
            else "Unavailable"
        ),
        "Token source": row.get("token_source"),
        "Latency (ms)": row.get("latency_ms"),
        "API cost": format_cost(row.get("cost_usd"), row.get("cost_status")),
        "Estimated strong baseline": format_cost(row.get("baseline_cost_usd")),
        "Estimated API savings": format_cost(
            float(row["baseline_cost_usd"]) - float(row["cost_usd"])
            if row.get("cost_usd") is not None and row.get("baseline_cost_usd") is not None
            else None
        ),
        "Cost status": row.get("cost_status") or "Unavailable",
    }


def _detail_panel(row: dict[str, Any], *, title: str = "Request details") -> None:
    st.subheader(title)
    left, right = st.columns(2)
    left.metric("Score", row.get("score", "Unavailable"))
    right.metric("Token source", row.get("token_source", "Unavailable"))
    st.markdown("**Routed to vs. served model**")
    served_tier = served_tier_label(row)
    st.write(
        f"{row.get('routed_to', 'unknown')} → {served_tier}, "
        f"{row.get('model', 'unknown')}"
    )
    st.markdown("**Reasons**")
    reasons = row.get("reasons") or []
    if reasons:
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.write("No strong complexity indicators detected.")
    if row.get("fallback"):
        st.warning(f"Fallback used: {row.get('fallback_reason') or 'reason unavailable'}")
    if row.get("answer"):
        st.markdown("**Answer**")
        st.markdown(row["answer"])
    if row.get("cost_usd") is None:
        st.caption(unavailable_cost_explanation(row))
    else:
        st.caption(f"Request cost: {format_cost(row['cost_usd'], row.get('cost_status'))}")


st.title("LLM Cost Router")
st.caption(
    "Live history uses the authenticated router API when configured; local Replay "
    "history uses SQLite. Page loads and filters do not call a model."
)

settings = get_settings()
history_error = None
history_available = True
api_key = os.getenv("ROUTER_API_KEY", "")
try:
    if api_key or os.getenv("API_URL"):
        if not api_key:
            raise ValueError(
                "Set ROUTER_API_KEY to load authenticated backend request history"
            )
        backend_config = load_api_config(api_url=api_url_from_env(), api_key=api_key)
        settings = settings_with_api_tiers(settings, backend_config)
        backend_rows = load_api_request_history(
            api_url=api_url_from_env(), api_key=api_key
        )
        try:
            replay_rows = load_request_history(settings)
        except (OSError, RuntimeError, ValueError):
            replay_rows = []
            st.warning("Could not read local Replay history from SQLite.")
        rows = merge_request_histories(backend_rows, replay_rows)
    else:
        rows = load_request_history(settings)
except (OSError, RuntimeError, ValueError) as error:
    history_available = False
    history_error = str(error)
    st.error(f"Request history and analytics are unavailable: {history_error}")
    rows = []
summary = summarize_requests(rows, settings) if history_available else None

st.header("Overview")
cards = st.columns(6)
cards[0].metric("Requests", summary["requests"] if summary else "Unavailable")
for column, tier in zip(cards[1:4], settings.tiers):
    if summary:
        share = summary["tier_shares"].get(tier.id, 0.0)
        count = summary["tier_counts"].get(tier.id, 0)
        column.metric(f"Tier {tier.id} share", f"{share:.0%}", delta=f"{count} requests")
    else:
        column.metric(f"Tier {tier.id} share", "Unavailable")
cards[4].metric("Fallbacks", summary["fallbacks"] if summary else "Unavailable")
cards[5].metric("Overrides", summary["overrides"] if summary else "Unavailable")

st.header("Cost Analytics")
cost_cards = st.columns(5)
cost_cards[0].metric("Actual API cost", format_cost(summary["measured_api_cost_usd"], "actual") if summary else "Unavailable")
cost_cards[1].metric("Input tokens", f"{summary['total_input_tokens']:,}" if summary else "Unavailable")
cost_cards[2].metric("Output tokens", f"{summary['total_output_tokens']:,}" if summary else "Unavailable")
cost_cards[3].metric("Average cost per request", format_cost(summary["average_api_cost_usd"], "actual") if summary else "Unavailable")
cost_cards[4].metric("Unavailable cost records", summary["unavailable_cost_requests"] if summary else "Unavailable")
cost_comparison = st.columns(5)
cost_comparison[0].metric("Estimated API cost", format_cost(summary["estimated_api_cost_usd"], "estimated") if summary else "Unavailable")
cost_comparison[1].metric("Local Ollama API cost", format_cost(summary["local_api_cost_usd"], "local") if summary else "Unavailable")
cost_comparison[2].metric("Estimated always-strong baseline", format_cost(summary["counterfactual_strong_cost_usd"], "estimated") if summary else "Unavailable")
cost_comparison[3].metric("Estimated API savings", format_cost(summary["estimated_savings_usd"], "estimated") if summary else "Unavailable")
cost_comparison[4].metric("Savings percentage", f"{summary['savings_percentage']:.1f}%" if summary and summary["savings_percentage"] is not None else "Unavailable")
if summary and summary["cost_by_model"]:
    model_col, tier_col = st.columns(2)
    model_col.markdown("**Measured API cost by model**")
    model_col.dataframe(pd.DataFrame([{"Model": key, "API cost": format_cost(value, "actual")} for key, value in summary["cost_by_model"].items()]), hide_index=True, use_container_width=True)
    tier_col.markdown("**Measured API cost by tier**")
    tier_col.dataframe(pd.DataFrame([{"Tier": key, "API cost": format_cost(value, "actual")} for key, value in summary["cost_by_tier"].items()]), hide_index=True, use_container_width=True)
latency_cards = st.columns(2)
for column, key, label in zip(
    latency_cards,
    ("p50_latency_ms", "p95_latency_ms"),
    ("p50 latency", "p95 latency"),
):
    value = summary[key] if summary else None
    column.metric(label, f"{value:.0f} ms" if value is not None else "Unavailable")
if summary and rows and (
    summary["measured_api_cost_usd"] is None
    or summary["comparable_requests"] < summary["requests"]
):
    missing_rates = missing_cost_rate_settings(settings)
    if missing_rates:
        st.info(
            "Some stored costs are unavailable. Configure these backend rates "
            "(USD per 1 million tokens) for new requests: "
            + ", ".join(f"`{name}`" for name in missing_rates)
            + ". Historical requests with missing rates remain unavailable."
        )
    elif any(
        row.get("input_tokens") is None or row.get("output_tokens") is None
        for row in rows
    ):
        st.info("Some stored costs are unavailable because token counts are missing.")
    else:
        st.info("Some historical requests were recorded without cost estimates.")
else:
    st.caption(
        "Actual API cost uses configured rates and provider reported usage. The always-strong baseline is a token-based hypothetical using the same token counts; it is not a second call or proof of identical answers. Local API inference is $0, excluding hardware and electricity."
    )

st.header("Ask a question")
replay_mode = st.toggle(
    "Replay mode (deterministic fake provider; no network or keys)",
    key="replay_mode",
    value=False,
)
override_labels = {
    "Automatic": None,
    "Local": "local",
    "Cheap": "cheap",
    "Strong": "strong",
}
with st.form("ask_router_form", clear_on_submit=False):
    question = st.text_area(
        "Question",
        max_chars=4000,
        height=120,
        placeholder="Enter a question to route...",
    )
    controls = st.columns(2)
    override_label = controls[0].selectbox("Model override", list(override_labels))
    threshold = controls[1].slider(
        "Complexity threshold",
        min_value=0,
        max_value=12,
        value=settings.complexity_threshold,
    )
    submitted = st.form_submit_button("Ask Router", type="primary")

if submitted:
    if not question.strip():
        st.error("Enter a question before submitting.")
    else:
        try:
            with st.spinner(
                "Generating a deterministic replay..."
                if replay_mode
                else "Sending the request to the router API..."
            ):
                if replay_mode:
                    result = submit_in_replay_mode(
                        question.strip(),
                        settings=settings,
                        force_model=override_labels[override_label],
                        threshold=threshold,
                    )
                else:
                    result = submit_to_api(
                        question.strip(),
                        api_url=api_url_from_env(),
                        api_key=os.getenv("ROUTER_API_KEY", ""),
                        force_model=override_labels[override_label],
                        threshold=threshold,
                    )
            result["request_mode"] = "Replay" if replay_mode else "Live API"
            st.session_state["last_submission"] = result
            st.session_state["last_submission_question"] = question.strip()
            st.rerun()
        except (RuntimeError, ValueError, requests.RequestException) as error:
            st.error(str(error))

last_submission = st.session_state.get("last_submission")
if last_submission:
    with st.container(border=True):
        st.caption(f"{last_submission.get('request_mode', 'Request')} request")
        st.markdown(
            f"**Latest answer** · request {last_submission.get('request_id', 'unknown')}"
        )
        st.markdown(last_submission.get("answer", ""))
        _detail_panel(last_submission, title="Latest request details")

st.header("Request history")
filters = st.columns(4)
tier_options = ["All tiers"] + [
    f"tier {tier.id} · {tier.name}" for tier in settings.tiers
]
tier_filter = filters[0].selectbox("Tier", tier_options)
fallback_filter = filters[1].selectbox("Fallback", ["All", "Yes", "No"])
override_filter = filters[2].selectbox("Override", ["All", "Yes", "No"])
search_filter = filters[3].text_input("Search stored question, model, reason, ID")
visible_rows = filter_requests(
    rows,
    tier=tier_filter,
    fallback=fallback_filter,
    overridden=override_filter,
    search=search_filter,
    settings=settings,
)

if visible_rows:
    st.dataframe(
        pd.DataFrame([_display_row(row, settings) for row in visible_rows]),
        use_container_width=True,
        hide_index=True,
    )
    ids = [row["request_id"] for row in visible_rows]
    selected_id = st.selectbox("Inspect request", ids)
    selected = next(row for row in visible_rows if row["request_id"] == selected_id)
    with st.container(border=True):
        st.caption(
            f"Question text: {selected['question_text']}"
            if selected.get("question_text")
            else "Question text was not stored (LOG_PROMPTS is disabled)."
        )
        _detail_panel(selected)
else:
    if history_error:
        st.warning("Request history could not be loaded, so filters were not applied.")
    else:
        st.info("No requests match these filters yet.")
