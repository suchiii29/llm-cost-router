import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


class RequestRecord(Base):
    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    question_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    question_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    reasons_json: Mapped[str] = mapped_column(Text, nullable=False)
    routed_to: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    provider: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    fallback: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fallback_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    overridden: Mapped[bool] = mapped_column(Boolean, nullable=False)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    token_source: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    baseline_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    input_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    output_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cost_currency: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)
    actual_model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    pricing_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cost_status: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    attempts_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_status: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)


@lru_cache(maxsize=8)
def get_engine(db_url: str):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args, future=True)


def initialize_database(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    engine = get_engine(settings.db_url)
    Base.metadata.create_all(engine)
    if not settings.db_url.startswith("sqlite"):
        return
    additions = {
        "total_tokens": "INTEGER",
        "input_cost_usd": "FLOAT",
        "output_cost_usd": "FLOAT",
        "cost_currency": "VARCHAR(3)",
        "actual_model": "VARCHAR(255)",
        "pricing_version": "TEXT",
        "cost_status": "VARCHAR(40)",
        "attempts_json": "TEXT",
        "request_status": "VARCHAR(16)",
        "provider": "VARCHAR(32)",
    }
    with engine.begin() as connection:
        columns = {column["name"] for column in inspect(connection).get_columns("requests")}
        for name, sql_type in additions.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE requests ADD COLUMN {name} {sql_type}"))
        connection.execute(text(
            "UPDATE requests SET actual_model = model WHERE actual_model IS NULL"
        ))
        connection.execute(text(
            "UPDATE requests SET total_tokens = input_tokens + output_tokens "
            "WHERE total_tokens IS NULL AND input_tokens IS NOT NULL AND output_tokens IS NOT NULL"
        ))
        connection.execute(text(
            "UPDATE requests SET cost_status = CASE WHEN cost_usd IS NULL "
            "THEN 'unavailable' ELSE 'legacy_estimate' END WHERE cost_status IS NULL"
        ))
        connection.execute(text(
            "UPDATE requests SET request_status = 'success' WHERE request_status IS NULL"
        ))


def log_request(
    *,
    question: str,
    score: int,
    reasons: list[str],
    routed_to: str,
    model: Optional[str],
    provider: Optional[str] = None,
    fallback: bool,
    fallback_reason: Optional[str],
    overridden: bool,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    token_source: str,
    latency_ms: Optional[int],
    cost_usd: Optional[float],
    baseline_cost_usd: Optional[float],
    total_tokens: Optional[int] = None,
    input_cost_usd: Optional[float] = None,
    output_cost_usd: Optional[float] = None,
    cost_currency: Optional[str] = "USD",
    actual_model: Optional[str] = None,
    pricing_version: Optional[str] = None,
    cost_status: Optional[str] = None,
    attempts_json: Optional[str] = None,
    request_status: Optional[str] = "success",
    cache_hit: bool = False,
    settings: Optional[Settings] = None,
) -> int:
    settings = settings or get_settings()
    engine = get_engine(settings.db_url)
    initialize_database(settings)
    record = RequestRecord(
        question_hash=hashlib.sha256(question.encode("utf-8")).hexdigest(),
        question_text=question if settings.log_prompts else None,
        score=score,
        reasons_json=json.dumps(reasons, ensure_ascii=False),
        routed_to=routed_to,
        model=model,
        provider=provider,
        fallback=fallback,
        fallback_reason=fallback_reason,
        overridden=overridden,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_source=token_source,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        baseline_cost_usd=baseline_cost_usd,
        cache_hit=cache_hit,
        total_tokens=total_tokens if total_tokens is not None else (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None else None
        ),
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        cost_currency=cost_currency,
        actual_model=actual_model,
        pricing_version=pricing_version,
        cost_status=cost_status or ("measured" if cost_usd is not None else "unavailable"),
        attempts_json=attempts_json,
        request_status=request_status,
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()
        return record.id
