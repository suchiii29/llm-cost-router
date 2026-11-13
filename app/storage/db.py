import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine
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


@lru_cache(maxsize=8)
def get_engine(db_url: str):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args, future=True)


def initialize_database(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    Base.metadata.create_all(get_engine(settings.db_url))


def log_request(
    *,
    question: str,
    score: int,
    reasons: list[str],
    routed_to: str,
    model: Optional[str],
    fallback: bool,
    fallback_reason: Optional[str],
    overridden: bool,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    token_source: str,
    latency_ms: Optional[int],
    cost_usd: Optional[float],
    baseline_cost_usd: Optional[float],
    cache_hit: bool = False,
    settings: Optional[Settings] = None,
) -> int:
    settings = settings or get_settings()
    engine = get_engine(settings.db_url)
    Base.metadata.create_all(engine)
    record = RequestRecord(
        question_hash=hashlib.sha256(question.encode("utf-8")).hexdigest(),
        question_text=question if settings.log_prompts else None,
        score=score,
        reasons_json=json.dumps(reasons, ensure_ascii=False),
        routed_to=routed_to,
        model=model,
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
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()
        return record.id
