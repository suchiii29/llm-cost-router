"""Compatibility command wrapper for request tracking."""

from app.tracking import (
    LOG_FILE,
    estimate_cost,
    estimate_tokens,
    main,
    save_log,
    track_question,
)

__all__ = [
    "LOG_FILE",
    "estimate_cost",
    "estimate_tokens",
    "save_log",
    "track_question",
    "main",
]


if __name__ == "__main__":
    main()
