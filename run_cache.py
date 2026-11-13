"""Compatibility CLI wrapper for split/tier cache generation."""

from scripts.run_cache import (
    SPLITS,
    TIERS,
    cache_path,
    cache_status,
    dataset_path,
    load_split,
    main,
    run_cache,
)

__all__ = [
    "SPLITS",
    "TIERS",
    "cache_path",
    "cache_status",
    "dataset_path",
    "load_split",
    "run_cache",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
