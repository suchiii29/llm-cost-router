def choose_tier(score: int, threshold: int = 3) -> int:
    """Map scores below the threshold to local tier 0, otherwise tier 2."""
    return 0 if score < threshold else 2
