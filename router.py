"""Compatibility command wrapper for the packaged router."""

from app.routing.complexity import calculate_complexity
from app.routing.router import ask_tier, main, route_question

__all__ = ["ask_tier", "calculate_complexity", "route_question", "main"]


if __name__ == "__main__":
    main()
