import re

from model_adapter import (
    GroqAPIError,
    GroqConfigurationError,
    ask_cheap_model,
    ask_strong_model,
)
from model_config import get_model_config


# Keywords that usually indicate a question needs more reasoning
REASONING_KEYWORDS = [
    "explain",
    "analyze",
    "analyse",
    "compare",
    "difference",
    "design",
    "develop",
    "evaluate",
    "calculate",
    "solve",
    "derive",
    "prove",
    "optimize",
    "optimization",
    "algorithm",
    "architecture",
    "strategy",
    "reasoning",
    "advantages",
    "disadvantages",
    "trade-off",
    "tradeoff",
]

# Keywords that often indicate coding/technical work
CODE_KEYWORDS = [
    "code",
    "python",
    "java",
    "javascript",
    "program",
    "programming",
    "function",
    "algorithm",
    "debug",
    "database",
    "sql",
    "api",
    "recursion",
    "class",
    "loop",
]

# Keywords that indicate a question may require more context.
# NOTE: these are multi-word phrases, so they still need substring
# matching against question_lower (not the single-word list).
CONTEXT_KEYWORDS = [
    "dataset",
    "research",
    "paper",
    "project",
    "system",
    "architecture",
    "implementation",
    "machine learning",
    "deep learning",
    "artificial intelligence",
]


def calculate_complexity(question):
    """
    Calculates a simple complexity score using rules.

    Higher score = more complex question.
    """

    question_lower = question.lower()
    words = question_lower.split()

    score = 0
    reasons = []

    # ---------------------------------------------------------
    # 1. Question length
    # ---------------------------------------------------------
    word_count = len(words)

    if word_count > 40:
        score += 2
        reasons.append("long question")
    elif word_count > 20:
        score += 1
        reasons.append("moderately long question")

    # ---------------------------------------------------------
    # 2. Reasoning keywords
    # FIX: match against `words` (the split word list), not
    # `question_lower` (the raw string). Checking against the raw
    # string causes false matches when a keyword is a substring of
    # a longer word — e.g. "api" inside "capital".
    # ---------------------------------------------------------
    reasoning_matches = [
        word for word in REASONING_KEYWORDS
        if word in words
    ]

    if reasoning_matches:
        score += 2
        reasons.append(
            "reasoning keywords: " + ", ".join(reasoning_matches[:3])
        )

    # ---------------------------------------------------------
    # 3. Coding / technical keywords
    # FIX: same word-boundary fix as above. This is the one that
    # was causing "capital" to falsely match "api".
    # ---------------------------------------------------------
    code_matches = [
        word for word in CODE_KEYWORDS
        if word in words
    ]

    if code_matches:
        score += 2
        reasons.append(
            "technical keywords: " + ", ".join(code_matches[:3])
        )

    # ---------------------------------------------------------
    # 4. Context-heavy keywords
    # NOT changed to `words` — CONTEXT_KEYWORDS has multi-word
    # phrases like "machine learning", which would never match
    # against a list of single split words. Substring matching is
    # correct here.
    # ---------------------------------------------------------
    context_matches = [
        word for word in CONTEXT_KEYWORDS
        if word in question_lower
    ]

    if context_matches:
        score += 1
        reasons.append(
            "context keywords: " + ", ".join(context_matches[:3])
        )

    # ---------------------------------------------------------
    # 5. Math / symbolic expressions
    # ---------------------------------------------------------
    math_symbols = ["+", "-", "*", "/", "=", "^", "%"]

    found_math_symbols = [
        symbol for symbol in math_symbols
        if symbol in question
    ]

    if len(found_math_symbols) >= 2:
        score += 2
        reasons.append("multiple mathematical symbols")
    elif len(found_math_symbols) == 1:
        score += 1
        reasons.append("mathematical expression")

    # ---------------------------------------------------------
    # 6. Advanced reasoning phrases (multi-word, substring is correct)
    # ---------------------------------------------------------
    advanced_phrases = [
        "step by step",
        "in detail",
        "with examples",
        "pros and cons",
        "compare and contrast",
        "what would happen if",
        "justify your answer",
        "provide a solution",
    ]

    advanced_matches = [
        phrase for phrase in advanced_phrases
        if phrase in question_lower
    ]

    if advanced_matches:
        score += 2
        reasons.append(
            "advanced reasoning phrase: " + advanced_matches[0]
        )
    numbers_found = re.findall(r'\d+', question)

    if len(numbers_found) >= 4:
        score += 2
        reasons.append("multiple numeric values (likely multi-step)")
    elif len(numbers_found) >= 2:
        score += 1
        reasons.append("several numeric values")

    # ---------------------------------------------------------
    # Final classification
    # ---------------------------------------------------------
    if score >= 3:
        complexity = "HIGH"
    else:
        complexity = "LOW"

    return complexity, score, reasons


def route_question(question, force_model=None, display=True):
    """
    Routes the question to the appropriate model.

    LOW  -> Cheap Groq model
    HIGH -> Groq strong model
    """

    if force_model not in (None, "local", "groq", "cheap", "strong"):
        raise ValueError(
            "force_model must be 'cheap', 'strong', 'local', 'groq', or None"
        )

    complexity, score, reasons = calculate_complexity(question)
    config = get_model_config()
    routed_to = "cheap" if complexity == "LOW" else "strong"
    selected_model = force_model or routed_to
    if selected_model == "local":
        selected_model = "cheap"
    elif selected_model == "groq":
        selected_model = "strong"
    fallback = False
    fallback_reason = None
    api_attempts = []

    def record_success(response):
        api_attempts.append({
            "model_id": response.model_id,
            "outcome": "success",
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "latency_seconds": response.latency_seconds,
        })

    def record_failure(model_id, error):
        api_attempts.append({
            "model_id": model_id,
            "outcome": "failure",
            "error_type": type(error).__name__,
            "status_code": getattr(error, "status_code", None),
            "error": str(error),
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "latency_seconds": None,
        })

    if display:
        print("\n" + "=" * 60)
        print("QUERY ROUTING")
        print("=" * 60)

        print(f"Question   : {question}")
        print(f"Complexity : {complexity}")
        print(f"Score      : {score}")

        if reasons:
            print("Reasons    :")
            for reason in reasons:
                print(f"  - {reason}")
        else:
            print("Reasons    : No strong complexity indicators detected")

    # ---------------------------------------------------------
    # Route to the appropriate model
    # ---------------------------------------------------------
    if force_model and display:
        print("[MANUAL OVERRIDE]")

    if selected_model == "cheap":
        if display:
            print(f"Model      : Groq cheap ({config.cheap_model_id})")
            print("Route      : CHEAP / EASY TIER")
            print("=" * 60)

        try:
            response = ask_cheap_model(question)
        except (GroqAPIError, GroqConfigurationError) as error:
            record_failure(config.cheap_model_id, error)
            raise
        record_success(response)
    else:
        if display:
            print(f"Model      : Groq strong ({config.strong_model_id})")
            print("Route      : STRONG / HARD TIER")
            print("=" * 60)

        try:
            response = ask_strong_model(question)
            record_success(response)
        except GroqConfigurationError as error:
            record_failure(config.strong_model_id, error)
            error.api_attempts = list(api_attempts)
            raise
        except GroqAPIError as error:
            record_failure(config.strong_model_id, error)
            if not error.recoverable:
                error.api_attempts = list(api_attempts)
                raise

            fallback_reason = str(error)
            fallback = True
            if display:
                print(
                    f"[FALLBACK: {fallback_reason}; trying cheap model "
                    f"{config.cheap_model_id}]"
                )
            try:
                response = ask_cheap_model(question)
                record_success(response)
            except (GroqAPIError, GroqConfigurationError) as fallback_error:
                record_failure(config.cheap_model_id, fallback_error)
                fallback_error.api_attempts = list(api_attempts)
                final_error = GroqAPIError(
                    f"Strong model failed ({fallback_reason}); cheap-model "
                    f"fallback also failed ({fallback_error})",
                    status_code=getattr(fallback_error, "status_code", None),
                    recoverable=False,
                )
                final_error.api_attempts = list(api_attempts)
                raise final_error from fallback_error

    if display:
        print("\nANSWER")
        print("-" * 60)
        print(response.answer)
        print("=" * 60)

    result = {
        "question": question,
        "complexity": complexity,
        "score": score,
        "routed_to": routed_to,
        "model": response.model_id,
        "model_tier": "cheap" if fallback else selected_model,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "api_attempts": api_attempts,
        "answer": response.answer,
        "reasons": reasons,
        "provider_usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        },
        "latency_seconds": response.latency_seconds,
    }

    if force_model:
        result["overridden"] = True

    return result


def main():
    """
    Interactive testing mode.
    """

    print("=" * 60)
    print("LLM COST ROUTER")
    print("=" * 60)
    print("Type a question to test the router.")
    print("Type 'exit' to stop.")
    print("=" * 60)

    while True:

        question = input("\nAsk a question: ").strip()
        force_model = None

        if question.lower().startswith("force cheap:"):
            force_model = "cheap"
            question = question[len("force cheap:"):].strip()
        elif question.lower().startswith("force strong:"):
            force_model = "strong"
            question = question[len("force strong:"):].strip()
        elif question.lower().startswith("force local:"):
            force_model = "cheap"
            question = question[len("force local:"):].strip()
        elif question.lower().startswith("force groq:"):
            force_model = "strong"
            question = question[len("force groq:"):].strip()

        if question.lower() == "exit":
            print("\nExiting router.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            route_question(question, force_model=force_model)

        except Exception as error:
            print("\nERROR:")
            print(error)


if __name__ == "__main__":
    main()