from model_adapter import ask_local_model, ask_strong_model


# Keywords that usually indicate a question needs more reasoning
REASONING_KEYWORDS = [
    "explain",
    "analyze",
    "analyse",
    "compare",
    "difference",
    "why",
    "how",
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

# Keywords that indicate a question may require more context
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
    # ---------------------------------------------------------
    reasoning_matches = [
        word for word in REASONING_KEYWORDS
        if word in question_lower
    ]

    if reasoning_matches:
        score += 2
        reasons.append(
            "reasoning keywords: " + ", ".join(reasoning_matches[:3])
        )

    # ---------------------------------------------------------
    # 3. Coding / technical keywords
    # ---------------------------------------------------------
    code_matches = [
        word for word in CODE_KEYWORDS
        if word in question_lower
    ]

    if code_matches:
        score += 2
        reasons.append(
            "technical keywords: " + ", ".join(code_matches[:3])
        )

    # ---------------------------------------------------------
    # 4. Context-heavy keywords
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
    # 6. Advanced reasoning phrases
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

    # ---------------------------------------------------------
    # Final classification
    # ---------------------------------------------------------
    if score >= 3:
        complexity = "HIGH"
    else:
        complexity = "LOW"

    return complexity, score, reasons


def route_question(question):
    """
    Routes the question to the appropriate model.

    LOW  -> Local Ollama model
    HIGH -> Groq strong model
    """

    complexity, score, reasons = calculate_complexity(question)

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
    if complexity == "LOW":

        print("Model      : Local Ollama (qwen2.5:3b)")
        print("Route      : CHEAP / EASY TIER")
        print("=" * 60)

        answer = ask_local_model(question)

    else:

        print("Model      : Groq (openai/gpt-oss-120b)")
        print("Route      : STRONG / HARD TIER")
        print("=" * 60)

        answer = ask_strong_model(question)

    print("\nANSWER")
    print("-" * 60)
    print(answer)
    print("=" * 60)

    return {
        "question": question,
        "complexity": complexity,
        "score": score,
        "model": (
            "local"
            if complexity == "LOW"
            else "groq"
        ),
        "answer": answer,
        "reasons": reasons,
    }


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

        if question.lower() == "exit":
            print("\nExiting router.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            route_question(question)

        except Exception as error:
            print("\nERROR:")
            print(error)


if __name__ == "__main__":
    main()