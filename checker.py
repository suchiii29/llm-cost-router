import re
import unicodedata
from typing import Optional


CLASSIFICATION_LABELS = frozenset({"billing", "technical", "account"})


def extract_number(text):
    matches = re.findall(
        r'final answer[\s\*:]*([+-]?)\s*\$?\s*([+-]?[\d,]*\.?\d+)',
        text,
        re.I,
    )
    if matches:
        sign, number = matches[-1]
        if sign:
            number = sign + number.lstrip("+-")
        return number.replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return nums[-1] if nums else None

def is_correct(answer_text, correct_answer):
    got = extract_number(answer_text)
    if got is None:
        return False
    try:
        expected = str(correct_answer).replace(",", "").strip()
        return abs(float(got) - float(expected)) < 1e-6
    except (TypeError, ValueError, OverflowError):
        return False


def normalize_text(text: str) -> str:
    """Normalize case and whitespace while preserving Unicode meaning."""
    normalized = unicodedata.normalize("NFC", str(text)).casefold()
    return " ".join(normalized.split())


def _letter_from_text(text: str) -> Optional[str]:
    text = unicodedata.normalize("NFC", str(text)).strip()
    explicit_patterns = (
        r"\b(?:final\s+)?answer\s*(?:is|:|=)?\s*[\(\[]?([A-Z])[\)\]]?\b",
        r"\b(?:option|choice)\s*[\(\[]?([A-Z])[\)\]]?\b",
    )
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return matches[-1].casefold()

    standalone = re.fullmatch(r"[\(\[]?([A-Z])[\)\]]?[.)]?", text, flags=re.I)
    if standalone:
        return standalone.group(1).casefold()

    letters = re.findall(r"\b([A-Z])\b", text, flags=re.I)
    return letters[-1].casefold() if letters else None


def is_letter_correct(answer_text: str, correct_answer: str) -> bool:
    """Compare a multiple-choice response with a single-letter answer key."""
    expected = re.fullmatch(
        r"\s*[\(\[]?([A-Z])[\)\]]?[.)]?\s*", str(correct_answer), flags=re.I
    )
    if not expected:
        return False
    got = _letter_from_text(answer_text)
    return got == expected.group(1).casefold()


def is_exact_correct(answer_text: str, correct_answer: str) -> bool:
    """Compare answers after Unicode, case, and whitespace normalization."""
    return normalize_text(answer_text) == normalize_text(correct_answer)


def is_contains_correct(answer_text: str, correct_answer: str) -> bool:
    """Accept exactly one allowed classification label and no other words."""
    expected = normalize_text(correct_answer)
    response = normalize_text(answer_text)
    return expected in CLASSIFICATION_LABELS and response == expected


def check_answer(answer_text: str, correct_answer: str, checker: str) -> bool:
    """Evaluate a response against ground truth using the dataset's checker."""
    checkers = {
        "numeric": is_correct,
        "letter": is_letter_correct,
        "exact": is_exact_correct,
        "contains": is_contains_correct,
    }
    check = checkers.get(checker)
    return bool(check and check(answer_text, correct_answer))
