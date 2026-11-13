"""Build reproducible router datasets when explicitly invoked by the user."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEV_SOURCE = PROJECT_ROOT / "benchmark_questions.json"
DEFAULT_SEED = 506
TRAIN_SOURCE_COUNT = 200
TEST_SOURCE_COUNT = 100
SIMPLE_TASK_TYPES = ("date", "email", "number", "json", "uppercase", "classify")
TASK_TYPES = {"math", "mcq", "extract", "classify", "reformat"}
CHECKERS = {"numeric", "letter", "exact", "contains"}

NAME_PARTS = (
    "Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Gray", "Harper",
    "Indigo", "Jordan", "Kai", "Logan", "Morgan", "Noel", "Oakley", "Parker",
    "Quinn", "Riley", "Sage", "Taylor", "Uma", "Val", "Winter", "Yael",
)
DOMAINS = ("example.org", "sample.net", "demo.test", "mail.example")
LIST_WORDS = (
    "amber", "birch", "cobalt", "dahlia", "ember", "fern", "garnet",
    "hazel", "indigo", "juniper", "khaki", "lilac", "maple", "navy",
    "olive", "pearl", "quartz", "ruby", "silver", "teal", "umber", "violet",
)
CLASSIFICATION_EXAMPLES = {
    "billing": (
        "I was charged twice for the same monthly plan.",
        "Please explain the unexpected fee on my invoice.",
        "My payment receipt lists the wrong total.",
        "I need to update the card used for my subscription.",
    ),
    "technical": (
        "The application closes whenever I open the report.",
        "The export button returns an error message.",
        "I cannot sign in because the page keeps reloading.",
        "The uploaded file is not appearing in my workspace.",
    ),
    "account": (
        "Please change the display name on my profile.",
        "I need to close my account and remove my profile.",
        "How can I update the email address on my account?",
        "I want to change the notification preferences.",
    ),
}
CLASSIFICATION_LABELS = tuple(CLASSIFICATION_EXAMPLES)


def normalize_question(question: str) -> str:
    """Normalize text only for identifying duplicate questions across splits."""
    return " ".join(question.casefold().split())


def validate_item(item: dict[str, Any], expected_split: Optional[str] = None) -> None:
    required = {"id", "split", "task_type", "question", "answer", "checker"}
    if set(item) != required:
        raise ValueError(f"Dataset item fields must be exactly {sorted(required)}")
    if not isinstance(item["id"], str) or not item["id"]:
        raise ValueError("Dataset item id must be a non-empty string")
    if expected_split is not None and item["split"] != expected_split:
        raise ValueError(f"Expected split {expected_split!r}, got {item['split']!r}")
    if item["split"] not in {"train_pool", "dev", "test"}:
        raise ValueError(f"Unknown split: {item['split']!r}")
    if item["task_type"] not in TASK_TYPES:
        raise ValueError(f"Unknown task type: {item['task_type']!r}")
    if item["checker"] not in CHECKERS:
        raise ValueError(f"Unknown checker: {item['checker']!r}")
    if not isinstance(item["question"], str) or not item["question"].strip():
        raise ValueError("Dataset question must be non-empty text")
    if not isinstance(item["answer"], (str, int, float)):
        raise ValueError("Dataset answer must be text or numeric")


def validate_items(items: list[dict[str, Any]], split: str) -> None:
    ids = set()
    for item in items:
        validate_item(item, split)
        if item["id"] in ids:
            raise ValueError(f"Duplicate item ID: {item['id']}")
        ids.add(item["id"])


def convert_dev_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap existing benchmark questions without changing their question/answer."""
    result = []
    for index, source in enumerate(questions):
        result.append(
            {
                "id": f"dev-existing-{index:03d}",
                "split": "dev",
                "task_type": "math",
                "question": source["question"],
                "answer": source["correct_answer"],
                "checker": "numeric",
            }
        )
    validate_items(result, "dev")
    return result


def _gsm8k_item(
    source: dict[str, Any], index: int, split: str
) -> Optional[dict[str, Any]]:
    match = re.search(r"####\s*(.+?)\s*$", source.get("answer", ""), re.S)
    if not match:
        return None
    answer = match.group(1).strip()
    if not answer:
        return None
    return {
        "id": f"gsm8k-{split}-{index:05d}",
        "split": "train_pool" if split == "train" else "test",
        "task_type": "math",
        "question": source["question"],
        "answer": answer,
        "checker": "numeric",
    }


def _arc_item(
    source: dict[str, Any], index: int, split: str
) -> Optional[dict[str, Any]]:
    choices = source.get("choices") or {}
    choice_texts = choices.get("text") or []
    choice_labels = choices.get("label") or []
    answer_key = str(source.get("answerKey", "")).strip()
    if not choice_texts:
        return None

    choice_index = None
    if answer_key in choice_labels:
        choice_index = choice_labels.index(answer_key)
    elif answer_key.isdigit():
        numeric_index = int(answer_key) - 1
        if 0 <= numeric_index < len(choice_texts):
            choice_index = numeric_index
    if choice_index is None or choice_index >= len(choice_texts):
        return None

    letters = [chr(ord("A") + choice) for choice in range(len(choice_texts))]
    options = "\n".join(
        f"{letter}. {text}" for letter, text in zip(letters, choice_texts)
    )
    return {
        "id": f"arc-challenge-{split}-{index:05d}",
        "split": "train_pool" if split == "train" else "test",
        "task_type": "mcq",
        "question": f"{source['question']}\nChoices:\n{options}\nReturn one letter.",
        "answer": letters[choice_index],
        "checker": "letter",
    }


def generate_simple_tasks(
    count: int, *, split: str, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Generate varied, deterministic extraction, classification and reformat tasks."""
    rng = random.Random(seed)
    items = []
    dates = [date(2024, 1, 1) + timedelta(days=rng.randrange(0, 1200)) for _ in range(count)]
    for index in range(count):
        task = SIMPLE_TASK_TYPES[index % len(SIMPLE_TASK_TYPES)]
        marker = f"{index + 1:04d}"
        name = rng.choice(NAME_PARTS)
        date_value = dates[index]
        if task == "date":
            date_text = (
                date_value.strftime("%Y-%m-%d")
                if index % 2 == 0
                else date_value.strftime("%B %d, %Y")
            )
            prompts = (
                "Read the note and return the meeting date in YYYY-MM-DD format.",
                "Extract the delivery date from this message; answer with an ISO date.",
                "What date is recorded for this appointment? Return YYYY-MM-DD only.",
            )
            question = (
                f"{rng.choice(prompts)} Ticket {marker}: {name} confirmed "
                f"the appointment for {date_text}."
            )
            answer = date_value.isoformat()
            task_type, checker = "extract", "exact"
        elif task == "email":
            email = f"{name.casefold()}.{marker}@{rng.choice(DOMAINS)}"
            prompts = (
                "Extract the contact email address. Return only the address.",
                "Which email is listed for this request? Give the exact address.",
                "Read the note and return its sender email, with no extra text.",
            )
            question = (
                f"{rng.choice(prompts)} Reference {marker}: {name} can be reached "
                f"at {email}; the phone extension is {rng.randrange(100, 999)}."
            )
            answer = email
            task_type, checker = "extract", "exact"
        elif task == "number":
            number = rng.randrange(100000, 999999)
            invoice = f"INV-{number}"
            amount = rng.randrange(20, 900)
            prompts = (
                "Return only the digits in the invoice reference.",
                "Extract the numeric reference ID, without the INV prefix.",
                "What is the invoice number? Answer with its digits only.",
            )
            question = (
                f"{rng.choice(prompts)} Batch {marker}: invoice {invoice}; "
                f"amount {amount} dollars; customer {name}."
            )
            answer = str(number)
            task_type, checker = "extract", "exact"
        elif task == "json":
            words = rng.sample(LIST_WORDS, 3)
            prompts = (
                "Convert the listed items into a JSON array of strings, preserving order.",
                "Return these values as compact JSON string-array syntax.",
                "Reformat the following list as a JSON array and add no commentary.",
            )
            question = (
                f"{rng.choice(prompts)} List {marker}: "
                + ", ".join(words)
                + "."
            )
            answer = json.dumps(words, ensure_ascii=False, separators=(",", ":"))
            task_type, checker = "reformat", "exact"
        elif task == "uppercase":
            phrases = (
                "please review the draft",
                "shipment arrives next week",
                "confirm the updated schedule",
                "save a copy for the team",
                "the meeting starts at noon",
                "your request is now complete",
            )
            phrase = rng.choice(phrases)
            question = (
                f"Convert this note to uppercase and return only the result. "
                f"Note {marker}: {phrase}."
            )
            answer = phrase.upper() + "."
            task_type, checker = "reformat", "exact"
        else:
            label = rng.choice(CLASSIFICATION_LABELS)
            description = rng.choice(CLASSIFICATION_EXAMPLES[label])
            prompts = (
                "Classify the request using one label only: billing, technical, account.",
                "Choose exactly one category (billing, technical, account) for this ticket.",
                "Return the best single label from billing, technical, account.",
            )
            question = f"{rng.choice(prompts)} Case {marker}: {description}"
            answer = label
            task_type, checker = "classify", "contains"

        items.append(
            {
                "id": f"simple-{split}-{index:04d}",
                "split": "train_pool" if split == "train" else "test",
                "task_type": task_type,
                "question": question,
                "answer": answer,
                "checker": checker,
            }
        )
    validate_items(items, "train_pool" if split == "train" else "test")
    return items


def _usable_source_items(
    source_items: Iterable[dict[str, Any]],
    converter: Callable[
        [dict[str, Any], int, str], Optional[dict[str, Any]]
    ],
    split: str,
) -> list[dict[str, Any]]:
    result = []
    for index, source in enumerate(source_items):
        item = converter(source, index, split)
        if item is not None:
            result.append(item)
    return result


def _choose_items(
    items: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    excluded_questions: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    excluded_questions = excluded_questions or set()
    candidates = [
        item
        for item in items
        if normalize_question(item["question"]) not in excluded_questions
    ]
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidates, found {len(candidates)}")
    selected = rng.sample(candidates, count)
    return sorted(selected, key=lambda item: item["id"])


def build_dataset_items(
    *,
    dev_questions: list[dict[str, Any]],
    gsm8k_train: list[dict[str, Any]],
    gsm8k_test: list[dict[str, Any]],
    arc_train: list[dict[str, Any]],
    arc_test: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Pure transformation; it never imports or calls the datasets package."""
    dev_items = convert_dev_questions(dev_questions)
    dev_keys = {normalize_question(item["question"]) for item in dev_items}
    rng = random.Random(seed)

    gsm_train_items = _usable_source_items(gsm8k_train, _gsm8k_item, "train")
    gsm_test_items = _usable_source_items(gsm8k_test, _gsm8k_item, "test")
    arc_train_items = _usable_source_items(arc_train, _arc_item, "train")
    arc_test_items = _usable_source_items(arc_test, _arc_item, "test")
    simple_train_items = generate_simple_tasks(
        TRAIN_SOURCE_COUNT, split="train", seed=seed + 1
    )
    simple_test_items = generate_simple_tasks(
        TEST_SOURCE_COUNT, split="test", seed=seed + 2
    )

    train_pool = (
        _choose_items(gsm_train_items, TRAIN_SOURCE_COUNT, rng)
        + _choose_items(arc_train_items, TRAIN_SOURCE_COUNT, rng)
        + simple_train_items
    )
    rng.shuffle(train_pool)
    test_items = (
        _choose_items(
            gsm_test_items, TEST_SOURCE_COUNT, rng, excluded_questions=dev_keys
        )
        + _choose_items(
            arc_test_items, TEST_SOURCE_COUNT, rng, excluded_questions=dev_keys
        )
        + simple_test_items
    )
    rng.shuffle(test_items)
    test_question_keys = {normalize_question(item["question"]) for item in test_items}
    train_pool = [
        item
        for item in train_pool
        if normalize_question(item["question"]) not in test_question_keys
    ]
    validate_items(train_pool, "train_pool")
    validate_items(test_items, "test")
    if len(dev_items) != len(dev_questions):
        raise ValueError("Dev conversion changed the source item count")
    return {"train_pool": train_pool, "dev": dev_items, "test": test_items}


def _jsonl_bytes(items: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in items
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def refuse_test_overwrite(data_dir: Path, *, force_test: bool) -> None:
    test_path = data_dir / "test.jsonl"
    if test_path.exists() and not force_test:
        raise FileExistsError(
            f"{test_path} already exists; pass --force-test to replace the frozen test set"
        )


def write_datasets(
    datasets: dict[str, list[dict[str, Any]]],
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    seed: int = DEFAULT_SEED,
    force_test: bool = False,
) -> dict[str, Any]:
    """Write JSONL datasets and their manifest, refusing implicit test replacement."""
    refuse_test_overwrite(data_dir, force_test=force_test)
    expected = {"train_pool", "dev", "test"}
    if set(datasets) != expected:
        raise ValueError(f"Expected dataset keys {sorted(expected)}")
    for split, items in datasets.items():
        validate_items(items, split)

    contents_by_file = {
        "train_pool.jsonl": _jsonl_bytes(datasets["train_pool"]),
        "dev.jsonl": _jsonl_bytes(datasets["dev"]),
        "test.jsonl": _jsonl_bytes(datasets["test"]),
    }
    for name, content in contents_by_file.items():
        _atomic_write(
            data_dir / name,
            content,
            exclusive=(name == "test.jsonl" and not force_test),
        )

    split_by_file = {
        "train_pool.jsonl": "train_pool",
        "dev.jsonl": "dev",
        "test.jsonl": "test",
    }
    manifest = {
        "seed": seed,
        "files": {
            name: {
                "item_count": len(datasets[split_by_file[name]]),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in contents_by_file.items()
        },
        "metadata": {
            "sources": {
                "GSM8K": "openai/gsm8k configuration main",
                "ARC-Challenge": "allenai/ai2_arc configuration ARC-Challenge",
                "dev": DEV_SOURCE.name,
            },
            "composition": {
                "train_pool": {
                    "gsm8k_train": sum(
                        item["id"].startswith("gsm8k-train-")
                        for item in datasets["train_pool"]
                    ),
                    "arc_challenge_train": sum(
                        item["id"].startswith("arc-challenge-train-")
                        for item in datasets["train_pool"]
                    ),
                    "synthetic_simple": sum(
                        item["id"].startswith("simple-train-")
                        for item in datasets["train_pool"]
                    ),
                },
                "test": {
                    "gsm8k_test": TEST_SOURCE_COUNT,
                    "arc_challenge_test_mcq": TEST_SOURCE_COUNT,
                    "synthetic_simple": TEST_SOURCE_COUNT,
                },
            },
        },
    }
    _atomic_write(
        data_dir / "MANIFEST.json",
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return manifest


def load_remote_datasets() -> tuple[Any, Any, Any, Any]:
    """Download selected source splits; called only by the explicit CLI."""
    from datasets import load_dataset

    gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    arc_train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    return gsm8k_train, gsm8k_test, arc_train, arc_test


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-test",
        action="store_true",
        help="explicitly replace an existing frozen data/test.jsonl",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    try:
        refuse_test_overwrite(args.data_dir, force_test=args.force_test)
    except FileExistsError as error:
        parser.error(str(error))

    dev_questions = json.loads(DEV_SOURCE.read_text(encoding="utf-8"))
    gsm8k_train, gsm8k_test, arc_train, arc_test = load_remote_datasets()
    datasets = build_dataset_items(
        dev_questions=dev_questions,
        gsm8k_train=list(gsm8k_train),
        gsm8k_test=list(gsm8k_test),
        arc_train=list(arc_train),
        arc_test=list(arc_test),
        seed=DEFAULT_SEED,
    )
    manifest = write_datasets(
        datasets,
        data_dir=args.data_dir,
        seed=DEFAULT_SEED,
        force_test=args.force_test,
    )
    counts = {name: entry["item_count"] for name, entry in manifest["files"].items()}
    print(json.dumps({"counts": counts, "seed": DEFAULT_SEED}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
