import json
import os
import time

from checker import is_correct
from model_adapter import ask_cheap_model
from model_config import get_model_config


N = 150
OUT = "train_labels_groq.json"
SUFFIX = "\n\nSolve step by step. End your reply with 'Final answer: <number>' and nothing after it."


def main():
    from datasets import load_dataset

    config = get_model_config()
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
    ).select(range(N))
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}

    for index, item in enumerate(dataset):
        if str(index) in done:
            continue

        correct_answer = (
            item["answer"].split("####")[-1].strip().replace(",", "")
        )
        started = time.time()
        try:
            response = ask_cheap_model(item["question"] + SUFFIX)
        except Exception as error:
            print("skipped", index, "-", error)
            continue

        done[str(index)] = {
            "question": item["question"],
            "model_id": response.model_id,
            "cheap_ok": is_correct(response.answer, correct_answer),
            "seconds": round(time.time() - started, 2),
        }
        with open(OUT, "w", encoding="utf-8") as file:
            json.dump(done, file, indent=2)
        print("done", index, "with", config.cheap_model_id)


if __name__ == "__main__":
    main()
