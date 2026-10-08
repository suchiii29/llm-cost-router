import json, os, time
from datasets import load_dataset
from model_adapter import ask_local_model
from checker import is_correct

N = 150
OUT = "train_labels.json"
SUFFIX = "\n\nSolve step by step. End your reply with 'Final answer: <number>' and nothing after it."

data = load_dataset("openai/gsm8k", "main", split="train").select(range(N))
done = json.load(open(OUT)) if os.path.exists(OUT) else {}

for i, item in enumerate(data):
    if str(i) in done:
        continue
    correct_answer = item["answer"].split("####")[-1].strip().replace(",", "")
    start = time.time()
    try:
        answer = ask_local_model(item["question"] + SUFFIX)
    except Exception as e:
        print("skipped", i, "-", e)
        continue
    done[str(i)] = {
        "question": item["question"],
        "local_ok": is_correct(answer, correct_answer),
        "seconds": round(time.time() - start, 2),
    }
    json.dump(done, open(OUT, "w"), indent=2)
    print("done", i)
