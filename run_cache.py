import json, os, time
from model_adapter import ask_local_model, ask_strong_model

CACHE = "answers_cache.json"
SUFFIX = "\n\nSolve step by step. End your reply with 'Final answer: <number>' and nothing after it."

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
questions = json.load(open("benchmark_questions.json"))

for i, item in enumerate(questions):
    for name, fn in [("local", ask_local_model), ("strong", ask_strong_model)]:
        key = f"{i}|{name}"
        if key in cache:
            continue
        start = time.time()
        try:
            answer = fn(item["question"] + SUFFIX)
        except Exception as e:
            print("skipped", key, "-", e)   # e.g. rate limit; just rerun later
            continue
        cache[key] = {"answer": answer, "seconds": round(time.time() - start, 2)}
        json.dump(cache, open(CACHE, "w"), indent=2)
        print("done", key)
        if name == "strong":
            time.sleep(2)   