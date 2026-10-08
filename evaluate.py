import json, math
from checker import is_correct, extract_number
from router import calculate_complexity

PRICE_IN, PRICE_OUT = 0.00015, 0.00075   # Groq gpt-oss-120b, per 1K tokens


def toks(text):
    return max(1, len(text) // 4)


def strong_cost(q, a):
    return toks(q) / 1000 * PRICE_IN + toks(a) / 1000 * PRICE_OUT


def wilson(k, n, z=1.96):
    """95% confidence interval for an accuracy of k out of n."""
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


questions = json.load(open("benchmark_questions.json"))
cache = json.load(open("answers_cache.json"))

rows = []
for i, item in enumerate(questions):
    lk, sk = f"{i}|local", f"{i}|strong"
    if lk not in cache or sk not in cache:
        continue
    q = item["question"]
    la, sa = cache[lk]["answer"], cache[sk]["answer"]
    complexity, score, _ = calculate_complexity(q)
    rows.append({
        "i": i,
        "question": q,
        "score": score,
        "rules_says_hard": complexity == "HIGH",
        "local_ok": is_correct(la, item["correct_answer"]),
        "strong_ok": is_correct(sa, item["correct_answer"]),
        "strong_cost": strong_cost(q, sa),
        "local_secs": cache[lk]["seconds"],
        "strong_secs": cache[sk]["seconds"],
    })

n = len(rows)
print(f"Evaluated {n} questions\n")

# Did the models follow the 'Final answer:' format?
for name in ("local", "strong"):
    k = sum("final answer" in v["answer"].lower()
            for key, v in cache.items() if key.endswith(name))
    print(f"{name}: {k} answers contain a 'Final answer' line")

# Sanity check on the checker: show a few 'wrong' strong answers
print("\nFirst 5 strong answers marked WRONG (check the checker isn't at fault):")
shown = 0
for r in rows:
    if not r["strong_ok"] and shown < 5:
        got = extract_number(cache[f"{r['i']}|strong"]["answer"])
        print(f"  q{r['i']}: extracted {got}, expected {questions[r['i']]['correct_answer']}")
        shown += 1

policies = {
    "Always local":           lambda r: False,
    "Always strong":          lambda r: True,
    "Rules router":           lambda r: r["rules_says_hard"],
    "Oracle (upper bound)":   lambda r: not r["local_ok"],
}

baseline = sum(r["strong_cost"] for r in rows)

print(f"\n{'Policy':<24}{'Accuracy':<26}{'Cost $':<12}{'To strong':<11}{'Saved'}")
for name, use_strong in policies.items():
    correct = cost = to_strong = 0
    for r in rows:
        s = use_strong(r)
        correct += r["strong_ok"] if s else r["local_ok"]
        cost += r["strong_cost"] if s else 0
        to_strong += s
    lo, hi = wilson(correct, n)
    acc = f"{correct/n*100:.0f}% ({lo*100:.0f}-{hi*100:.0f}%)"
    saved = (1 - cost / baseline) * 100 if baseline else 0
    print(f"{name:<24}{acc:<26}{cost:<12.6f}{to_strong}/{n:<8}{saved:.0f}%")

json.dump(rows, open("labels.json", "w"), indent=2)
print("\nSaved labels.json (training data for the classifier)")