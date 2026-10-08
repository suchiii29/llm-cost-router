from datasets import load_dataset
import json


print("Downloading GSM8K sample...")

dataset = load_dataset("openai/gsm8k", "main", split="test")



sample = dataset.select(range(60))

questions = []
for item in sample:
   
    full_answer = item["answer"]
    final_answer = full_answer.split("####")[-1].strip()

    questions.append({
        "question": item["question"],
        "correct_answer": final_answer,
        "category": "math"
    })

with open("benchmark_questions.json", "w") as f:
    json.dump(questions, f, indent=2)

print(f"Saved {len(questions)} benchmark questions to benchmark_questions.json")