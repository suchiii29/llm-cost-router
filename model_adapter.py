import os
import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def ask_local_model(question):
    """Sends a question to the local Ollama model (cheap/easy tier)."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:3b",
        "prompt": question,
        "stream": False
    }
    response = requests.post(url, json=payload)
    return response.json()["response"]


def ask_strong_model(question):
    """Sends a question to Groq's free API (strong/hard tier)."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "user", "content": question}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    question = input("Ask me anything: ")

    print("\n--- Small local model answer ---")
    print(ask_local_model(question))

    print("\n--- Strong API model (Groq) answer ---")
    print(ask_strong_model(question))