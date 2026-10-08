import requests
url = "http://localhost:11434/api/generate"


payload = {
    "model": "qwen2.5:3b",
    "prompt": "What is 2+2? Answer in one word.",
    "stream": False  
}

response = requests.post(url, json=payload)
data = response.json()

print("Model said:", data["response"])