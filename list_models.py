import os
import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/models"
headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

response = requests.get(url, headers=headers)
data = response.json()

for model in data["data"]:
    print(model["id"])