import requests

from model_adapter import get_groq_api_key


def main():
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {get_groq_api_key()}"},
        timeout=30,
    )
    response.raise_for_status()

    for model in response.json()["data"]:
        print(model["id"])


if __name__ == "__main__":
    main()
