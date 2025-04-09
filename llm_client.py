import os
import requests
import urllib3
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Disable SSL warnings (since we’re using verify=False as a workaround)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def query_mistral(prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    """Send a prompt to Mistral LLM via Hugging Face API and return the response."""
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True
        }
    }

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            verify=False  # SSL workaround
        )
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    
        # full_response = response.json()[0]["generated_text"]

        # if prompt in full_response:
        #     return full_response.replace(prompt, "").strip()
        # return full_response.strip()

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error: {http_err}")
        print(f"Response content: {response.text}")
        return "Mistral API returned an HTTP error."

    except Exception as err:
        print(f"Unexpected error: {err}")
        return "An unexpected error occurred while calling Mistral."
