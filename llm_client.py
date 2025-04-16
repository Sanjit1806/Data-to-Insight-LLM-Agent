# for mistral llm
# import os
# import requests
# import urllib3
# from dotenv import load_dotenv

# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")

# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
# HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# # Disable SSL warnings (since we’re using verify=False as a workaround)
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# def query_mistral(prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": max_new_tokens,
#             "temperature": temperature,
#             "do_sample": True
#         }
#     }

#     try:
#         response = requests.post(
#             API_URL,
#             headers=HEADERS,
#             json=payload,
#             verify=False  # SSL workaround
#         )
#         response.raise_for_status()
#         return response.json()[0]["generated_text"]
    
#         # full_response = response.json()[0]["generated_text"]

#         # if prompt in full_response:
#         #     return full_response.replace(prompt, "").strip()
#         # return full_response.strip()

#     except requests.exceptions.HTTPError as http_err:
#         print(f"HTTP Error: {http_err}")
#         print(f"Response content: {response.text}")
#         return "Mistral API returned an HTTP error."

#     except Exception as err:
#         print(f"Unexpected error: {err}")
#         return "An unexpected error occurred while calling Mistral."




#for gemini llm
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

def query_mistral(prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Gemini API returned an error."
