import requests
import json

LM_STUDIO_URL = "http://192.168.29.41:1234/v1/chat/completions"
# Note: Use the exact ID from your earlier curl: "qwen2.5-coder-7b-instruct"
MODEL_NAME = "qwen2.5-coder-7b-instruct" 

def ask_qwen(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert AI Engineer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    response = requests.post(LM_STUDIO_URL, json=payload)
    return response.json()['choices'][0]['message']['content']

# Example usage for your Enterprise Architecture platform
print(ask_qwen("which is the best model in LMstudio for python coding ?"))