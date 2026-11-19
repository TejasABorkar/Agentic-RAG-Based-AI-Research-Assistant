from groq import Groq
from src.config import GROQ_API_KEY
import time

client = Groq(api_key=GROQ_API_KEY)

def ask_llm(prompt):
    models_to_try = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "429" in error_msg or "rate limit" in error_msg.lower():
                    print(f"Model {model_name} is busy. Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"Model {model_name} failed: {error_msg}")
                    break 
                    
    return "ERROR: All Groq models are currently busy. Please try again in a few minutes."