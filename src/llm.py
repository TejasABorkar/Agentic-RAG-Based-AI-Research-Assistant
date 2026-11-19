from google import genai
from google.genai import types  # <-- हे नवीन इम्पोर्ट जोड
from src.config import GEMINI_API_KEY

# क्लायंट बनवताना आवृत्ती v1beta अशी सेट कर
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(api_version='v1beta')  # <-- हे महत्त्वाचं
)

def ask_llm(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"ERROR: {str(e)}"