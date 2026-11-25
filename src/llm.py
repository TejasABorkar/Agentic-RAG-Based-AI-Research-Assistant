import time

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODELS,
    LLM_PROVIDER,
)


class LLMError(Exception):
    """Raised when no language model could answer."""


_gemini_client = None
_groq_client = None


def _is_busy(error):
    msg = str(error).lower()
    keys = ("503", "429", "rate limit", "quota", "overloaded", "unavailable")
    return any(k in msg for k in keys)


def _ask_gemini(prompt):
    global _gemini_client
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set")
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    last_error = "unknown error"
    for attempt in range(3):
        try:
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
            text = response.text
            if text and text.strip():
                return text.strip()
            last_error = "empty answer"
            break
        except Exception as e:
            last_error = str(e)
            if _is_busy(e):
                time.sleep(2 * (attempt + 1))
                continue
            break
    raise LLMError(f"Gemini failed ({last_error[:150]})")


def _ask_groq(prompt):
    global _groq_client
    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set")
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)

    last_error = "unknown error"
    for model_name in GROQ_MODELS:
        for attempt in range(3):
            try:
                response = _groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                )
                text = response.choices[0].message.content
                if text and text.strip():
                    return text.strip()
                last_error = f"{model_name} returned an empty answer"
                break
            except Exception as e:
                last_error = f"{model_name}: {e}"
                if _is_busy(e):
                    time.sleep(2 * (attempt + 1))
                    continue
                break
    raise LLMError(f"Groq failed ({last_error[:150]})")


def ask_llm(prompt):
    """Return the model's answer text. Raises LLMError if every model fails."""
    if LLM_PROVIDER == "groq":
        providers = [_ask_groq, _ask_gemini]
    else:
        providers = [_ask_gemini, _ask_groq]

    errors = []
    for provider in providers:
        try:
            return provider(prompt)
        except Exception as e:  # LLMError, missing package, network problem
            errors.append(str(e)[:200])
    raise LLMError(" | ".join(errors))