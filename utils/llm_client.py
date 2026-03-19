import os
import json
import time
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup basic logging for failovers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Provider SDK imports ──
try:
    from groq import Groq
    GROQ_AVAILABLE = True
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except ImportError:
    GROQ_AVAILABLE = False
    groq_client = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False

# ── Retry & Cooldown Config ──
GROQ_MAX_RETRIES = 3               # Retry up to 3 times before fallback
GROQ_BASE_DELAY = 1.0              # 1s → 2s → 4s exponential backoff
GROQ_COOLDOWN_SECONDS = 60         # After a rate-limit, skip Groq for 60s
_groq_cooldown_until = 0.0         # Timestamp when cooldown expires


def _is_rate_limit(error_msg: str) -> bool:
    """Check if an error string indicates a rate limit."""
    return "429" in error_msg or "rate limit" in error_msg.lower() or "rate_limit" in error_msg.lower()


def get_llm_response(
    messages: list,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    response_format: dict = None
) -> str:
    """
    Centralized LLM client with retry + exponential backoff.

    Strategy:
      1. If Groq is in cooldown → skip straight to Gemini
      2. Try Groq up to GROQ_MAX_RETRIES times with exponential backoff
      3. On rate-limit → set 60s cooldown, fall back to Gemini
      4. On other errors → fall back to Gemini immediately
    """
    global _groq_cooldown_until

    # ── Skip Groq if in cooldown ──
    if GROQ_AVAILABLE and time.time() < _groq_cooldown_until:
        logger.info("Groq in cooldown — routing directly to Gemini")
        return _fallback_to_gemini(messages, temperature, max_tokens, response_format)

    # ── Attempt Groq with retry ──
    if GROQ_AVAILABLE:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        last_error = None
        for attempt in range(GROQ_MAX_RETRIES):
            try:
                response = groq_client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                error_msg = str(e)

                if _is_rate_limit(error_msg):
                    delay = GROQ_BASE_DELAY * (2 ** attempt)
                    if attempt < GROQ_MAX_RETRIES - 1:
                        logger.warning(
                            f"Groq 429 (attempt {attempt+1}/{GROQ_MAX_RETRIES}) — "
                            f"retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        # Exhausted retries — activate cooldown
                        _groq_cooldown_until = time.time() + GROQ_COOLDOWN_SECONDS
                        logger.warning(
                            f"Groq rate-limited after {GROQ_MAX_RETRIES} retries — "
                            f"cooldown {GROQ_COOLDOWN_SECONDS}s, falling back to Gemini"
                        )
                else:
                    # Non-rate-limit error — don't retry, go straight to fallback
                    logger.error(f"Groq error (non-retryable): {e}")
                    break

        # All retries exhausted or non-retryable error
        return _fallback_to_gemini(messages, temperature, max_tokens, response_format)
    else:
        return _fallback_to_gemini(messages, temperature, max_tokens, response_format)


GEMINI_MAX_RETRIES = 2
GEMINI_RETRY_DELAY = 16  # seconds — slightly over the 14s the API suggests


def _fallback_to_gemini(messages, temperature, max_tokens, response_format):
    """
    Fallback implementation using Google Gemini.
    Translates OpenAI/Groq-style messages into Gemini format.
    Retries up to GEMINI_MAX_RETRIES times on 429 rate-limit errors.
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Groq failed, and Gemini is not available or 'GEMINI_API_KEY' is missing.")

    # Convert messages — Gemini uses system_instruction + user/model turns
    system_instruction = ""
    history = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_instruction += content + "\n"
        elif role == "user":
            history.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            history.append({"role": "model", "parts": [content]})

    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system_instruction.strip() if system_instruction else None,
    )

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    if response_format and response_format.get("type") == "json_object":
        generation_config.response_mime_type = "application/json"

    last_error = None
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            hist_copy = [h.copy() for h in history]
            if len(hist_copy) > 0 and hist_copy[-1]["role"] == "user":
                last_msg = hist_copy.pop()["parts"][0]
            else:
                last_msg = ""

            chat = gemini_model.start_chat(history=hist_copy)
            response = chat.send_message(last_msg, generation_config=generation_config)
            return response.text
        except Exception as e:
            last_error = e
            error_msg = str(e)

            if "429" in error_msg and attempt < GEMINI_MAX_RETRIES:
                logger.warning(
                    f"Gemini 429 (attempt {attempt+1}/{GEMINI_MAX_RETRIES+1}) — "
                    f"retrying in {GEMINI_RETRY_DELAY}s..."
                )
                time.sleep(GEMINI_RETRY_DELAY)
            else:
                logger.error(f"Gemini fallback failed: {e}")
                raise e

    raise last_error


def get_json_response(
    messages: list,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> dict:
    """
    Helper that returns parsed JSON dicts.
    Used by agents that require structured JSON output.
    """
    try:
        raw_text = get_llm_response(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to get JSON response: {e}")
        return {}
