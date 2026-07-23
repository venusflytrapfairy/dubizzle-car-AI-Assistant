"""
Thin wrapper around LiteLLM so the rest of the codebase doesn't need to know
which provider/model we're using. Swapping models (or providers) later is a
one-line change in config.py.
"""
import os

from backend.config import GEMINI_API_KEY, GEMINI_MODEL

if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

import litellm  # noqa: E402  (import after env var set)

litellm.drop_params = True  # silently ignore provider-unsupported params instead of erroring


def chat_completion(messages: list[dict], tools: list[dict] | None = None, **kwargs):
    """Full chat completion call, returns the raw LiteLLM response object so
    callers can inspect tool_calls, finish_reason, etc."""
    return litellm.completion(
        model=GEMINI_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto" if tools else None,
        temperature=0.4,
        **kwargs,
    )


def complete_text(prompt: str) -> str:
    """Simple single-turn text completion, used for background tasks like
    session summarization."""
    resp = litellm.completion(
        model=GEMINI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )
    return resp.choices[0].message.content or ""
