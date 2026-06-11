"""Thin LLM wrapper. Default backend is Groq (you already used it on your
LLM Evaluator project, so the stack stays familiar).

The agent accepts any callable of the form `llm_fn(messages, stop) -> str`.
That means you can swap models or inject a fake function in tests without
touching the agent code -- a small design choice that's worth pointing to
when you describe the project.
"""
from __future__ import annotations

import os
from typing import Optional

# Change this to any current Groq model id.
# Check the live list at https://console.groq.com/docs/models
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def groq_llm(
    messages: list[dict],
    stop: Optional[list[str]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    """Call Groq's chat completion endpoint and return the text content.

    `groq` is imported lazily so that tests using a fake llm_fn don't need
    the package or an API key installed.
    """
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stop=stop,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""
