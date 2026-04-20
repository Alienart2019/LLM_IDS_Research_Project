"""
llm_explainer.py — optional LLM-backed natural-language explanations.

When ``IDS_USE_LLM`` is enabled, every non-allowlisted detection is
annotated with a short paragraph from the configured LLM endpoint (an
Ollama-compatible HTTP server by default). When it's disabled, a
deterministic fallback string is returned so the rest of the pipeline
always has a non-empty ``explanation`` field to display.

This module is intentionally defensive: any LLM failure (network, timeout,
bad JSON) is swallowed and a fallback string is returned, so a flaky LLM
cannot take the detection path down.
"""

import json

import requests

from app.config import USE_LLM, LLM_API_URL, LLM_MODEL
from app.schemas import LogEvent


def explain_event_with_llm(
    event: LogEvent,
    label: str,
    confidence: float,
    flags: list[str],
) -> str:
    """
    Return a one-line explanation of why ``event`` was given ``label``.

    Parameters
    ----------
    event:
        The original event being explained.
    label:
        Predicted class (``benign`` / ``suspicious`` / ``malicious``).
    confidence:
        Classifier probability for ``label``.
    flags:
        Rule flags that fired on the event.

    Returns
    -------
    str
        LLM-generated explanation if ``IDS_USE_LLM=true`` and the call
        succeeded; otherwise a deterministic fallback string.
    """
    if not USE_LLM:
        return (
            f"Predicted as {label} with confidence {confidence:.2f}. "
            f"Flags: {', '.join(flags) if flags else 'none'}."
        )

    prompt = f"""
You are a defensive intrusion detection analyst.

Explain this classification briefly.

Event:
{json.dumps({
    "timestamp": event.timestamp,
    "source_ip": event.source_ip,
    "hostname": event.hostname,
    "service": event.service,
    "message": event.message,
}, indent=2)}

Prediction:
label={label}
confidence={confidence:.2f}
flags={flags}

Respond with one concise explanation.
"""

    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}

    try:
        response = requests.post(LLM_API_URL, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as exc:
        # Never let a flaky LLM break the detection path.
        return (
            f"Predicted as {label} with confidence {confidence:.2f}. "
            f"LLM explanation unavailable: {exc}"
        )
