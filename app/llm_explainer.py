import json
import requests
from app.config import USE_LLM, LLM_API_URL, LLM_MODEL
from app.schemas import LogEvent


def explain_event_with_llm(event: LogEvent, label: str, confidence: float, flags: list[str]) -> str:
    if not USE_LLM:
        return f"Predicted as {label} with confidence {confidence:.2f}. Flags: {', '.join(flags) if flags else 'none'}."

    prompt = f"""
You are a defensive intrusion detection analyst.

Explain this classification briefly.

Event:
{json.dumps({
    "timestamp": event.timestamp,
    "source_ip": event.source_ip,
    "hostname": event.hostname,
    "service": event.service,
    "message": event.message
}, indent=2)}

Prediction:
label={label}
confidence={confidence:.2f}
flags={flags}

Respond with one concise explanation.
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(LLM_API_URL, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"Predicted as {label} with confidence {confidence:.2f}. LLM explanation unavailable: {e}"
