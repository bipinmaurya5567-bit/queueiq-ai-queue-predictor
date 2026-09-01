"""
groq_alerts.py
--------------
Natural-language alert generation via the Groq API (llama-3.3-70b-versatile).

Pipeline:
  1. Collect the current system state (counter readings, predictions, recommendation).
  2. Build a compact prompt and call Groq's chat completion API.
  3. Return the model's short alert sentence.
  4. FALLBACK: if Groq call fails (network, timeout, quota), fall back to a
     deterministic template built from the raw numbers — demo never breaks.

Authentication:
  Set GROQ_API_KEY in a .env file (see .env.example).
  Loaded automatically via python-dotenv.
"""

import os
import time
import threading
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads .env from current directory

from config import GROQ_MODEL, GROQ_TIMEOUT_SEC, GROQ_MAX_TOKENS

MAX_TOKENS = GROQ_MAX_TOKENS



def _build_prompt(
    counter_states: list[dict],
    predictions:    list[dict],
    recommendation: dict,
) -> str:
    """Build a compact system-state prompt for the LLM."""
    lines = ["Current queue status:"]
    for cs, pred in zip(counter_states, predictions):
        wait_str = f"{cs.get('wait_time', '?')} min wait" if cs.get("wait_time") else "wait unknown"
        pred_str = (
            f"predicted {pred['predicted_count']} people in {pred['predicted_in_min']:.0f} min"
            if pred.get("predicted_count") is not None
            else "prediction unavailable"
        )
        lines.append(f"  - {cs['name']}: {cs['people_count']} people, {wait_str}, {pred_str}")

    lines.append(f"\nSystem recommendation: {recommendation['message']}")
    lines.append(
        "\nWrite ONE concise alert sentence (max 30 words) for the operations manager. "
        "Be direct and specific. Include the counter name, numbers, and recommended action."
    )
    return "\n".join(lines)


def _fallback_alert(
    counter_states: list[dict],
    predictions:    list[dict],
    recommendation: dict,
) -> str:
    """
    Template-based fallback alert — runs when Groq API is unavailable.
    Always deterministic, zero network dependency.
    """
    # Find the most overloaded counter
    worst = max(counter_states, key=lambda c: c.get("people_count", 0))
    worst_pred = next(
        (p for p in predictions if p.get("predicted_count") is not None
         and counter_states.index(worst) < len(predictions)),
        None
    )

    action_map = {
        "redirect":      "redirect customers to a less busy counter",
        "open_counter":  "open an additional counter immediately",
        "no_action":     "maintain current operations",
    }
    action_str = action_map.get(recommendation["action"], recommendation["message"])

    if worst_pred and worst_pred.get("predicted_count"):
        return (
            f"[Auto-alert] {worst['name']} currently has {worst['people_count']} people "
            f"and is projected to reach {worst_pred['predicted_count']} in "
            f"{int(worst_pred['predicted_in_min'])} minutes — {action_str}."
        )
    else:
        return (
            f"[Auto-alert] {worst['name']} currently has {worst['people_count']} people "
            f"— {action_str}."
        )


def generate_alert(
    counter_states: list[dict],
    predictions:    list[dict],
    recommendation: dict,
) -> dict:
    """
    Generate a natural-language alert for the operations dashboard.

    Args:
        counter_states: List of dicts with keys 'name', 'people_count',
                         optionally 'wait_time'.
        predictions:    List of prediction dicts from predictor.py.
        recommendation: Result dict from recommender.py.

    Returns:
        Dict with:
            alert_text  (str)  — the alert sentence (Groq or fallback)
            source      (str)  — 'groq' | 'fallback'
            error       (str or None) — error message if fallback was triggered
    """
    # Try: 1) environment variable (.env / system), 2) Streamlit Cloud secrets
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        try:
            import streamlit as _st
            api_key = _st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass


    if not api_key or api_key.startswith("your_"):
        # No valid key — go straight to fallback (offline mode)
        return {
            "alert_text": _fallback_alert(counter_states, predictions, recommendation),
            "source":     "fallback",
            "error":      "GROQ_API_KEY not configured — using template alert.",
        }

    prompt = _build_prompt(counter_states, predictions, recommendation)

    # ── Thread-based hard timeout ───────────────────────────────────────────
    # The Groq SDK's own `timeout=` parameter relies on httpx, which sometimes
    # fails to interrupt on Windows-level socket errors (wsarecv hang).
    # Running the API call in a daemon thread and joining with a deadline gives
    # us a rock-solid fallback that fires even on total network failures.
    result_holder = {}   # shared dict between threads

    def _groq_call():
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant for a queue management system at public service centers. "
                            "Generate concise, actionable alerts for operations managers. "
                            "Always mention specific counter names and numbers."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.3,
            )
            result_holder["text"] = (
                response.choices[0].message.content
                .strip()
                .replace("\u202f", " ")   # narrow NBSP → regular space (Windows safe)
                .replace("\u00a0", " ")   # regular NBSP → regular space
            )
        except Exception as exc:
            result_holder["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    worker = threading.Thread(target=_groq_call, daemon=True)
    worker.start()
    worker.join(timeout=GROQ_TIMEOUT_SEC)   # hard wall-clock deadline

    if worker.is_alive():
        # Thread still running → network completely stalled (e.g., wsarecv hang)
        return {
            "alert_text": _fallback_alert(counter_states, predictions, recommendation),
            "source":     "fallback",
            "error":      f"Groq API timed out after {GROQ_TIMEOUT_SEC}s (network unreachable). Using template alert.",
        }

    if "error" in result_holder:
        return {
            "alert_text": _fallback_alert(counter_states, predictions, recommendation),
            "source":     "fallback",
            "error":      f"Groq API error: {result_holder['error']}",
        }

    # Success — return Groq-generated text
    return {
        "alert_text": result_holder.get("text", "No alert generated."),
        "source":     "groq",
        "error":      None,
    }


# ──────────────────────────────────────────────
# Quick sanity test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Groq Alerts Sanity Test ===\n")

    mock_states = [
        {"name": "Counter 1 (Savings)", "people_count": 18, "wait_time": 32},
        {"name": "Counter 2 (Current)", "people_count": 4,  "wait_time": 7},
        {"name": "Counter 3 (Loans)",   "people_count": 21, "wait_time": 38},
    ]
    mock_preds = [
        {"predicted_count": 35, "predicted_in_min": 20},
        {"predicted_count":  6, "predicted_in_min": 20},
        {"predicted_count": 40, "predicted_in_min": 20},
    ]
    mock_rec = {
        "action":  "redirect",
        "message": "Redirect customers from Counter 3 (Loans) to Counter 2 (Current).",
        "severity": "high",
    }

    result = generate_alert(mock_states, mock_preds, mock_rec)
    safe_alert = result['alert_text'].encode('cp1252', errors='replace').decode('cp1252')
    print(f"Source:     {result['source']}")
    print(f"Alert:      {safe_alert}")
    if result["error"]:
        safe_err = result['error'].encode('cp1252', errors='replace').decode('cp1252')
        print(f"Note:       {safe_err}")
