"""
risk_engine.py
--------------
Dedicated risk/congestion classification engine for QueueIQ.

Inputs:  current queue, predicted queue, arrival rate, service rate,
         wait time, threshold, utilisation.

Output:  LOW | MEDIUM | HIGH | CRITICAL
         + structured reason for dashboard display.

Design principle: all threshold logic lives HERE, not scattered in app.py.
"""

from config import (
    THRESHOLD_LOW, THRESHOLD_MEDIUM, THRESHOLD_HIGH, THRESHOLD_CRITICAL,
    WAIT_LOW, WAIT_MEDIUM, WAIT_HIGH, WAIT_CRITICAL,
    UTILIZATION_HIGH, UTILIZATION_CRITICAL,
)


def classify_risk(
    current_count: int,
    predicted_count: int | None,
    wait_time: float | None,
    arrival_rate: float | None = None,
    service_rate: float | None = None,
    threshold: int = THRESHOLD_HIGH,
) -> dict:
    """
    Classify the congestion risk for a SINGLE counter.

    Returns:
        {
            "level":   str  — "low" | "medium" | "high" | "critical"
            "label":   str  — human-readable level name
            "reasons": list[str]  — why this level was chosen
            "utilization": float | None  — ρ = λ/μ if rates available
        }
    """
    reasons = []
    scores  = []   # 0=low, 1=medium, 2=high, 3=critical; we take the max

    # ── Check 1: current queue vs threshold ──────────────────────────────────
    if current_count >= THRESHOLD_CRITICAL:
        scores.append(3)
        reasons.append(f"Queue ({current_count}) is severely overloaded (≥{THRESHOLD_CRITICAL}).")
    elif current_count >= threshold:
        scores.append(2)
        reasons.append(f"Queue ({current_count}) exceeds alert threshold ({threshold}).")
    elif current_count >= THRESHOLD_MEDIUM:
        scores.append(1)
        reasons.append(f"Queue ({current_count}) is building up.")
    else:
        scores.append(0)

    # ── Check 2: predicted queue ─────────────────────────────────────────────
    if predicted_count is not None:
        if predicted_count >= THRESHOLD_CRITICAL:
            scores.append(3)
            reasons.append(f"Predicted to reach {predicted_count} — critical overload imminent.")
        elif predicted_count >= threshold:
            scores.append(2)
            reasons.append(f"Predicted {predicted_count} will exceed threshold ({threshold}).")
        elif predicted_count >= THRESHOLD_MEDIUM:
            scores.append(1)
            reasons.append(f"Predicted queue of {predicted_count} is above medium range.")

    # ── Check 3: wait time ───────────────────────────────────────────────────
    if wait_time is not None:
        if wait_time >= WAIT_CRITICAL:
            scores.append(3)
            reasons.append(f"Wait time ({wait_time:.0f} min) is critically high.")
        elif wait_time >= WAIT_HIGH:
            scores.append(2)
            reasons.append(f"Wait time ({wait_time:.0f} min) is high.")
        elif wait_time >= WAIT_MEDIUM:
            scores.append(1)
            reasons.append(f"Wait time ({wait_time:.0f} min) is moderate.")

    # ── Check 4: utilisation ρ ───────────────────────────────────────────────
    utilization = None
    if arrival_rate is not None and service_rate is not None and service_rate > 0:
        utilization = arrival_rate / service_rate
        if utilization >= UTILIZATION_CRITICAL:
            scores.append(3)
            reasons.append(
                f"Utilisation ρ={utilization:.2f}: arrivals nearly saturating capacity."
            )
        elif utilization >= UTILIZATION_HIGH:
            scores.append(2)
            reasons.append(
                f"Utilisation ρ={utilization:.2f}: system is stressed."
            )
        elif utilization > 1.0:
            scores.append(3)
            reasons.append(
                f"Arrival rate ({arrival_rate:.2f}/min) exceeds service capacity ({service_rate:.2f}/min) — queue will grow indefinitely."
            )

    # ── Determine final level ─────────────────────────────────────────────────
    max_score = max(scores) if scores else 0
    level = ["low", "medium", "high", "critical"][max_score]
    label = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}[level]

    if not reasons:
        reasons.append("Queue is within normal operating range.")

    return {
        "level":       level,
        "label":       label,
        "reasons":     reasons,
        "utilization": round(utilization, 3) if utilization is not None else None,
    }


def classify_facility_risk(counter_risks: list[dict]) -> dict:
    """
    Aggregate individual counter risks into a single facility-level risk.

    Args:
        counter_risks: List of results from classify_risk(), one per counter.

    Returns:
        {
            "level":    str  — overall facility risk level
            "label":    str
            "summary":  str  — one-sentence summary
            "n_high":   int  — number of high/critical counters
        }
    """
    if not counter_risks:
        return {"level": "low", "label": "Low", "summary": "No counters to evaluate.", "n_high": 0}

    score_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    scores = [score_map[r["level"]] for r in counter_risks]

    max_score   = max(scores)
    avg_score   = sum(scores) / len(scores)
    n_high      = sum(1 for s in scores if s >= 2)
    n_counters  = len(counter_risks)

    # Facility is critical if ANY counter is critical,
    # or majority are high.
    if max_score == 3:
        facility_score = 3
    elif max_score == 2 and n_high >= max(1, n_counters // 2):
        facility_score = 2
    elif avg_score >= 1.5:
        facility_score = 2
    elif avg_score >= 0.75:
        facility_score = 1
    else:
        facility_score = 0

    level = ["low", "medium", "high", "critical"][facility_score]
    label = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}[level]

    if level == "critical":
        summary = f"CRITICAL: {n_high}/{n_counters} counters are severely overloaded."
    elif level == "high":
        summary = f"HIGH: {n_high}/{n_counters} counters are above threshold — intervention required."
    elif level == "medium":
        summary = f"MEDIUM: Queue building up — monitor closely."
    else:
        summary = f"LOW: All {n_counters} counters are operating normally."

    return {"level": level, "label": label, "summary": summary, "n_high": n_high}
