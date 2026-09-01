"""
recommender.py
--------------
Smart, multi-signal recommendation engine for QueueIQ.

Evaluates ALL counters as a system (not in isolation) and recommends:
  1. Redirect customers  — feasible when relief counter exists
  2. Open new counter    — when system-wide load is high
  3. Increase service    — when load is high but counter opening isn't feasible
  4. Monitor            — borderline, watch closely
  5. No action          — all counters normal

Every recommendation includes:
  - action (machine-readable)
  - title  (display headline)
  - message (why this action)
  - impact  (what-if result from the actual queue model)
  - severity (low | medium | high | critical)
  - arrival_vs_service (rate comparison explanation)
"""

from config import (
    REDIRECT_WAIT_THRESHOLD,
    REDIRECT_WAIT_DIFF_MIN,
    OPEN_COUNTER_AVG_WAIT,
    REDIRECT_ABSORPTION,
)
from queue_math import estimate_wait_time


# ── Internal helpers ────────────────────────────────────────────────────────

def _make_result(action, title, message, from_counter, to_counter, severity,
                 impact=None, arrival_vs_service=None):
    return {
        "action":              action,
        "title":               title,
        "message":             message,
        "from_counter":        from_counter,
        "to_counter":          to_counter,
        "severity":            severity,
        "impact":              impact or {},
        "arrival_vs_service":  arrival_vs_service or "",
    }


def _arrival_vs_service_msg(counters_current, wait_times):
    """Generate a short explanation comparing arrival rate vs service capacity."""
    if not counters_current:
        return ""
    # Estimate implicit arrival rate from wait time: λ ≈ queue / wait (M/M/1 inversion)
    # This is approximate but directionally correct.
    overloaded = [
        c["name"] for i, c in enumerate(counters_current)
        if i < len(wait_times) and wait_times[i] is not None and wait_times[i] >= REDIRECT_WAIT_THRESHOLD
    ]
    if overloaded:
        names = ", ".join(overloaded)
        return (
            f"Arrival rate is exceeding service capacity at {names}. "
            "Customers are accumulating faster than they can be served."
        )
    return "Arrival rate is within service capacity across all counters."


def _what_if_redirect(counters_current, wait_times, service_rate, from_idx, to_idx):
    """Quick inline what-if for redirect impact (avoids circular import of what_if.py)."""
    if from_idx is None or to_idx is None:
        return {}
    fc = counters_current[from_idx]
    tc = counters_current[to_idx]
    moved       = int(round(fc["people_count"] * REDIRECT_ABSORPTION))
    new_from    = fc["people_count"] - moved
    new_to      = tc["people_count"] + moved
    new_from_w  = estimate_wait_time(new_from, service_rate)
    new_to_w    = estimate_wait_time(new_to,   service_rate)
    old_from_w  = wait_times[from_idx] if from_idx < len(wait_times) else None
    return {
        "people_moved":   moved,
        "before_wait":    round(old_from_w, 1) if old_from_w else None,
        "after_wait":     round(new_from_w, 1) if new_from_w else None,
        "target_before":  tc["people_count"],
        "target_after":   new_to,
        "target_wait":    round(new_to_w, 1)   if new_to_w   else None,
    }


def _what_if_open_counter(counters_current, wait_times, service_rate):
    """Quick inline what-if for opening a new counter."""
    if not counters_current:
        return {}
    max_idx    = max(range(len(counters_current)),
                     key=lambda i: counters_current[i].get("people_count", 0))
    fc         = counters_current[max_idx]
    moved      = int(round(fc["people_count"] * 0.45))
    new_count  = fc["people_count"] - moved
    new_wait   = estimate_wait_time(new_count, service_rate)
    old_wait   = wait_times[max_idx] if max_idx < len(wait_times) else None
    return {
        "people_moved":   moved,
        "before_wait":    round(old_wait, 1)  if old_wait  else None,
        "after_wait":     round(new_wait, 1)  if new_wait  else None,
        "new_count":      new_count,
        "source_counter": fc["name"],
    }


# ── Main recommendation function ────────────────────────────────────────────

def recommend_action(
    counters_current:   list[dict],
    counters_predicted: list[dict],
    wait_times:         list[float | None],
    service_rate:       float = 1.0,
) -> dict:
    """
    Recommend the best operational action for the facility.

    Args:
        counters_current:   [{counter_id, name, people_count}, ...]
        counters_predicted: [{counter_id, predicted_count, predicted_in_min, alert}, ...]
        wait_times:         [float | None, ...]  — estimated wait per counter
        service_rate:       float — customers/min (for what-if calculation)

    Returns:
        Dict with: action, title, message, from_counter, to_counter,
                   severity, impact, arrival_vs_service
    """
    n = len(counters_current)
    if n == 0:
        return _make_result("no_action", "No action needed", "No counters to evaluate.",
                            None, None, "low")

    avs_msg = _arrival_vs_service_msg(counters_current, wait_times)

    # ── Classify each counter ─────────────────────────────────────────────
    overloaded_idx = []
    relief_idx     = []

    for i in range(n):
        wait = wait_times[i] if i < len(wait_times) else None
        pred = counters_predicted[i].get("predicted_count") if i < len(counters_predicted) else None
        alert = counters_predicted[i].get("alert", False) if i < len(counters_predicted) else False

        is_overloaded = (
            (wait is not None and wait >= REDIRECT_WAIT_THRESHOLD) or
            (pred is not None and pred >= REDIRECT_WAIT_THRESHOLD) or
            alert
        )
        is_relief = (
            wait is not None and
            wait <= (REDIRECT_WAIT_THRESHOLD - REDIRECT_WAIT_DIFF_MIN)
        )

        if is_overloaded:
            overloaded_idx.append(i)
        if is_relief:
            relief_idx.append(i)

    # ── Find best redirect pair ───────────────────────────────────────────
    best_from = None
    best_to   = None
    best_gap  = 0

    for fi in overloaded_idx:
        fw = wait_times[fi] if fi < len(wait_times) and wait_times[fi] else 0
        for ti in relief_idx:
            if ti == fi:
                continue
            tw = wait_times[ti] if ti < len(wait_times) and wait_times[ti] else 0
            gap = fw - tw
            if gap > best_gap:
                best_gap  = gap
                best_from = fi
                best_to   = ti

    if best_from is not None and best_to is not None:
        fc      = counters_current[best_from]
        tc      = counters_current[best_to]
        fw      = wait_times[best_from] or 0
        tw      = wait_times[best_to]   or 0
        impact  = _what_if_redirect(counters_current, wait_times, service_rate, best_from, best_to)
        impact_str = ""
        if impact.get("before_wait") and impact.get("after_wait"):
            impact_str = (
                f" Moving ~{impact['people_moved']} customers would reduce "
                f"{fc['name']} wait from {impact['before_wait']:.0f} → {impact['after_wait']:.0f} min."
            )
        msg = (
            f"Redirect customers from {fc['name']} (wait: {fw:.0f} min) "
            f"to {tc['name']} (wait: {tw:.0f} min) — difference of {best_gap:.0f} min.{impact_str}"
        )
        severity = "critical" if fw >= REDIRECT_WAIT_THRESHOLD * 1.5 else "high"
        return _make_result(
            "redirect",
            f"Redirect to {tc['name']}",
            msg,
            fc["counter_id"], tc["counter_id"],
            severity, impact, avs_msg,
        )

    # ── Check system-level load → open new counter ────────────────────────
    valid_waits = [w for w in wait_times if w is not None]
    if valid_waits:
        avg_wait = sum(valid_waits) / len(valid_waits)
        if avg_wait >= OPEN_COUNTER_AVG_WAIT or len(overloaded_idx) >= max(1, n // 2):
            impact = _what_if_open_counter(counters_current, wait_times, service_rate)
            impact_str = ""
            if impact.get("before_wait") and impact.get("after_wait"):
                impact_str = (
                    f" Opening a new counter could move ~{impact['people_moved']} "
                    f"customers from {impact['source_counter']}, "
                    f"reducing wait {impact['before_wait']:.0f} → {impact['after_wait']:.0f} min."
                )
            msg = (
                f"Overall system load is high (avg wait: {avg_wait:.0f} min "
                f"across {n} counters). Opening an additional counter is recommended.{impact_str}"
            )
            return _make_result(
                "open_counter",
                "Open additional counter",
                msg, None, None, "high", impact, avs_msg,
            )

    # ── Borderline: any overloaded counter but no relief ──────────────────
    if overloaded_idx and not relief_idx:
        fc      = counters_current[overloaded_idx[0]]
        fw      = wait_times[overloaded_idx[0]] or 0
        msg = (
            f"{fc['name']} is overloaded (wait: {fw:.0f} min) but no lighter counter "
            "is available for redirect. Consider opening a new counter or adding staff."
        )
        return _make_result(
            "open_counter", "Open counter or add staff",
            msg, None, None, "medium", {}, avs_msg,
        )

    # ── Watchlist ─────────────────────────────────────────────────────────
    # Any counter with predicted alert (but not yet at threshold)
    alert_preds = [
        i for i, p in enumerate(counters_predicted)
        if p.get("alert") and i not in overloaded_idx
    ]
    if alert_preds:
        names = ", ".join(counters_current[i]["name"] for i in alert_preds)
        pred_counts = ", ".join(
            str(counters_predicted[i].get("predicted_count", "?"))
            for i in alert_preds
        )
        msg = (
            f"Monitor closely: {names} are predicted to exceed threshold "
            f"({pred_counts} people). Current queues are manageable but trending up."
        )
        return _make_result(
            "monitor", "Monitor — trending up",
            msg, None, None, "medium", {}, avs_msg,
        )

    # ── All good ──────────────────────────────────────────────────────────
    avg_wait_display = (sum(valid_waits) / len(valid_waits)) if valid_waits else 0
    msg = (
        f"All {n} counters are within normal range (avg wait: {avg_wait_display:.0f} min). "
        "No immediate action needed."
    )
    return _make_result("no_action", "No action needed", msg, None, None, "low", {}, avs_msg)


# ── Quick sanity test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Recommender Tests ===\n")

    # Scenario 1: Counter 1 overloaded, Counter 2 light
    curr  = [
        {"counter_id": 1, "name": "Counter 1", "people_count": 27},
        {"counter_id": 2, "name": "Counter 2", "people_count":  6},
        {"counter_id": 3, "name": "Counter 3", "people_count": 14},
    ]
    preds = [
        {"counter_id": 1, "predicted_count": 41, "predicted_in_min": 20, "alert": True},
        {"counter_id": 2, "predicted_count":  7, "predicted_in_min": 20, "alert": False},
        {"counter_id": 3, "predicted_count": 17, "predicted_in_min": 20, "alert": False},
    ]
    waits = [27.0, 6.0, 14.0]
    rec = recommend_action(curr, preds, waits, service_rate=1.0)
    print(f"S1 (overloaded + relief): action={rec['action']}, severity={rec['severity']}")
    print(f"   Title: {rec['title']}")
    print(f"   Msg:   {rec['message'][:100]}...")
    print(f"   Impact: {rec['impact']}")
    assert rec["action"] == "redirect", f"Expected redirect, got {rec['action']}"
    print("   PASS")

    # Scenario 2: All overloaded
    curr2  = [{"counter_id": i+1, "name": f"C{i+1}", "people_count": 25+i*3} for i in range(3)]
    preds2 = [{"counter_id": i+1, "predicted_count": 35+i*5, "alert": True} for i in range(3)]
    waits2 = [28.0, 31.0, 35.0]
    rec2 = recommend_action(curr2, preds2, waits2, service_rate=1.0)
    print(f"\nS2 (all overloaded): action={rec2['action']}")
    assert rec2["action"] in ("open_counter", "redirect"), f"Got {rec2['action']}"
    print("   PASS")

    # Scenario 3: All fine
    curr3  = [{"counter_id": i+1, "name": f"C{i+1}", "people_count": 5} for i in range(3)]
    preds3 = [{"counter_id": i+1, "predicted_count": 7, "alert": False} for i in range(3)]
    waits3 = [5.0, 4.0, 6.0]
    rec3 = recommend_action(curr3, preds3, waits3, service_rate=1.0)
    print(f"\nS3 (all fine): action={rec3['action']}")
    assert rec3["action"] == "no_action", f"Got {rec3['action']}"
    print("   PASS")

    print("\nAll recommender tests passed.")
