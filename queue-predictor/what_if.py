"""
what_if.py
----------
What-If simulation engine for QueueIQ.

Answers: "What would happen if we opened a new counter / redirected customers?"

Uses the actual queue dynamics model — NOT hard-coded numbers.

Available scenarios:
  - open_counter:       Add a new service counter, absorbing load from highest-loaded counter.
  - redirect_customers: Move a fraction of customers from overloaded to relief counter.
  - increase_service:   Increase service rate by a configurable factor.
  - close_counter:      Remove a counter and redistribute its load.
"""

from config import NEW_COUNTER_ABSORPTION, REDIRECT_ABSORPTION
from queue_math import estimate_wait_time


def _impact_label(delta: float) -> str:
    """Format a delta as '+N' or '-N'."""
    return f"+{delta:.0f}" if delta > 0 else f"{delta:.0f}"


def scenario_open_counter(
    counters: list[dict],
    wait_times: list[float],
    service_rate: float,
    new_counter_name: str = "New Counter",
    absorption: float = NEW_COUNTER_ABSORPTION,
) -> dict:
    """
    Simulate opening one additional counter.

    The new counter absorbs `absorption` fraction of the most-loaded counter's queue.
    The original counter retains (1 - absorption) of its queue.

    Args:
        counters:      List of counter dicts with 'name', 'people_count'.
        wait_times:    Current estimated wait times (minutes), one per counter.
        service_rate:  Service rate (cust/min) for the new counter.
        new_counter_name: Display name for the hypothetical counter.
        absorption:    Fraction of load moved to the new counter (0–1).

    Returns:
        Scenario result dict with before/after comparison.
    """
    if not counters:
        return {"error": "No counters available."}

    # Find most-loaded counter
    max_idx = max(range(len(counters)), key=lambda i: counters[i].get("people_count", 0))
    max_counter = counters[max_idx]
    old_count   = max_counter["people_count"]
    old_wait    = wait_times[max_idx] if max_idx < len(wait_times) else None

    moved_count = int(round(old_count * absorption))
    new_count_source = old_count - moved_count
    new_counter_count = moved_count

    # Recalculate wait times
    new_wait_source  = estimate_wait_time(new_count_source, service_rate)
    new_wait_counter = estimate_wait_time(new_counter_count, service_rate)

    # Overall before/after
    total_before = sum(c.get("people_count", 0) for c in counters)
    total_after  = total_before  # same people, just redistributed

    avg_wait_before = (sum(w for w in wait_times if w) / len([w for w in wait_times if w])
                       if any(wait_times) else 0)
    all_new_waits = list(wait_times)
    all_new_waits[max_idx] = new_wait_source or 0
    all_new_waits.append(new_wait_counter or 0)
    avg_wait_after = sum(all_new_waits) / len(all_new_waits) if all_new_waits else 0

    wait_reduction = (old_wait or 0) - (new_wait_source or 0)

    return {
        "scenario":           "open_counter",
        "scenario_label":     f"Open {new_counter_name}",
        "target_counter":     max_counter["name"],
        "before_count":       old_count,
        "after_count":        new_count_source,
        "before_wait":        round(old_wait, 1) if old_wait else None,
        "after_wait":         round(new_wait_source, 1) if new_wait_source else None,
        "new_counter_count":  new_counter_count,
        "new_counter_wait":   round(new_wait_counter, 1) if new_wait_counter else None,
        "new_counter_name":   new_counter_name,
        "total_before":       total_before,
        "total_after":        total_after,
        "avg_wait_before":    round(avg_wait_before, 1),
        "avg_wait_after":     round(avg_wait_after, 1),
        "wait_reduction":     round(wait_reduction, 1),
        "people_moved":       moved_count,
        "summary": (
            f"Opening {new_counter_name} would move ~{moved_count} people from "
            f"{max_counter['name']} — reducing its queue from {old_count} to {new_count_source} "
            f"and wait from {old_wait:.0f} to {new_wait_source:.0f} min."
            if old_wait and new_wait_source else
            f"Opening {new_counter_name} would move ~{moved_count} people from "
            f"{max_counter['name']} — reducing queue from {old_count} to {new_count_source}."
        ),
    }


def scenario_redirect_customers(
    counters: list[dict],
    wait_times: list[float],
    service_rate: float,
    from_idx: int | None = None,
    to_idx:   int | None = None,
    fraction: float = REDIRECT_ABSORPTION,
) -> dict:
    """
    Simulate redirecting customers from overloaded to relief counter.

    Args:
        counters:    List of counter dicts.
        wait_times:  Current wait times per counter.
        service_rate: Service rate for both counters (simplified).
        from_idx:    Index of source counter (default: highest load).
        to_idx:      Index of relief counter (default: lowest load).
        fraction:    Fraction of source queue to redirect.
    """
    if len(counters) < 2:
        return {"error": "Need at least 2 counters for redirect scenario."}

    if from_idx is None:
        from_idx = max(range(len(counters)), key=lambda i: counters[i].get("people_count", 0))
    if to_idx is None:
        to_idx = min(range(len(counters)),
                     key=lambda i: counters[i].get("people_count", 0) if i != from_idx else 999)

    from_c   = counters[from_idx]
    to_c     = counters[to_idx]
    moved    = int(round(from_c["people_count"] * fraction))

    new_from = from_c["people_count"] - moved
    new_to   = to_c["people_count"]   + moved

    from_wait_before = wait_times[from_idx] if from_idx < len(wait_times) else None
    to_wait_before   = wait_times[to_idx]   if to_idx   < len(wait_times) else None

    from_wait_after  = estimate_wait_time(new_from, service_rate)
    to_wait_after    = estimate_wait_time(new_to,   service_rate)

    return {
        "scenario":          "redirect",
        "scenario_label":    f"Redirect: {from_c['name']} → {to_c['name']}",
        "from_counter":      from_c["name"],
        "to_counter":        to_c["name"],
        "before_from":       from_c["people_count"],
        "after_from":        new_from,
        "before_to":         to_c["people_count"],
        "after_to":          new_to,
        "wait_from_before":  round(from_wait_before, 1) if from_wait_before else None,
        "wait_from_after":   round(from_wait_after, 1)  if from_wait_after  else None,
        "wait_to_before":    round(to_wait_before, 1)   if to_wait_before   else None,
        "wait_to_after":     round(to_wait_after, 1)    if to_wait_after    else None,
        "people_moved":      moved,
        "summary": (
            f"Redirecting ~{moved} customers from {from_c['name']} to {to_c['name']}: "
            f"source queue {from_c['people_count']} → {new_from} "
            f"(wait: {from_wait_before:.0f} → {from_wait_after:.0f} min)."
            if from_wait_before and from_wait_after else
            f"Redirecting ~{moved} customers: {from_c['name']} {from_c['people_count']} → {new_from}."
        ),
    }


def scenario_increase_service(
    counters: list[dict],
    wait_times: list[float],
    service_rate: float,
    factor: float = 1.5,
    target_idx: int | None = None,
) -> dict:
    """
    Simulate increasing service rate (adding staff / faster processing).

    Args:
        factor:  Multiplier on current service_rate (e.g., 1.5 = 50% faster).
        target_idx: Counter index to improve (default: highest load).
    """
    if not counters:
        return {"error": "No counters."}

    if target_idx is None:
        target_idx = max(range(len(counters)), key=lambda i: counters[i].get("people_count", 0))

    tc = counters[target_idx]
    old_wait  = wait_times[target_idx] if target_idx < len(wait_times) else None
    new_sr    = service_rate * factor
    new_wait  = estimate_wait_time(tc["people_count"], new_sr)

    return {
        "scenario":       "increase_service",
        "scenario_label": f"Increase service rate ×{factor} on {tc['name']}",
        "target_counter": tc["name"],
        "before_wait":    round(old_wait, 1)  if old_wait  else None,
        "after_wait":     round(new_wait, 1)  if new_wait  else None,
        "before_sr":      service_rate,
        "after_sr":       round(new_sr, 2),
        "people_count":   tc["people_count"],
        "summary": (
            f"Increasing service speed by {(factor-1)*100:.0f}% on {tc['name']}: "
            f"wait {old_wait:.0f} → {new_wait:.0f} min."
            if old_wait and new_wait else
            f"Service rate increased to {new_sr:.1f}/min on {tc['name']}."
        ),
    }
