"""
simulator.py
------------
Simulates queue counters at various location presets (Bank, Hospital, College, Railway).
Each counter tracks arrival and service rates; readings are generated per "tick".

Rolling history (last 30 readings) is kept in-memory as a list of dicts — no database.
"""

import numpy as np
from datetime import datetime, timedelta
import time

# ──────────────────────────────────────────────
# LOCATION PRESETS
# Keys: location_name → list of counter configs
#   Each counter: {name, arrival_rate (cust/min), service_rate (cust/min)}
# ──────────────────────────────────────────────
PRESETS = {
    "Bank": [
        {"name": "Counter 1 (Savings)",   "arrival_rate": 3.0,  "service_rate": 4.0},
        {"name": "Counter 2 (Current)",   "arrival_rate": 2.0,  "service_rate": 3.5},
        {"name": "Counter 3 (Loans)",     "arrival_rate": 1.5,  "service_rate": 2.5},
    ],
    "Hospital": [
        {"name": "Counter 1 (OPD)",       "arrival_rate": 5.0,  "service_rate": 4.5},
        {"name": "Counter 2 (Pharmacy)",  "arrival_rate": 4.0,  "service_rate": 5.0},
        {"name": "Counter 3 (Lab)",       "arrival_rate": 2.5,  "service_rate": 3.0},
    ],
    "College Office": [
        {"name": "Counter 1 (Admissions)","arrival_rate": 3.5,  "service_rate": 3.0},
        {"name": "Counter 2 (Fees)",      "arrival_rate": 4.5,  "service_rate": 4.0},
        {"name": "Counter 3 (Documents)", "arrival_rate": 2.0,  "service_rate": 2.5},
    ],
    "Railway Station": [
        {"name": "Counter 1 (Booking)",   "arrival_rate": 6.0,  "service_rate": 5.5},
        {"name": "Counter 2 (Inquiry)",   "arrival_rate": 4.0,  "service_rate": 5.0},
        {"name": "Counter 3 (Refunds)",   "arrival_rate": 2.0,  "service_rate": 3.0},
    ],
}

MAX_HISTORY = 30   # rolling window size


def build_counters_state(preset_name: str, n_counters: int = None) -> list[dict]:
    """
    Initialise the mutable state for each counter at a given preset location.

    Args:
        preset_name: One of the keys in PRESETS.
        n_counters:  How many counters to activate (default: all in preset).

    Returns:
        List of counter-state dicts, each with keys:
            id, name, arrival_rate, service_rate, people_count, history, sim_start_time
    """
    template = PRESETS.get(preset_name, PRESETS["Bank"])
    if n_counters is not None:
        # Repeat/trim template to match requested count
        template = (template * ((n_counters // len(template)) + 1))[:n_counters]

    sim_start = time.time()
    counters = []
    for idx, cfg in enumerate(template):
        counter = {
            "id": idx + 1,
            "name": cfg["name"] if n_counters is None or idx < len(PRESETS.get(preset_name, [])) else f"Counter {idx+1}",
            "arrival_rate": cfg["arrival_rate"],
            "service_rate": cfg["service_rate"],
            "people_count": max(0, int(np.random.poisson(cfg["arrival_rate"] * 2))),  # seed with some initial people
            "history": [],      # will be filled by generate_reading()
            "sim_start_time": sim_start,
        }
        counters.append(counter)
    return counters


def generate_reading(counters_state: list[dict], tick_seconds: float = 30.0) -> list[dict]:
    """
    Advance the simulation by one tick for all counters.

    Per tick (representing `tick_seconds` of real-world time):
      - New arrivals drawn from Poisson(arrival_rate * tick_minutes)
      - Departures drawn from Poisson(service_rate * tick_minutes), capped at current queue
      - people_count updated: += arrivals - departures (floored at 0)
      - Reading appended to rolling history (max MAX_HISTORY entries)

    Args:
        counters_state: List of counter dicts (mutated in-place).
        tick_seconds:   Simulated time per tick in seconds.

    Returns:
        List of reading dicts: [{counter_id, name, people_count, timestamp_epoch, timestamp_str}, …]
    """
    tick_minutes = tick_seconds / 60.0
    readings = []
    now_epoch = time.time()
    now_str = datetime.now().strftime("%H:%M:%S")

    for counter in counters_state:
        arrivals   = np.random.poisson(counter["arrival_rate"] * tick_minutes)
        departures = np.random.poisson(counter["service_rate"] * tick_minutes)
        departures = min(departures, counter["people_count"])  # can't serve more than waiting

        counter["people_count"] = max(0, counter["people_count"] + arrivals - departures)

        reading = {
            "counter_id":       counter["id"],
            "name":             counter["name"],
            "people_count":     counter["people_count"],
            "timestamp_epoch":  now_epoch,
            "timestamp_str":    now_str,
        }

        # Append to rolling history, trim to MAX_HISTORY
        counter["history"].append(reading)
        if len(counter["history"]) > MAX_HISTORY:
            counter["history"] = counter["history"][-MAX_HISTORY:]

        readings.append(reading)

    return readings


# ──────────────────────────────────────────────
# Quick sanity test (run directly)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Simulator Sanity Test ===\n")
    for preset_name in PRESETS:
        print(f"Preset: {preset_name}")
        state = build_counters_state(preset_name)
        for _ in range(5):
            readings = generate_reading(state, tick_seconds=30)
        for r in readings:
            print(f"  {r['name']}: {r['people_count']} people  @ {r['timestamp_str']}")
        print()
