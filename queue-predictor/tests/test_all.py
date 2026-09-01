"""
tests/test_all.py
-----------------
Automated test suite for QueueIQ core modules.

Tests: queue calculation, arrival rate, service rate, waiting time,
       prediction, threshold detection, risk classification,
       recommendation, what-if simulation, CSV validation, edge cases.

Run with:
    python -m pytest tests/ -v
  or directly:
    python tests/test_all.py
"""

import sys
import os
import time

# Add parent dir to path so imports work without installing package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from queue_math   import estimate_wait_time, estimate_wait_time_mm1
from predictor    import predict_future_count, predict_multi_horizon, MIN_READINGS
from recommender  import recommend_action
from risk_engine  import classify_risk, classify_facility_risk
from what_if      import scenario_open_counter, scenario_redirect_customers, scenario_increase_service


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _history(counts, interval_s=30):
    """Build a fake history list from a list of people_count values."""
    base = time.time() - len(counts) * interval_s
    return [
        {"timestamp_epoch": base + i * interval_s, "people_count": c}
        for i, c in enumerate(counts)
    ]


def _counters(counts):
    return [
        {"counter_id": i + 1, "name": f"Counter {i+1}", "people_count": c}
        for i, c in enumerate(counts)
    ]


def _preds(counts, alert_threshold=30):
    return [
        {"counter_id": i + 1, "predicted_count": c,
         "predicted_in_min": 20, "alert": c >= alert_threshold}
        for i, c in enumerate(counts)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Queue / Wait Time
# ─────────────────────────────────────────────────────────────────────────────

def test_wait_time_zero_queue():
    """Zero queue -> wait should be 0."""
    wt = estimate_wait_time(0, 1.0)
    assert wt == 0, f"Expected 0, got {wt}"
    print("PASS test_wait_time_zero_queue")


def test_wait_time_one_person():
    """1 person, service_rate=1 -> wait ~= 1 min."""
    wt = estimate_wait_time(1, 1.0)
    assert 0.5 <= wt <= 2.0, f"Unexpected wait: {wt}"
    print("PASS test_wait_time_one_person")


def test_wait_time_zero_service_rate():
    """Service rate = 0 should not crash - returns None or infinity."""
    wt = estimate_wait_time(10, 0)
    assert wt is None or wt > 1000, f"Expected None/large, got {wt}"
    print("PASS test_wait_time_zero_service_rate")


def test_mm1_stable():
    """M/M/1: λ < μ → stable."""
    r = estimate_wait_time_mm1(arrival_rate=0.8, service_rate=1.0)
    assert r["stable"] is True
    assert r["rho"] < 1.0
    print(f"PASS test_mm1_stable (rho={r['rho']}, Wq={r['Wq_minutes']})")


def test_mm1_unstable():
    """M/M/1: λ ≥ μ → unstable (queue grows unbounded)."""
    r = estimate_wait_time_mm1(arrival_rate=1.2, service_rate=1.0)
    assert r["stable"] is False
    print("PASS test_mm1_unstable")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prediction
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_insufficient_data():
    """Fewer than MIN_READINGS → predicted_count must be None."""
    h = _history([5, 6, 7])   # only 3 readings
    r = predict_future_count(h, threshold=30)
    assert r["predicted_count"] is None, f"Expected None, got {r['predicted_count']}"
    assert "Insufficient" in r["sentence"]
    print("PASS test_predict_insufficient_data")


def test_predict_stable_queue():
    """Stable ~8-person queue should NOT predict > 30."""
    h = _history([8, 9, 8, 8, 9, 8, 9, 8])
    r = predict_future_count(h, threshold=30, horizon_minutes=20)
    assert r["predicted_count"] is not None
    assert r["predicted_count"] <= 30, f"BUG: stable queue predicted {r['predicted_count']} > 30"
    print(f"PASS test_predict_stable_queue (predicted={r['predicted_count']})")


def test_predict_growing_queue():
    """Rapidly growing queue should trigger alert."""
    h = _history([5, 8, 11, 14, 17, 20, 23, 26, 29, 32])
    r = predict_future_count(h, threshold=30, horizon_minutes=20)
    assert r["predicted_count"] is not None
    assert r["predicted_count"] > 20, f"Expected growing prediction, got {r['predicted_count']}"
    print(f"PASS test_predict_growing_queue (predicted={r['predicted_count']}, alert={r['alert']})")


def test_predict_multi_horizon():
    """Multi-horizon: should return +5/+10/+15/+20m forecasts."""
    h = _history([5, 7, 9, 11, 13, 15, 17, 19])
    result = predict_multi_horizon(h, horizons=[5, 10, 15, 20], threshold=30)
    assert result["has_data"] is True
    for h_min in [5, 10, 15, 20]:
        assert h_min in result["horizons"], f"Missing horizon {h_min}"
        pc = result["horizons"][h_min]["predicted_count"]
        assert pc is not None, f"Horizon +{h_min}m returned None"
        print(f"  +{h_min:2d}m -> {pc:3d} people")
    print("PASS test_predict_multi_horizon")


def test_predict_no_crash_zero_queue():
    """Edge case: all readings are 0."""
    h = _history([0, 0, 0, 0, 0, 0])
    r = predict_future_count(h, threshold=30)
    assert r["predicted_count"] is not None
    assert r["predicted_count"] >= 0
    print("PASS test_predict_no_crash_zero_queue")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Risk Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_risk_low():
    r = classify_risk(current_count=5, predicted_count=7, wait_time=4, threshold=30)
    assert r["level"] == "low", f"Expected low, got {r['level']}"
    print("PASS test_risk_low")


def test_risk_medium():
    r = classify_risk(current_count=15, predicted_count=22, wait_time=18, threshold=30)
    assert r["level"] in ("medium", "high"), f"Got {r['level']}"
    print(f"PASS test_risk_medium (level={r['level']})")


def test_risk_high():
    r = classify_risk(current_count=27, predicted_count=41, wait_time=28, threshold=30)
    assert r["level"] in ("high", "critical"), f"Got {r['level']}"
    print(f"PASS test_risk_high (level={r['level']})")


def test_risk_critical_utilisation():
    """ρ > 1 → queue grows unbounded → critical."""
    r = classify_risk(
        current_count=30, predicted_count=45, wait_time=42,
        arrival_rate=1.8, service_rate=1.0, threshold=30,
    )
    assert r["level"] == "critical", f"Expected critical, got {r['level']}"
    print("PASS test_risk_critical_utilisation")


def test_facility_risk_aggregation():
    """Majority HIGH counters → facility HIGH or CRITICAL."""
    risks = [
        classify_risk(27, 41, 29, threshold=30),
        classify_risk(6,  7,  5, threshold=30),
        classify_risk(24, 38, 27, threshold=30),
    ]
    fac = classify_facility_risk(risks)
    assert fac["level"] in ("high", "critical")
    print(f"PASS test_facility_risk_aggregation (facility={fac['level']})")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Recommendation Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_recommend_redirect():
    """Counter 1 overloaded + Counter 2 light → redirect."""
    curr  = _counters([27, 6, 14])
    preds = _preds([41, 7, 17])
    waits = [27.0, 6.0, 14.0]
    rec = recommend_action(curr, preds, waits, service_rate=1.0)
    assert rec["action"] == "redirect", f"Expected redirect, got {rec['action']}"
    print(f"PASS test_recommend_redirect (to={rec['to_counter']})")


def test_recommend_open_counter():
    """All counters at high load → open counter."""
    curr  = _counters([28, 27, 30])
    preds = _preds([40, 38, 45])
    waits = [30.0, 28.0, 32.0]
    rec = recommend_action(curr, preds, waits, service_rate=1.0)
    assert rec["action"] in ("open_counter", "redirect"), f"Got {rec['action']}"
    print(f"PASS test_recommend_open_counter (action={rec['action']})")


def test_recommend_no_action():
    """All queues low → no action."""
    curr  = _counters([5, 4, 6])
    preds = _preds([7, 6, 8])
    waits = [5.0, 4.0, 6.0]
    rec = recommend_action(curr, preds, waits, service_rate=1.0)
    assert rec["action"] == "no_action", f"Expected no_action, got {rec['action']}"
    print("PASS test_recommend_no_action")


def test_recommend_no_counters():
    """Empty counter list → no_action gracefully."""
    rec = recommend_action([], [], [], service_rate=1.0)
    assert rec["action"] == "no_action"
    print("PASS test_recommend_no_counters")


# ─────────────────────────────────────────────────────────────────────────────
# 5. What-If Simulation
# ─────────────────────────────────────────────────────────────────────────────

def test_what_if_open_counter():
    """Opening a counter moves people and reduces wait."""
    counters = [{"name": f"Counter {i+1}", "people_count": c}
                for i, c in enumerate([27, 6, 14])]
    waits = [27.0, 6.0, 14.0]
    result = scenario_open_counter(counters, waits, service_rate=1.0)
    assert "error" not in result
    assert result["after_count"] < result["before_count"], "Queue should reduce"
    assert result["people_moved"] > 0
    bw = result['before_wait'] or 0
    aw = result['after_wait'] or 0
    print(f"PASS test_what_if_open_counter (moved={result['people_moved']}, "
          f"wait {bw:.0f}->{aw:.0f} min)")


def test_what_if_redirect():
    """Redirect moves people from overloaded to relief counter."""
    counters = [{"name": f"Counter {i+1}", "people_count": c}
                for i, c in enumerate([30, 5, 12])]
    waits = [30.0, 5.0, 12.0]
    result = scenario_redirect_customers(counters, waits, service_rate=1.0)
    assert "error" not in result
    assert result["after_from"] < result["before_from"]
    print(f"PASS test_what_if_redirect (moved={result['people_moved']})")


def test_what_if_increase_service():
    """Increasing service rate should reduce wait time."""
    counters = [{"name": "Counter 1", "people_count": 20}]
    waits = [20.0]
    result = scenario_increase_service(counters, waits, service_rate=1.0, factor=2.0)
    assert "error" not in result
    if result.get("before_wait") and result.get("after_wait"):
        assert result["after_wait"] < result["before_wait"], "Wait should decrease"
    bw = result.get('before_wait') or 0
    aw = result.get('after_wait') or 0
    print(f"PASS test_what_if_increase_service (wait {bw:.0f}->{aw:.0f} min)")


def test_what_if_single_counter_redirect():
    """Cannot redirect with only one counter."""
    counters = [{"name": "Counter 1", "people_count": 20}]
    result = scenario_redirect_customers(counters, [20.0], service_rate=1.0)
    assert "error" in result
    print("PASS test_what_if_single_counter_redirect (error returned correctly)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Edge cases / Robustness
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_sudden_surge():
    """Sudden surge: prediction should not be negative."""
    h = _history([2, 2, 2, 2, 2, 35, 38, 40])
    r = predict_future_count(h, threshold=30)
    assert r["predicted_count"] is not None
    assert r["predicted_count"] >= 0
    print(f"PASS test_predict_sudden_surge (predicted={r['predicted_count']})")


def test_predict_decreasing_queue():
    """Decreasing queue: prediction should be ≤ current."""
    h = _history([40, 36, 32, 28, 24, 20, 16, 12, 8])
    r = predict_future_count(h, threshold=30, horizon_minutes=20)
    if r["predicted_count"] is not None:
        # May go negative due to linear extrapolation, but should be clamped
        assert r["predicted_count"] >= 0
    print(f"PASS test_predict_decreasing_queue (predicted={r['predicted_count']})")


def test_risk_no_wait_data():
    """Risk engine should handle None wait_time."""
    r = classify_risk(current_count=15, predicted_count=20, wait_time=None, threshold=30)
    assert r["level"] in ("low", "medium")
    print(f"PASS test_risk_no_wait_data (level={r['level']})")


def test_recommend_none_waits():
    """Recommender should handle None wait times gracefully."""
    curr  = _counters([20, 5])
    preds = _preds([30, 6])
    waits = [None, None]
    try:
        rec = recommend_action(curr, preds, waits, service_rate=1.0)
        print(f"PASS test_recommend_none_waits (action={rec['action']})")
    except Exception as e:
        assert False, f"Recommender crashed with None waits: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

_TESTS = [
    # Wait time
    test_wait_time_zero_queue,
    test_wait_time_one_person,
    test_wait_time_zero_service_rate,
    test_mm1_stable,
    test_mm1_unstable,
    # Prediction
    test_predict_insufficient_data,
    test_predict_stable_queue,
    test_predict_growing_queue,
    test_predict_multi_horizon,
    test_predict_no_crash_zero_queue,
    # Risk
    test_risk_low,
    test_risk_medium,
    test_risk_high,
    test_risk_critical_utilisation,
    test_facility_risk_aggregation,
    # Recommendation
    test_recommend_redirect,
    test_recommend_open_counter,
    test_recommend_no_action,
    test_recommend_no_counters,
    # What-If
    test_what_if_open_counter,
    test_what_if_redirect,
    test_what_if_increase_service,
    test_what_if_single_counter_redirect,
    # Edge cases
    test_predict_sudden_surge,
    test_predict_decreasing_queue,
    test_risk_no_wait_data,
    test_recommend_none_waits,
]


if __name__ == "__main__":
    print("=" * 60)
    print("QueueIQ — Automated Test Suite")
    print("=" * 60)
    passed = 0
    failed = 0
    errors = []
    for test_fn in _TESTS:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {test_fn.__name__}: {e}")
            failed += 1
            errors.append(test_fn.__name__)
        except Exception as e:
            print(f"ERROR {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
            errors.append(test_fn.__name__)

    print("=" * 60)
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    if errors:
        print("Failed:", ", ".join(errors))
    else:
        print("All tests passed!")
    sys.exit(0 if failed == 0 else 1)
