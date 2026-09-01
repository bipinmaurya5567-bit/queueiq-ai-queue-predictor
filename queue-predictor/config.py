"""
config.py
---------
Single source of truth for all QueueIQ thresholds and configuration.
Modify this file to tune the system without touching business logic.
"""

# ── Queue thresholds (people count) ────────────────────────────────────────
THRESHOLD_LOW      = 10   # below this = low load
THRESHOLD_MEDIUM   = 20   # below this = medium load
THRESHOLD_HIGH     = 30   # at or above = high load (configurable via UI)
THRESHOLD_CRITICAL = 45   # at or above = critical (override always)

# ── Wait time thresholds (minutes) ─────────────────────────────────────────
WAIT_LOW      =  8    # below this → low
WAIT_MEDIUM   = 15    # below this → medium
WAIT_HIGH     = 25    # at or above → high
WAIT_CRITICAL = 40    # at or above → critical

# ── Arrival/departure rate ──────────────────────────────────────────────────
# Utilisation ρ = arrival_rate / service_rate
UTILIZATION_HIGH     = 0.85   # ρ ≥ this → system stressed
UTILIZATION_CRITICAL = 0.95   # ρ ≥ this → system near-saturation

# ── Prediction ───────────────────────────────────────────────────────────────
MIN_READINGS_FOR_FORECAST = 5    # fewer readings → return None (insufficient data)
DEFAULT_HORIZON_MINUTES   = 20
FORECAST_HORIZONS         = [5, 10, 15, 20]   # multi-horizon outputs

# ── Simulator ────────────────────────────────────────────────────────────────
TICK_SECONDS   = 30    # how many real-world seconds each simulation tick represents
MAX_HISTORY    = 30    # rolling window of readings kept per counter

# ── Recommendation ───────────────────────────────────────────────────────────
REDIRECT_WAIT_DIFF_MIN   = 10   # redirect only if relief counter is ≥ this much lighter
OPEN_COUNTER_AVG_WAIT    = 20   # open new counter if avg system wait exceeds this
REDIRECT_WAIT_THRESHOLD  = 25   # counter wait ≥ this is considered "overloaded"

# ── What-If ──────────────────────────────────────────────────────────────────
# When a new counter is opened, what fraction of the overloaded counter's load
# does it absorb? Conservative estimate for honest demo.
NEW_COUNTER_ABSORPTION  = 0.45   # 45 % of highest-load counter goes to new counter
REDIRECT_ABSORPTION     = 0.40   # 40 % moves on redirect

# ── CV / YOLO ────────────────────────────────────────────────────────────────
YOLO_CONFIDENCE   = 0.35    # detection confidence threshold
YOLO_CLASS_PERSON = 0       # COCO class 0 = person
FRAME_SKIP        = 3       # process every Nth frame for speed
IOU_TRACK_THRESH  = 0.35    # IoU threshold for simple tracker match

# ── Groq ─────────────────────────────────────────────────────────────────────
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_TIMEOUT_SEC = 6
GROQ_MAX_TOKENS  = 150

# ── UI labels ────────────────────────────────────────────────────────────────
RISK_LABELS = {
    "low":      ("🟢", "Low",      "#10B981"),
    "medium":   ("🟡", "Medium",   "#F59E0B"),
    "high":     ("🔴", "High",     "#EF4444"),
    "critical": ("🚨", "Critical", "#DC2626"),
}

MODE_LABELS = {
    "simulation": "🔁 Simulation",
    "real_data":  "📂 Historical Data",
    "cv":         "📹 AI CCTV Analysis",
}
