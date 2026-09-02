"""
app.py  ──  QueueIQ · Queue Intelligence Platform
==================================================
Three data modes — all render through a single shared dashboard:

  Simulation   Poisson-based multi-counter simulation
  Upload       CSV → predictor / recommender / risk engine
  Camera       YOLOv8n person detection on images/video

Architecture
  • Single-file Streamlit app
  • CSS design tokens injected at boot; swapped per theme at runtime
  • Altair charts with per-render theme configuration (no dark-chart-on-light-page)
  • All business logic in external modules — zero logic in this file
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from config      import FORECAST_HORIZONS, RISK_LABELS
from groq_alerts import generate_alert
from predictor   import predict_future_count, predict_multi_horizon, MIN_READINGS
from queue_math  import estimate_wait_time, estimate_wait_time_mm1
from recommender import recommend_action
from risk_engine import classify_risk, classify_facility_risk
from simulator   import build_counters_state, generate_reading, PRESETS
from what_if     import (
    scenario_increase_service,
    scenario_open_counter,
    scenario_redirect_customers,
)

# ── Camera — optional; fails gracefully ───────────────────────────────────────
_CV_AVAILABLE = False
try:
    import cv_detector as _cvd
    _CV_AVAILABLE = _cvd._YOLO_AVAILABLE
except Exception:
    pass


@st.cache_resource(show_spinner=False)
def _get_yolo_model():
    if not _CV_AVAILABLE:
        return None
    try:
        return _cvd.load_yolo_model()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QueueIQ",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%232F81F7'/><text x='16' y='22' font-family='Inter,sans-serif' font-weight='700' font-size='18' fill='white' text-anchor='middle'>Q</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
COUNTER_COLORS = ["#2F81F7", "#3FB950", "#D29922", "#F85149", "#A371F7", "#58A6FF"]
ADMIN_PASSWORD = "queueiq2024"

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM BOOT
# Semantic CSS tokens only — no component styles here.
# All surfaces, text, borders use var(--token).
# inject_theme() swaps :root tokens at runtime.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════════════
   SEMANTIC DESIGN TOKENS  —  Dark (default)
   ═══════════════════════════════════════════════ */
:root {
  /* ── Backgrounds */
  --bg-base:         #0D1117;
  --bg-secondary:    #161B22;
  --bg-tertiary:     #1C2128;
  --surface:         #21262D;
  --surface-raised:  #2D333B;
  --surface-inset:   #13181E;
  --surface-hover:   rgba(177,186,196,0.06);
  --surface-active:  rgba(177,186,196,0.10);

  /* ── Text */
  --text-primary:    #E6EDF3;
  --text-secondary:  #8B949E;
  --text-tertiary:   #484F58;
  --text-disabled:   #30363D;
  --text-inverse:    #0D1117;

  /* ── Borders */
  --border:          rgba(48,54,61,0.9);
  --border-subtle:   rgba(240,246,252,0.06);
  --border-strong:   rgba(139,148,158,0.28);

  /* ── Accent (blue — not neon) */
  --accent:          #2F81F7;
  --accent-hover:    #388BFD;
  --accent-muted:    rgba(47,129,247,0.10);
  --accent-border:   rgba(47,129,247,0.25);

  /* ── Semantic states */
  --success:         #3FB950;
  --success-muted:   rgba(63,185,80,0.10);
  --success-border:  rgba(63,185,80,0.28);
  --warning:         #D29922;
  --warning-muted:   rgba(210,153,34,0.10);
  --warning-border:  rgba(210,153,34,0.28);
  --danger:          #F85149;
  --danger-muted:    rgba(248,81,73,0.09);
  --danger-border:   rgba(248,81,73,0.28);
  --info:            #79C0FF;
  --info-muted:      rgba(121,192,255,0.10);
  --info-border:     rgba(121,192,255,0.25);

  /* ── Counter palette */
  --c0:#2F81F7; --c1:#3FB950; --c2:#D29922;
  --c3:#F85149; --c4:#A371F7; --c5:#58A6FF;

  /* ── Spacing scale (4-base) */
  --s1:4px; --s2:8px; --s3:12px; --s4:16px;
  --s5:20px; --s6:24px; --s8:32px; --s10:40px; --s12:48px;

  /* ── Radius */
  --r-xs: 3px;
  --r-sm: 5px;
  --r-md: 7px;
  --r-lg: 10px;

  /* ── Elevation */
  --shadow-sm: 0 1px 3px rgba(0,0,0,.22), 0 1px 2px rgba(0,0,0,.16);
  --shadow-md: 0 4px 14px rgba(0,0,0,.30);

  /* ── Chart theme (read by Python chart builder) */
  --chart-bg:        #21262D;
  --chart-text:      #8B949E;
  --chart-grid:      rgba(48,54,61,0.55);
  --chart-axis:      rgba(139,148,158,0.30);
}

/* ═══════════════════════════════════════════════
   GLOBAL RESET + BASE
   ═══════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

[data-testid="stAppViewContainer"]  { background: var(--bg-base) !important; }
[data-testid="stHeader"]            { display: none !important; }
[data-testid="stToolbar"]           { display: none !important; }
#MainMenu, footer                   { visibility: hidden !important; }

/* ── Scrollbars ── */
::-webkit-scrollbar              { width: 5px; height: 5px; }
::-webkit-scrollbar-track        { background: var(--bg-base); }
::-webkit-scrollbar-thumb        { background: var(--surface-raised); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover  { background: var(--border-strong); }

/* ═══════════════════════════════════════════════
   SIDEBAR — always dark regardless of page theme
   ═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: #161B22 !important;
  border-right: 1px solid rgba(48,54,61,.9) !important;
  min-width: 252px !important; max-width: 252px !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #E6EDF3 !important; }
[data-testid="stSidebar"] p { color: #8B949E !important; font-size: 12px !important; }
[data-testid="stSidebar"] hr { border-color: rgba(48,54,61,.9) !important; margin: 4px 0 !important; }
[data-testid="collapsedControl"] {
  background: #161B22 !important;
  border: 1px solid rgba(48,54,61,.9) !important;
}

/* ── Sidebar: force ALL form controls to dark — sidebar never switches light ── */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
  background: #21262D !important;
  border: 1px solid rgba(48,54,61,.9) !important;
  border-radius: 5px !important;
  color: #E6EDF3 !important;
}
[data-testid="stSidebar"] .stSelectbox input,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div {
  background: #21262D !important;
  color: #E6EDF3 !important;
}
/* Force select dropdown icon dark */
[data-testid="stSidebar"] .stSelectbox svg { fill: #8B949E !important; }

/* ── Sidebar: radio buttons — clean minimal style ── */
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label {
  display: flex !important; align-items: center !important;
  padding: 6px 10px 6px 8px !important;
  border-radius: 5px !important;
  cursor: pointer !important;
  transition: background .12s !important;
  width: 100% !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(177,186,196,0.06) !important;
}
/* The circle radio indicator */
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio label span {
  font-size: 13px !important;
  font-weight: 400 !important;
  color: #8B949E !important;
}
[data-testid="stSidebar"] .stRadio label[data-checked="true"] span,
[data-testid="stSidebar"] .stRadio label[aria-checked="true"] span,
[data-testid="stSidebar"] [aria-checked="true"] ~ div p {
  color: #E6EDF3 !important;
  font-weight: 500 !important;
}
/* Make the radio dot smaller and match accent */
[data-testid="stSidebar"] .stRadio input[type="radio"] {
  width: 14px !important; height: 14px !important;
  accent-color: #2F81F7 !important;
}

/* ── Sidebar: sliders always dark ── */
[data-testid="stSidebar"] .stSlider [data-testid="stSlider"] { color: #8B949E !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: #2F81F7 !important; }
[data-testid="stSidebar"] .stSlider label { color: #8B949E !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] { color: #484F58 !important; }

/* ── Sidebar: buttons ── */
[data-testid="stSidebar"] .stButton > button {
  background: #21262D !important;
  border: 1px solid rgba(48,54,61,.9) !important;
  color: #8B949E !important;
  font-size: 12px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: #2D333B !important;
  border-color: rgba(139,148,158,.28) !important;
  color: #E6EDF3 !important;
}

/* ── Sidebar: theme toggle button — full width, secondary style ── */
#theme_btn > button {
  background: rgba(177,186,196,0.05) !important;
  border: 1px solid rgba(48,54,61,.9) !important;
  color: #8B949E !important;
  border-radius: 5px !important;
  padding: 7px 12px !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  width: 100% !important;
  text-align: left !important;
  letter-spacing: 0 !important;
  transform: none !important;
  box-shadow: none !important;
  margin: 0 !important;
}
#theme_btn > button:hover {
  background: rgba(177,186,196,0.10) !important;
  border-color: rgba(139,148,158,.28) !important;
  color: #E6EDF3 !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ── Sidebar: select inside dark always ── */
[data-testid="stSidebar"] [data-baseweb="select"] {
  background: #21262D !important;
  border-color: rgba(48,54,61,.9) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #21262D !important;
  color: #E6EDF3 !important;
  border-color: rgba(48,54,61,.9) !important;
}
[data-testid="stSidebar"] [data-baseweb="menu"],
[data-testid="stSidebar"] [data-baseweb="popover"] {
  background: #21262D !important;
  border: 1px solid rgba(48,54,61,.9) !important;
}
[data-testid="stSidebar"] [data-baseweb="option"] {
  background: #21262D !important;
  color: #8B949E !important;
}
[data-testid="stSidebar"] [data-baseweb="option"]:hover {
  background: rgba(177,186,196,0.08) !important;
  color: #E6EDF3 !important;
}

/* ── Sidebar: toggle ── */
[data-testid="stSidebar"] .stToggle > label { color: #8B949E !important; }
[data-testid="stSidebar"] .stToggle input:checked + div { background: #2F81F7 !important; }

/* ── Sidebar: captions ── */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  color: #484F58 !important; font-size: 11px !important;
}

/* ═══════════════════════════════════════════════
   MAIN WORKSPACE
   ═══════════════════════════════════════════════ */
[data-testid="stMain"] { background: var(--bg-base) !important; }
.block-container {
  background: var(--bg-base) !important;
  padding: 0 var(--s6) var(--s12) !important;
  max-width: 100% !important;
}

/* ═══════════════════════════════════════════════
   FORM CONTROLS
   ═══════════════════════════════════════════════ */
.stSelectbox > div > div,
.stTextInput > div > div,
.stTextArea > div > div,
.stNumberInput > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  transition: border-color .15s;
}
.stTextInput input, .stTextArea textarea,
.stNumberInput input, .stSelectbox select {
  color: var(--text-primary) !important;
  background: transparent !important;
  caret-color: var(--accent) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
  color: var(--text-tertiary) !important;
  opacity: 1 !important;
}
.stTextInput > div > div:focus-within,
.stSelectbox > div > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-muted) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label,
.stSlider label, .stNumberInput label, .stCheckbox label,
.stRadio label, .stMultiSelect label, .stDateInput label {
  color: var(--text-secondary) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}
.stCheckbox span, .stRadio span, .stToggle > label {
  color: var(--text-secondary) !important;
}
div[data-testid="stRadioButton"] label,
div[data-testid="stRadioButton"] p { color: var(--text-secondary) !important; }
.stSlider > div > div > div > div { background: var(--accent) !important; }

/* ── Buttons ── */
.stButton > button {
  background: var(--surface) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 6px 14px !important;
  transition: all .15s ease;
  letter-spacing: 0;
}
.stButton > button:hover {
  background: var(--surface-raised) !important;
  border-color: var(--border-strong) !important;
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.stDownloadButton > button {
  background: var(--accent-muted) !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent-border) !important;
  border-radius: var(--r-sm) !important;
  font-weight: 600 !important;
}
.stDownloadButton > button:hover { background: rgba(47,129,247,.16) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-tertiary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  gap: 2px !important; padding: 3px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-secondary) !important;
  border-radius: var(--r-sm) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  padding: 5px 14px !important;
  transition: all .12s;
}
.stTabs [aria-selected="true"] {
  background: var(--surface-raised) !important;
  color: var(--text-primary) !important;
  box-shadow: var(--shadow-sm) !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Expander ── */
.stExpander {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
}
.stExpander summary { color: var(--text-secondary) !important; font-size: 13px !important; font-weight: 500 !important; }
.stExpander summary:hover { color: var(--text-primary) !important; }
.stExpander [data-testid="stExpanderDetails"] p { color: var(--text-secondary) !important; }

/* ── Alerts ── */
.stInfo    { background: var(--info-muted) !important;    border-color: var(--info-border) !important;    border-radius: var(--r-md) !important; }
.stWarning { background: var(--warning-muted) !important; border-color: var(--warning-border) !important; border-radius: var(--r-md) !important; }
.stSuccess { background: var(--success-muted) !important; border-color: var(--success-border) !important; border-radius: var(--r-md) !important; }
.stError   { background: var(--danger-muted) !important;  border-color: var(--danger-border) !important;  border-radius: var(--r-md) !important; }
.stInfo p    { color: var(--text-primary) !important; }
.stWarning p { color: #E3B341 !important; }
.stSuccess p { color: #56D364 !important; }
.stError p   { color: #FF7B72 !important; }

/* ── Markdown ── */
[data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li { color: var(--text-secondary) !important; }
[data-testid="stMarkdown"] h1, [data-testid="stMarkdown"] h2,
[data-testid="stMarkdown"] h3, [data-testid="stMarkdown"] h4 { color: var(--text-primary) !important; }
h1,h2,h3,h4 { color: var(--text-primary) !important; font-family: 'Inter', sans-serif; }
p  { color: var(--text-secondary); font-size: 13px; line-height: 1.6; }
li { color: var(--text-secondary); font-size: 13px; }
td,th { color: var(--text-secondary) !important; font-size: 13px !important; }
code {
  background: var(--surface-raised) !important;
  color: var(--info) !important;
  border-radius: var(--r-xs); padding: 2px 6px;
  font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
hr { border-color: var(--border) !important; }
.stCaption, [data-testid="stCaptionContainer"] p {
  color: var(--text-tertiary) !important; font-size: 11px !important;
}

/* ── Dataframe ── */
.stDataFrame,
.stDataFrame [data-testid="stDataFrameResizable"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
}
/* Fix inner dataframe surfaces */
.stDataFrame [data-testid="stDataFrameResizable"] > div { background: var(--surface) !important; }

/* ── Altair/Vega chart container ── */
[data-testid="stVegaLiteChart"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  overflow: hidden;
}

/* ── Metric tiles ── */
div[data-testid="metric-container"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 12px 16px;
  transition: border-color .15s;
}
div[data-testid="metric-container"]:hover { border-color: var(--border-strong); }
div[data-testid="metric-container"] label {
  color: var(--text-tertiary) !important;
  font-size: 10px !important; font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: .07em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--text-primary) !important;
  font-size: 20px !important; font-weight: 700 !important;
  font-variant-numeric: tabular-nums;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── File uploader ── */
.stFileUploader {
  background: var(--surface) !important;
  border: 1px dashed var(--border-strong) !important;
  border-radius: var(--r-lg) !important;
  transition: border-color .2s, background .2s;
}
.stFileUploader:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploadDropzone"] { background: transparent !important; }
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small { color: var(--text-secondary) !important; }

/* ═══════════════════════════════════════════════
   QUEUEIQ COMPONENT SYSTEM
   ═══════════════════════════════════════════════ */

/* ── Sidebar brand ── */
.qiq-brand {
  display: flex; align-items: center; gap: 10px;
  padding: 14px 14px 12px;
  border-bottom: 1px solid var(--border);
}
.qiq-logo {
  width: 28px; height: 28px; border-radius: 7px;
  background: var(--accent); flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; color: #fff; letter-spacing: -.5px;
}
.qiq-wordmark { font-size: 14px; font-weight: 700; color: var(--text-primary) !important; letter-spacing: -.3px; line-height: 1.1; }
.qiq-tagline  { font-size: 10px; color: var(--text-tertiary) !important; margin-top: 1px; }

/* ── Sidebar section labels ── */
.sb-section {
  display: block; padding: 12px 14px 5px;
  font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .10em;
  color: var(--text-tertiary) !important;
}
.sb-lbl {
  display: block; padding: 3px 0 6px;
  font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .10em;
  color: var(--text-tertiary);
}
.sb-divider { height: 1px; background: var(--border); margin: 6px 0; }
.sb-user-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--success); display: inline-block; margin-right: 4px;
}

/* ── Page header ── */
.qiq-page-hdr {
  display: flex; align-items: flex-start;
  justify-content: space-between; gap: 16px;
  padding: 18px 0 14px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}
.qiq-page-title { font-size: 16px; font-weight: 600; color: var(--text-primary); letter-spacing: -.3px; }
.qiq-page-sub   { font-size: 12px; color: var(--text-tertiary); margin-top: 3px; }
.qiq-page-hdr-r { display: flex; align-items: center; gap: 8px; flex-shrink: 0; padding-top: 2px; }

/* ── Chips / pills ── */
.chip {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 9px; border-radius: 20px;
  font-size: 11px; font-weight: 500; border: 1px solid;
  white-space: nowrap;
}
.chip-mode { background: var(--surface); border-color: var(--border); color: var(--text-secondary); }
.chip-ok   { background: var(--success-muted); border-color: var(--success-border); color: var(--success); }
.chip-warn { background: var(--warning-muted); border-color: var(--warning-border); color: var(--warning); }
.chip-crit { background: var(--danger-muted);  border-color: var(--danger-border);  color: var(--danger); }
.chip-sim  { background: var(--accent-muted);  border-color: var(--accent-border);  color: var(--accent); }
.chip-dot  { width: 5px; height: 5px; border-radius: 50%; background: currentColor; flex-shrink: 0; animation: pulse-dot 2.4s ease-in-out infinite; }
.ts-mono   { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-tertiary); }

/* ── Section header ── */
.sec-hdr {
  font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .09em;
  color: var(--text-tertiary);
  margin: 20px 0 10px;
  display: flex; align-items: center; gap: 10px;
}
.sec-hdr::after { content:''; flex:1; height:1px; background: var(--border); }

/* ── Status bar ── */
.status-bar {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 10px 16px;
  margin-top: 14px;
  border-left: 3px solid;
  position: relative; overflow: hidden;
}
.status-bar.ok   { border-left-color: var(--success); }
.status-bar.warn { border-left-color: var(--warning); }
.status-bar.crit { border-left-color: var(--danger); }
.status-bar.sim  { border-left-color: var(--accent); }
.sb-level { font-size: 13px; font-weight: 700; letter-spacing: -.2px; }
.sb-summary { font-size: 12px; color: var(--text-secondary); }
.sb-sep { width: 1px; height: 16px; background: var(--border); align-self: center; margin: 0 2px; }
.sb-kpi { text-align: center; }
.sb-kpi-v { font-size: 16px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1; }
.sb-kpi-l { font-size: 10px; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: .07em; margin-top: 2px; }

/* ── Intelligence cards ── */
.intel-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin-bottom: 0; }
@media (max-width: 900px) { .intel-grid { grid-template-columns: repeat(2,1fr); } }
@media (max-width: 550px) { .intel-grid { grid-template-columns: 1fr; } }
.intel-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 12px 14px;
  border-top: 2px solid;
  animation: fadeUp .18s ease both;
}
.ic-eyebrow { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .09em; margin-bottom: 5px; }
.ic-val     { font-size: 14px; font-weight: 600; color: var(--text-primary); margin-bottom: 3px; line-height: 1.3; }
.ic-sub     { font-size: 11px; color: var(--text-secondary); line-height: 1.5; }

/* ── Counter cards ── */
.counter-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px,1fr)); gap: 8px; }
.cc {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 14px 16px 12px;
  border-left: 3px solid;
  transition: border-color .15s, box-shadow .15s;
  animation: fadeUp .20s ease both;
}
.cc:hover { border-color: var(--border-strong); box-shadow: var(--shadow-sm); }
.cc-0{border-left-color:var(--c0)} .cc-1{border-left-color:var(--c1)}
.cc-2{border-left-color:var(--c2)} .cc-3{border-left-color:var(--c3)}
.cc-4{border-left-color:var(--c4)} .cc-5{border-left-color:var(--c5)}
.cc-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
.cc-name { font-size: 11px; color: var(--text-secondary); font-weight: 500; }
.cc-count { font-size: 26px; font-weight: 700; color: var(--text-primary); font-variant-numeric: tabular-nums; line-height: 1; }
.cc-unit  { font-size: 11px; color: var(--text-tertiary); margin-top: 2px; margin-bottom: 10px; }
.cc-row   { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.cc-wait  { font-size: 12px; color: var(--text-secondary); }
.cc-pred  { font-size: 11px; color: var(--text-tertiary); }
.cc-pred strong { color: var(--accent); }
.cc-trend     { font-size: 11px; font-weight: 600; }
.cc-trend.up  { color: var(--danger); }
.cc-trend.dn  { color: var(--success); }
.cc-trend.flat{ color: var(--text-tertiary); }
.util-wrap { background: var(--bg-tertiary); border-radius: 3px; height: 3px; margin-top: 10px; overflow: hidden; }
.util-fill { height: 100%; border-radius: 3px; transition: width .4s ease; }

/* ── Status badges ── */
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px; border-radius: 20px;
  font-size: 10px; font-weight: 600;
  letter-spacing: .03em; border: 1px solid;
}
.badge.ok   { background:var(--success-muted); border-color:var(--success-border); color:var(--success); }
.badge.warn { background:var(--warning-muted); border-color:var(--warning-border); color:var(--warning); }
.badge.crit { background:var(--danger-muted);  border-color:var(--danger-border);  color:var(--danger); }

/* ── Action intelligence panel (recommendation + AI alert merged) ── */
.action-panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 0;
  overflow: hidden;
}
.ap-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px 10px;
  border-bottom: 1px solid var(--border);
}
.ap-header.no_action    { border-left: 3px solid var(--success); }
.ap-header.redirect     { border-left: 3px solid var(--warning); }
.ap-header.open_counter { border-left: 3px solid var(--danger);  }
.ap-header.monitor      { border-left: 3px solid var(--warning); }
.ap-eyebrow { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .09em; color: var(--text-tertiary); }
.ap-title   { font-size: 14px; font-weight: 600; color: var(--text-primary); margin-top: 3px; }
.ap-body    { padding: 12px 16px; }
.ap-msg     { font-size: 13px; color: var(--text-secondary); line-height: 1.6; margin-bottom: 10px; }
.ap-why     {
  font-size: 12px; color: var(--text-secondary);
  padding: 7px 10px; background: var(--surface-inset);
  border-radius: var(--r-sm); border-left: 2px solid var(--border-strong);
  margin-bottom: 10px; line-height: 1.55;
}
.ap-impact  { display: flex; gap: 20px; flex-wrap: wrap; padding-top: 4px; }
.ap-metric  { }
.ap-mval    { font-size: 18px; font-weight: 700; font-variant-numeric: tabular-nums; }
.ap-mlbl    { font-size: 10px; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: .07em; margin-top: 2px; }
.ap-divider { height: 1px; background: var(--border); }
.ap-alert   { padding: 12px 16px; }
.ap-alert-hdr { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.ap-alert-src { font-size: 10px; color: var(--text-tertiary); margin-top: 8px; font-style: italic; }
.ap-alert-text { font-size: 13px; color: var(--text-primary); line-height: 1.7; }
.ap-alert-sev {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px; border-radius: 20px;
  font-size: 10px; font-weight: 600; border: 1px solid;
}

/* ── Upload drop zone ── */
.drop-zone {
  background: var(--surface); border: 1.5px dashed var(--border-strong);
  border-radius: var(--r-lg); padding: 32px 24px; text-align: center;
  transition: border-color .2s, background .2s;
}
.drop-zone:hover { border-color: var(--accent); background: var(--accent-muted); }
.dz-icon  { font-size: 28px; margin-bottom: 12px; opacity: .65; }
.dz-title { font-size: 14px; font-weight: 600; color: var(--text-primary); margin-bottom: 6px; }
.dz-sub   { font-size: 12px; color: var(--text-secondary); line-height: 1.6; }
.dz-tags  { display: flex; gap: 5px; justify-content: center; flex-wrap: wrap; margin-top: 14px; }
.dz-tag {
  font-size: 11px; color: var(--text-tertiary);
  background: var(--bg-tertiary); border: 1px solid var(--border);
  border-radius: var(--r-xs); padding: 2px 8px;
  font-family: 'JetBrains Mono', monospace;
}

/* ── Camera workspace ── */
.cam-state {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: 32px 24px; text-align: center;
}
.cam-state-icon  { font-size: 32px; opacity: .55; margin-bottom: 12px; }
.cam-state-title { font-size: 14px; font-weight: 600; color: var(--text-primary); margin-bottom: 5px; }
.cam-state-sub   { font-size: 12px; color: var(--text-secondary); line-height: 1.6; max-width: 300px; margin: 0 auto; }

/* ── Simulation placeholder ── */
.sim-ready {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: 32px 24px; text-align: center;
  margin-top: 16px;
}

/* ── Login page ── */
.lp-bg   { position:absolute; inset:0; z-index:0;
  background-image: linear-gradient(var(--border-subtle) 1px, transparent 1px),
                    linear-gradient(90deg, var(--border-subtle) 1px, transparent 1px);
  background-size: 48px 48px; }
.lp-orb  { position:absolute; width:320px; height:320px; border-radius:50%;
  background:radial-gradient(circle, rgba(47,129,247,.09) 0%, transparent 70%);
  top:-80px; right:-80px; z-index:0; }
.lp-wrap { min-height:100vh; display:flex; flex-direction:column; align-items:center;
  justify-content:center; padding:48px 32px; position:relative; z-index:2; }
.lp-logo { width:48px; height:48px; border-radius:12px; background:var(--accent);
  display:flex; align-items:center; justify-content:center;
  font-size:20px; font-weight:700; color:#fff; margin-bottom:16px;
  box-shadow:0 4px 20px rgba(47,129,247,.40); }
.lp-brand   { font-size:24px; font-weight:700; color:#E6EDF3; text-align:center; letter-spacing:-.4px; margin-bottom:4px; }
.lp-tagline { font-size:11px; color:rgba(139,148,158,.7); text-align:center;
  text-transform:uppercase; letter-spacing:.10em; margin-bottom:32px; }
.lp-stat    { text-align:center; background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.07); border-radius:var(--r-md);
  padding:10px 14px; min-width:60px; }
.lp-sv { font-size:15px; font-weight:700; color:#E6EDF3; }
.lp-sl { font-size:9px; color:rgba(139,148,158,.6); text-transform:uppercase; letter-spacing:.09em; margin-top:2px; }
.rp-lbl   { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.12em;
  color:var(--accent); margin-bottom:18px; display:flex; align-items:center; gap:8px; }
.rp-lbl::before { content:''; display:inline-block; width:14px; height:2px;
  background:var(--accent); border-radius:2px; }
.rp-title { font-size:22px; font-weight:700; letter-spacing:-.4px; line-height:1.2; margin-bottom:5px; }
.rp-sub   { font-size:13px; line-height:1.6; margin-bottom:22px; }
.rp-hint  { margin-top:14px; padding:9px 13px; background:rgba(9,105,218,.07);
  border-radius:var(--r-md); border-left:3px solid rgba(9,105,218,.28);
  font-size:12px; line-height:1.55; }
.rp-foot  { margin-top:22px; padding-top:13px; border-top:1px solid #E1E4E8;
  font-size:11px; color:#8C959F; text-align:center; }

/* ── Animations ── */
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.8)} }
@keyframes fadeUp    { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:translateY(0)} }
@media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none !important; transition:none !important; } }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# THEME SYSTEM
# inject_theme() swaps ALL :root tokens.
# Charts are theme-aware via _chart_tc() returning per-render config.
# ─────────────────────────────────────────────────────────────────────────────

def inject_theme(theme: str) -> None:
    """Swap semantic tokens and override all Streamlit surfaces."""
    # Keep sidebar visible (login page hides it)
    st.markdown(
        "<style>"
        "[data-testid='stSidebar']{display:flex!important;visibility:visible!important;}"
        "</style>",
        unsafe_allow_html=True,
    )

    if theme == "light":
        st.markdown("""
<style>
:root {
  --bg-base:         #F6F8FA;
  --bg-secondary:    #FFFFFF;
  --bg-tertiary:     #EAEEF2;
  --surface:         #FFFFFF;
  --surface-raised:  #F6F8FA;
  --surface-inset:   #F0F3F6;
  --surface-hover:   rgba(27,31,36,0.04);
  --surface-active:  rgba(27,31,36,0.08);
  --text-primary:    #1F2328;
  --text-secondary:  #57606A;
  --text-tertiary:   #8C959F;
  --text-disabled:   #CFD4DB;
  --text-inverse:    #FFFFFF;
  --border:          rgba(208,215,222,0.9);
  --border-subtle:   rgba(208,215,222,0.5);
  --border-strong:   rgba(140,149,159,0.5);
  --accent:          #0969DA;
  --accent-hover:    #0860CA;
  --accent-muted:    rgba(9,105,218,0.08);
  --accent-border:   rgba(9,105,218,0.22);
  --success:         #1A7F37;
  --success-muted:   rgba(26,127,55,0.08);
  --success-border:  rgba(26,127,55,0.24);
  --warning:         #9A6700;
  --warning-muted:   rgba(154,103,0,0.08);
  --warning-border:  rgba(154,103,0,0.24);
  --danger:          #CF222E;
  --danger-muted:    rgba(207,34,46,0.07);
  --danger-border:   rgba(207,34,46,0.24);
  --info:            #0969DA;
  --info-muted:      rgba(9,105,218,0.08);
  --info-border:     rgba(9,105,218,0.22);
  --c0:#0969DA; --c1:#1A7F37; --c2:#9A6700; --c3:#CF222E; --c4:#8250DF; --c5:#0550AE;
  --chart-bg:        #FFFFFF;
  --chart-text:      #57606A;
  --chart-grid:      rgba(208,215,222,0.7);
  --chart-axis:      rgba(208,215,222,0.9);
  --shadow-sm: 0 1px 3px rgba(27,31,36,0.12), 0 1px 2px rgba(27,31,36,0.08);
  --shadow-md: 0 4px 14px rgba(27,31,36,0.18);
}
html,body,.stApp,[class*="css"]{background:#F6F8FA!important;color:#1F2328!important;}
[data-testid="stAppViewContainer"]{background:#F6F8FA!important;}
[data-testid="stMain"]{background:#F6F8FA!important;}
.block-container{background:#F6F8FA!important;}

/* Sidebar stays dark in light mode — deliberate contrast anchor */
[data-testid="stSidebar"]{background:#0D1117!important;border-right:1px solid rgba(48,54,61,.9)!important;}
[data-testid="stSidebar"] *{color:#E6EDF3!important;}
[data-testid="stSidebar"] p{color:#8B949E!important;}
[data-testid="stSidebar"] hr{border-color:rgba(48,54,61,.9)!important;}

::-webkit-scrollbar-track{background:#F6F8FA;}
::-webkit-scrollbar-thumb{background:#D0D7DE;}

.stSelectbox>div>div,.stTextInput>div>div,.stTextArea>div>div,.stNumberInput>div>div{background:#FFFFFF!important;border-color:#D0D7DE!important;}
.stTextInput input,.stTextArea textarea,.stNumberInput input{color:#1F2328!important;caret-color:#0969DA!important;}
.stTextInput input::placeholder,.stTextArea textarea::placeholder{color:#8C959F!important;}
.stTextInput>div>div:focus-within,.stSelectbox>div>div:focus-within{border-color:#0969DA!important;box-shadow:0 0 0 3px rgba(9,105,218,.10)!important;}
.stTextInput label,.stTextArea label,.stSelectbox label,.stSlider label,.stNumberInput label,.stCheckbox label,.stRadio label{color:#57606A!important;}
.stCheckbox span,.stRadio span,.stToggle>label{color:#57606A!important;}
div[data-testid="stRadioButton"] label,div[data-testid="stRadioButton"] p{color:#1F2328!important;}
.stSlider>div>div>div>div{background:#0969DA!important;}

.stButton>button{background:#FFFFFF!important;color:#1F2328!important;border-color:#D0D7DE!important;}
.stButton>button:hover{background:#F6F8FA!important;border-color:#8C959F!important;}
.stDownloadButton>button{background:rgba(9,105,218,.08)!important;color:#0969DA!important;border-color:rgba(9,105,218,.22)!important;}

.stTabs [data-baseweb="tab-list"]{background:#EAEEF2!important;border-color:#D0D7DE!important;}
.stTabs [data-baseweb="tab"]{color:#57606A!important;}
.stTabs [aria-selected="true"]{background:#FFFFFF!important;color:#1F2328!important;}

.stExpander{background:#FFFFFF!important;border-color:#D0D7DE!important;}
.stExpander summary{color:#57606A!important;}
.stExpander [data-testid="stExpanderDetails"] p{color:#57606A!important;}

.stInfo{background:rgba(9,105,218,.07)!important;border-color:rgba(9,105,218,.22)!important;}
.stInfo p{color:#1F2328!important;}
.stWarning{background:rgba(154,103,0,.07)!important;border-color:rgba(154,103,0,.22)!important;}
.stWarning p{color:#633B00!important;}
.stSuccess{background:rgba(26,127,55,.07)!important;border-color:rgba(26,127,55,.22)!important;}
.stSuccess p{color:#116329!important;}
.stError{background:rgba(207,34,46,.07)!important;border-color:rgba(207,34,46,.22)!important;}
.stError p{color:#82071E!important;}

[data-testid="stMarkdown"] p,[data-testid="stMarkdown"] li{color:#57606A!important;}
[data-testid="stMarkdown"] h1,[data-testid="stMarkdown"] h2,[data-testid="stMarkdown"] h3,[data-testid="stMarkdown"] h4{color:#1F2328!important;}
h1,h2,h3,h4{color:#1F2328!important;} p{color:#57606A;} li{color:#57606A;} td,th{color:#57606A!important;}
code{background:#EAEEF2!important;color:#0969DA!important;}
hr{border-color:#D0D7DE!important;}
.stCaption,[data-testid="stCaptionContainer"] p{color:#8C959F!important;}

.stDataFrame,.stDataFrame [data-testid="stDataFrameResizable"]{background:#FFFFFF!important;border-color:#D0D7DE!important;}
.stDataFrame [data-testid="stDataFrameResizable"]>div{background:#FFFFFF!important;}
[data-testid="stVegaLiteChart"]{background:#FFFFFF!important;border-color:#D0D7DE!important;}

div[data-testid="metric-container"]{background:#FFFFFF!important;border-color:#D0D7DE!important;}
div[data-testid="metric-container"]:hover{border-color:#0969DA!important;}
div[data-testid="metric-container"] label{color:#8C959F!important;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#1F2328!important;}

.stFileUploader{background:#FFFFFF!important;border-color:#8C959F!important;}
[data-testid="stFileUploadDropzone"]{background:#FFFFFF!important;}
[data-testid="stFileUploadDropzone"] p,[data-testid="stFileUploadDropzone"] small{color:#57606A!important;}
</style>""", unsafe_allow_html=True)

    else:  # dark — reinforce tokens
        st.markdown("""
<style>
:root {
  --bg-base:#0D1117; --bg-secondary:#161B22; --bg-tertiary:#1C2128;
  --surface:#21262D; --surface-raised:#2D333B; --surface-inset:#13181E;
  --text-primary:#E6EDF3; --text-secondary:#8B949E; --text-tertiary:#484F58;
  --border:rgba(48,54,61,.9); --border-strong:rgba(139,148,158,.28);
  --accent:#2F81F7; --accent-muted:rgba(47,129,247,.10); --accent-border:rgba(47,129,247,.25);
  --success:#3FB950; --success-muted:rgba(63,185,80,.10); --success-border:rgba(63,185,80,.28);
  --warning:#D29922; --warning-muted:rgba(210,153,34,.10); --warning-border:rgba(210,153,34,.28);
  --danger:#F85149; --danger-muted:rgba(248,81,73,.09); --danger-border:rgba(248,81,73,.28);
  --chart-bg:#21262D; --chart-text:#8B949E; --chart-grid:rgba(48,54,61,.55); --chart-axis:rgba(139,148,158,.30);
}
html,body,.stApp,[class*="css"]{background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stAppViewContainer"]{background:#0D1117!important;}
[data-testid="stMain"]{background:#0D1117!important;}
.block-container{background:#0D1117!important;}
</style>""", unsafe_allow_html=True)


def _chart_tc(theme: str) -> dict:
    """Return Altair chart color config for the current theme."""
    if theme == "light":
        return {
            "bg":   "#FFFFFF",
            "text": "#57606A",
            "grid": "rgba(208,215,222,0.7)",
            "axis": "rgba(208,215,222,0.9)",
        }
    return {
        "bg":   "#21262D",
        "text": "#8B949E",
        "grid": "rgba(48,54,61,0.55)",
        "axis": "rgba(139,148,158,0.30)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS LOGIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_level(wait_min) -> str:
    if wait_min is None or wait_min < 12:
        return "ok"
    return "crit" if wait_min >= 25 else "warn"


def _badge(load: str) -> str:
    labels = {"ok": "Low", "warn": "Med", "crit": "High"}
    return f'<span class="badge {load}">● {labels.get(load, load)}</span>'


@st.cache_data(show_spinner=False)
def generate_sample_csv() -> bytes:
    base = datetime.now().replace(second=0, microsecond=0) - timedelta(hours=4)
    rng  = np.random.default_rng(42)
    rows = []
    for i in range(49):
        t    = base + timedelta(minutes=5 * i)
        frac = i / 48.0
        ramp = min(1.0, frac * 1.5)
        c1   = int(max(3, 4 + 33 * ramp * (1 - .12 * ramp) + rng.integers(-2, 3)))
        c2   = int(max(1, 7 + 6 * np.sin(np.pi * frac)     + rng.integers(-2, 3)))
        c3   = int(max(1, 30 - 25 * frac                    + rng.integers(-3, 4)))
        for name, cnt in [("Counter 1 (Main)", c1), ("Counter 2 (Express)", c2), ("Counter 3 (Inquiry)", c3)]:
            rows.append({"timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                         "counter_name": name, "people_count": cnt})
    buf = io.BytesIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()


def parse_csv_raw(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    missing = {"timestamp", "counter_name", "people_count"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["timestamp"]    = pd.to_datetime(df["timestamp"])
    df["people_count"] = pd.to_numeric(df["people_count"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df = df.sort_values("timestamp")
    configs = []
    for idx, (name, grp) in enumerate(sorted(df.groupby("counter_name"), key=lambda x: x[0])):
        history = [
            {"counter_id": idx+1, "name": str(name), "people_count": int(r["people_count"]),
             "timestamp_epoch": r["timestamp"].timestamp(),
             "timestamp_str":   r["timestamp"].strftime("%H:%M")}
            for _, r in grp.sort_values("timestamp").iterrows()
        ]
        configs.append({"id": idx+1, "name": str(name), "history": history, "arrival_rate": None})
    return configs


def apply_service_rate(configs_raw, service_rate):
    return [{**c, "service_rate": service_rate} for c in configs_raw]


# ─────────────────────────────────────────────────────────────────────────────
# RENDER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def render_header(mode: str, info: str, theme: str) -> None:
    icon_map  = {"simulation": "◉", "real_data": "↑", "cv": "◈"}
    label_map = {"simulation": "Simulation", "real_data": "CSV", "cv": "Vision"}
    level_map = {"simulation": "sim", "real_data": "ok", "cv": "ok"}
    ts = datetime.now().strftime("%H:%M:%S")
    lbl  = label_map.get(mode, mode)
    lcls = level_map.get(mode, "ok")
    icon = icon_map.get(mode, "◈")
    st.markdown(f"""
<div class="qiq-page-hdr">
  <div>
    <div class="qiq-page-title">Queue Intelligence</div>
    <div class="qiq-page-sub">{info}</div>
  </div>
  <div class="qiq-page-hdr-r">
    <span class="chip chip-mode">{icon} {lbl}</span>
    <span class="chip chip-{lcls}"><span class="chip-dot"></span>LIVE</span>
    <span class="ts-mono">{ts}</span>
  </div>
</div>""", unsafe_allow_html=True)


def render_status_bar(readings, wait_times, preds, rec, threshold, theme: str) -> None:
    """Compact single-row facility status — replaces redundant hero + metrics sections."""
    if not readings:
        return
    counter_risks = []
    for i, r in enumerate(readings):
        wt = wait_times[i] if i < len(wait_times) else None
        pc = preds[i].get("predicted_count") if i < len(preds) else None
        counter_risks.append(classify_risk(r["people_count"], pc, wt, threshold))
    facility = classify_facility_risk(counter_risks)

    lvl      = facility["level"]
    label    = RISK_LABELS[lvl][1]
    css_cls  = {"normal": "ok", "moderate": "warn", "high": "warn", "critical": "crit"}.get(lvl, "ok")
    lc_map   = {"ok": "var(--success)", "warn": "var(--warning)", "crit": "var(--danger)"}
    lc       = lc_map.get(css_cls, "var(--text-secondary)")

    total    = sum(r["people_count"] for r in readings)
    valid_w  = [w for w in wait_times if w is not None]
    avg_w    = sum(valid_w) / len(valid_w) if valid_w else 0
    max_w    = max(valid_w) if valid_w else 0
    pred_tot = sum((preds[i].get("predicted_count") or r["people_count"]) for i, r in enumerate(readings))
    action   = rec.get("title", rec.get("action", "Monitor").replace("_", " ").title())

    st.markdown(f"""
<div class="status-bar {css_cls}">
  <div>
    <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.09em;color:var(--text-tertiary)">FACILITY STATUS</div>
    <div class="sb-level" style="color:{lc}">{label}</div>
    <div class="sb-summary">{facility['summary']}</div>
  </div>
  <div class="sb-sep"></div>
  <div class="sb-kpi">
    <div class="sb-kpi-v">{total}</div>
    <div class="sb-kpi-l">In Queue</div>
  </div>
  <div class="sb-kpi">
    <div class="sb-kpi-v" style="color:{lc}">{pred_tot}</div>
    <div class="sb-kpi-l">Predicted</div>
  </div>
  <div class="sb-kpi">
    <div class="sb-kpi-v">{avg_w:.0f}m</div>
    <div class="sb-kpi-l">Avg Wait</div>
  </div>
  <div class="sb-kpi">
    <div class="sb-kpi-v" style="color:{lc}">{max_w:.0f}m</div>
    <div class="sb-kpi-l">Peak Wait</div>
  </div>
  <div class="sb-sep"></div>
  <div style="font-size:12px;font-weight:500;color:var(--text-primary);max-width:160px;line-height:1.3">{action}</div>
</div>""", unsafe_allow_html=True)
    return counter_risks, facility


def render_intel_cards(readings, wait_times, preds, rec, horizon) -> None:
    """Four operational intelligence questions — compact and integrated."""
    if not readings:
        return
    total   = sum(r["people_count"] for r in readings)
    valid_w = [w for w in wait_times if w is not None]
    avg_w   = sum(valid_w) / len(valid_w) if valid_w else 0

    worst_pred  = max(preds, key=lambda p: p.get("predicted_count") or 0) if preds else {}
    pred_count  = worst_pred.get("predicted_count")
    pred_min    = worst_pred.get("predicted_in_min", horizon)
    pred_str    = f"{pred_count} ppl in +{int(pred_min)}m" if pred_count else "Collecting data…"
    pred_sub    = (worst_pred.get("sentence", "")[:60] + "…") if worst_pred.get("sentence") else ""

    why_text    = rec.get("arrival_vs_service") or "Queue dynamics are stable."
    what_title  = rec.get("title", "Monitor")
    what_text   = (rec.get("message", "")[:72] + "…") if len(rec.get("message", "")) > 72 else rec.get("message", "")

    colors = ["var(--accent)", "var(--info)", "var(--warning)", "var(--success)"]
    labels = ["① NOW", "② NEXT", "③ WHY", "④ ACTION"]
    vals   = [f"{total} waiting", pred_str, why_text[:68] + ("…" if len(why_text) > 68 else ""), what_title]
    subs   = [f"Avg wait {avg_w:.0f}m · {len(readings)} counter{'s' if len(readings)!=1 else ''}", pred_sub, "", what_text]

    cards_html = "".join(f"""
<div class="intel-card" style="border-top-color:{c}">
  <div class="ic-eyebrow" style="color:{c}">{l}</div>
  <div class="ic-val">{v}</div>
  <div class="ic-sub">{s}</div>
</div>""" for c, l, v, s in zip(colors, labels, vals, subs))

    st.markdown(f'<div class="sec-hdr">Operational Intelligence</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="intel-grid">{cards_html}</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


def render_counter_cards(readings, wait_times, preds) -> None:
    st.markdown('<div class="sec-hdr">Live Counter Status</div>', unsafe_allow_html=True)
    cols = st.columns(max(1, min(len(readings), 4)))
    for i, r in enumerate(readings):
        wt   = wait_times[i] if i < len(wait_times) else None
        pred = preds[i]      if i < len(preds)       else {}
        load = _load_level(wt)
        pc   = pred.get("predicted_count")
        cnt  = r["people_count"]
        util = min(int((cnt / 30) * 100), 100) if cnt else 0
        uc   = {"ok": "var(--success)", "warn": "var(--warning)", "crit": "var(--danger)"}.get(load, "var(--text-tertiary)")

        if pc is not None:
            delta = pc - cnt
            tcls  = "up" if delta > 1 else ("dn" if delta < -1 else "flat")
            tsym  = (f"↗ +{int(delta)}" if delta > 1 else (f"↘ {int(delta)}" if delta < -1 else "→ stable"))
        else:
            tcls, tsym = "flat", "—"

        pm_str = f"{float(pred.get('predicted_in_min', '?')):.0f}m" if pred.get("predicted_in_min") not in (None, "?") else "—"
        wt_str = f"{wt:.0f} min" if wt is not None else "—"
        pc_str = str(pc) if pc is not None else "—"

        with cols[i % len(cols)]:
            st.markdown(f"""
<div class="cc cc-{i % 6}">
  <div class="cc-top">
    <div>
      <div class="cc-name">{r['name']}</div>
      <div class="cc-count">{cnt}</div>
      <div class="cc-unit">people waiting</div>
    </div>
    {_badge(load)}
  </div>
  <div class="cc-row">
    <span class="cc-wait">⏱ {wt_str}</span>
    <span class="cc-trend {tcls}">{tsym}</span>
  </div>
  <div class="cc-pred">→ <strong>{pc_str}</strong> in +{pm_str}</div>
  <div class="util-wrap">
    <div class="util-fill" style="width:{util}%;background:{uc}"></div>
  </div>
</div>""", unsafe_allow_html=True)


def render_forecast_chart(counter_configs, preds, threshold, horizon, theme: str) -> None:
    """Altair line chart — fully theme-aware, no hardcoded dark colors."""
    st.markdown('<div class="sec-hdr">Queue Trend &amp; Forecast</div>', unsafe_allow_html=True)
    tc = _chart_tc(theme)

    rows: list[dict] = []
    window = 20
    for c in counter_configs:
        hist = c.get("history", [])[-window:]
        for idx, h in enumerate(hist):
            rows.append({"tick": idx, "series": c["name"], "count": h["people_count"], "kind": "hist"})

    fore_rows: list[dict] = []
    for i, c in enumerate(counter_configs):
        if i < len(preds):
            pc = preds[i].get("predicted_count")
            if pc is not None:
                t = len(c.get("history", [])[-window:])
                fore_rows.append({"tick": t, "series": c["name"], "count": pc, "kind": "fore"})

    thr_rows = [{"tick": t, "series": "Threshold", "count": threshold, "kind": "thr"}
                for t in range(window + 2)]

    if not rows and not fore_rows:
        st.info("No data to chart yet.")
        return

    all_series = list({r["series"] for r in rows + fore_rows + thr_rows})
    color_map  = {c["name"]: COUNTER_COLORS[i % 6] for i, c in enumerate(counter_configs)}
    color_map["Threshold"] = "#D29922"

    def _ax(title=""):
        return alt.Axis(
            labelColor=tc["text"], titleColor=tc["text"],
            gridColor=tc["grid"],  domainColor=tc["axis"],
            tickColor=tc["axis"],  labelFont="Inter, sans-serif",
            titleFont="Inter, sans-serif", title=title,
        )

    base = (
        alt.Chart(pd.DataFrame(rows))
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(filled=True, size=35))
        .encode(
            x=alt.X("tick:Q", axis=_ax("Ticks (oldest → now)")),
            y=alt.Y("count:Q", axis=_ax("People in queue")),
            color=alt.Color("series:N",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                legend=alt.Legend(
                    labelColor=tc["text"], titleColor=tc["text"],
                    fillColor=tc["bg"],    strokeColor=tc["axis"],
                    labelFont="Inter, sans-serif", titleFont="Inter, sans-serif",
                    title="Counter",
                )
            ),
            tooltip=["series:N", "count:Q", "tick:Q"],
        )
    )

    forecast = (
        alt.Chart(pd.DataFrame(fore_rows or [{"tick": 0, "series": "x", "count": 0, "kind": "fore"}]))
        .mark_point(size=110, filled=True, shape="diamond")
        .encode(
            x="tick:Q", y="count:Q",
            color=alt.Color("series:N",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                legend=None
            ),
            tooltip=["series:N", "count:Q"],
        )
    ) if fore_rows else alt.Chart(pd.DataFrame()).mark_point()

    thr_line = (
        alt.Chart(pd.DataFrame(thr_rows))
        .mark_line(strokeDash=[6, 4], strokeWidth=1.5, color="#D29922", opacity=.75)
        .encode(x="tick:Q", y="count:Q")
    )

    chart = (
        (base + forecast + thr_line)
        .configure(background=tc["bg"],
                   view=alt.ViewConfig(stroke=tc["axis"], strokeWidth=1))
        .properties(height=220)
        .interactive()
    )
    st.altair_chart(chart, width='stretch')
    st.caption(f"Historical (dots) · Forecast ◆ +{horizon}m · Dashed = threshold ({threshold} ppl)")


def render_action_panel(rec: dict, alert_res: dict | None) -> None:
    """Merged recommendation + AI alert in one cohesive panel."""
    action   = rec.get("action", "no_action")
    titles   = {
        "redirect":     "Redirect Customers to Adjacent Counter",
        "open_counter": "Activate an Additional Counter",
        "no_action":    "Queue Operating Within Normal Parameters",
        "monitor":      "Monitor — Elevated Activity Detected",
    }
    sev_map = {
        "redirect":     ("warn", "MEDIUM", "var(--warning)", "var(--warning-muted)", "var(--warning-border)"),
        "open_counter": ("crit", "HIGH",   "var(--danger)",  "var(--danger-muted)",  "var(--danger-border)"),
        "no_action":    ("ok",   "LOW",    "var(--success)", "var(--success-muted)", "var(--success-border)"),
        "monitor":      ("warn", "MEDIUM", "var(--warning)", "var(--warning-muted)", "var(--warning-border)"),
    }
    _, sev_txt, sev_c, sev_bg, sev_bd = sev_map.get(action, ("ok","LOW","var(--success)","var(--success-muted)","var(--success-border)"))

    avs     = rec.get("arrival_vs_service", "")
    msg     = rec.get("message", "")
    why_html = f'<div class="ap-why">📊 {avs}</div>' if avs else ""

    impact  = rec.get("impact", {})
    bw, aw  = impact.get("before_wait"), impact.get("after_wait")
    pm      = impact.get("people_moved", "")
    imp_html = ""
    if bw and aw:
        saved = bw - aw
        imp_html = f"""
<div class="ap-impact">
  <div class="ap-metric">
    <div class="ap-mval" style="color:var(--text-secondary)">{bw:.0f}m</div>
    <div class="ap-mlbl">Current Wait</div>
  </div>
  <div style="font-size:16px;color:var(--text-tertiary);align-self:center">→</div>
  <div class="ap-metric">
    <div class="ap-mval" style="color:var(--success)">{aw:.0f}m</div>
    <div class="ap-mlbl">After Action</div>
  </div>
  <div class="ap-metric">
    <div class="ap-mval" style="color:var(--success)">−{saved:.0f}m</div>
    <div class="ap-mlbl">Saved</div>
  </div>
  {"<div class='ap-metric'><div class='ap-mval'>" + str(pm) + "</div><div class='ap-mlbl'>Moved</div></div>" if pm else ""}
</div>"""

    # AI alert part
    ai_src  = ("llama-3.3-70b via Groq" if alert_res and alert_res.get("source") == "groq" else "Template (offline)")
    ai_text = alert_res.get("alert_text", "Alert will appear once data is available.") if alert_res else "—"
    ai_action = rec.get("action", "no_action")
    ai_sev  = {"redirect":"medium","open_counter":"high","no_action":"low","monitor":"medium"}.get(ai_action,"low")
    ai_sc   = {"high":"var(--danger)","medium":"var(--warning)","low":"var(--success)"}.get(ai_sev,"var(--text-tertiary)")
    ai_sb   = {"high":"var(--danger-muted)","medium":"var(--warning-muted)","low":"var(--success-muted)"}.get(ai_sev,"var(--surface)")
    ai_bd   = {"high":"var(--danger-border)","medium":"var(--warning-border)","low":"var(--success-border)"}.get(ai_sev,"var(--border)")
    ai_slbl = {"high":"HIGH RISK","medium":"MODERATE","low":"NORMAL"}.get(ai_sev,"INFO")

    st.markdown(f"""
<div class="action-panel">
  <div class="ap-header {action}">
    <div>
      <div class="ap-eyebrow">Recommended Action</div>
      <div class="ap-title">{rec.get('title', titles.get(action, ''))}</div>
    </div>
    <span class="ap-alert-sev" style="background:{sev_bg};border-color:{sev_bd};color:{sev_c}">
      <span style="width:5px;height:5px;border-radius:50%;background:{sev_c};display:inline-block"></span>
      {sev_txt}
    </span>
  </div>
  <div class="ap-body">
    <div class="ap-msg">{msg}</div>
    {why_html}
    {imp_html}
  </div>
  <div class="ap-divider"></div>
  <div class="ap-alert">
    <div class="ap-alert-hdr">
      <div class="ap-eyebrow">AI Narrative Alert</div>
      <span class="ap-alert-sev" style="background:{ai_sb};border-color:{ai_bd};color:{ai_sc}">
        <span style="width:5px;height:5px;border-radius:50%;background:{ai_sc};display:inline-block"></span>
        {ai_slbl}
      </span>
    </div>
    <div class="ap-alert-text">{ai_text}</div>
    <div class="ap-alert-src">Model: {ai_src}</div>
  </div>
</div>""", unsafe_allow_html=True)
    if alert_res and alert_res.get("error"):
        st.caption(f"ℹ {alert_res['error']}")


def render_mm1_expander(counter_configs: list[dict]) -> None:
    with st.expander("M/M/1 Queueing Theory Model", expanded=False):
        st.markdown("""
**M/M/1**: single server, Poisson arrivals (λ), exponential service (μ).

| Symbol | Meaning |
|--------|---------|
| ρ = λ/μ | Traffic intensity (must be < 1) |
| Wq = λ/(μ(μ−λ)) | Mean wait time (minutes) |
| Lq = ρ²/(1−ρ) | Mean queue length |
""")
        rows = []
        for c in counter_configs:
            if c.get("arrival_rate"):
                mm1 = estimate_wait_time_mm1(c["arrival_rate"], c["service_rate"])
                rows.append({"Counter": c["name"], "λ/min": c["arrival_rate"], "μ/min": c["service_rate"],
                             "ρ": mm1.get("rho","—"), "Wq (min)": mm1.get("Wq_minutes","∞"), "Stable": "✓" if mm1.get("stable") else "✗"})
            else:
                rows.append({"Counter": c["name"], "λ/min": "N/A", "μ/min": c["service_rate"], "ρ": "—", "Wq (min)": "—", "Stable": "—"})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def render_prediction_expander(preds: list[dict]) -> None:
    with st.expander("Prediction Analysis", expanded=False):
        for p in preds:
            if p.get("sentence"):
                fn = st.error if p.get("alert") else (st.warning if p.get("capped") else st.info)
                fn(p["sentence"])
            if p.get("slope") is not None:
                st.caption(f"Trend: {p['slope']:+.2f} ppl/min · R² = {p.get('r_squared','?')}" +
                           ("  · ⚠ capped" if p.get("capped") else ""))


def render_multi_horizon_expander(preds_multi: list[dict]) -> None:
    if not preds_multi:
        return
    with st.expander("Multi-Horizon Forecast (+5/+10/+15/+20 min)", expanded=False):
        rows = []
        for pm in preds_multi:
            if not pm.get("has_data"):
                continue
            row = {"Counter": pm.get("name", "?")}
            for h in FORECAST_HORIZONS:
                hp = pm["horizons"].get(h, {})
                pc = hp.get("predicted_count")
                rl = hp.get("range_low"); rh = hp.get("range_high")
                row[f"+{h}m"] = f"{pc} [{rl}–{rh}]" if pc is not None else "—"
            row["Trend"]  = f"{pm.get('slope',0):+.2f} ppl/min" if pm.get("slope") else "—"
            row["Method"] = pm.get("forecast_method", "—")
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            st.caption("Range = [low–high] ≈ ±1 std empirical interval")
        else:
            st.info(f"Need at least {MIN_READINGS} readings per counter.")


def render_what_if_panel(readings, wait_times, counter_configs, service_rate) -> None:
    with st.expander("What-If Simulator", expanded=False):
        st.markdown("Model the effect of interventions using the actual queue equations.")
        scenario = st.selectbox("Scenario", ["Open Additional Counter", "Redirect Customers", "Increase Service Rate"], key="what_if_choice")
        counters_simple = [
            {"name": c["name"], "people_count": (c["history"][-1]["people_count"] if c.get("history") else 0)}
            for c in counter_configs
        ]
        if scenario == "Open Additional Counter":
            absorption = st.slider("Load absorbed (%)", 20, 70, 45, 5, key="wi_abs") / 100
            result = scenario_open_counter(counters_simple, wait_times, service_rate, absorption=absorption)
        elif scenario == "Redirect Customers":
            result = scenario_redirect_customers(counters_simple, wait_times, service_rate)
        else:
            factor = st.slider("Service rate multiplier", 1.2, 3.0, 1.5, .1, key="wi_factor")
            result = scenario_increase_service(counters_simple, wait_times, service_rate, factor=factor)

        if "error" in result:
            st.warning(result["error"])
        else:
            st.success(f"**{result['scenario_label']}**")
            st.info(result["summary"])
            bw = result.get("before_wait") or result.get("wait_from_before")
            aw = result.get("after_wait") or result.get("wait_from_after")
            if bw and aw:
                c1, c2, c3 = st.columns(3)
                c1.metric("Before", f"{bw:.0f} min")
                c2.metric("After",  f"{aw:.0f} min", delta=f"{aw-bw:.0f} min", delta_color="inverse")
                c3.metric("Saved",  f"{bw-aw:.0f} min")


def render_full_dashboard(
    readings, wait_times, preds, counter_configs, rec, alert_res,
    horizon, threshold=30, preds_multi=None, service_rate=1.0, theme="dark",
) -> None:
    """Single entry-point — all three data modes render through here."""
    # 1. Compact status bar (replaces redundant hero + 5-column metrics)
    render_status_bar(readings, wait_times, preds, rec, threshold, theme)
    # 2. Operational intelligence (4 questions — integrated)
    render_intel_cards(readings, wait_times, preds, rec, horizon)
    # 3. Counter cards (primary view — no redundant table)
    render_counter_cards(readings, wait_times, preds)
    # 4. Altair forecast chart (theme-aware)
    render_forecast_chart(counter_configs, preds, threshold, horizon, theme)
    # 5. Action intelligence (recommendation + AI alert merged)
    st.markdown('<div class="sec-hdr">Action Intelligence</div>', unsafe_allow_html=True)
    render_action_panel(rec, alert_res)
    # 6. Detailed analysis (expandable — out of primary flow)
    st.markdown('<div class="sec-hdr">Detailed Analysis</div>', unsafe_allow_html=True)
    if preds_multi:
        render_multi_horizon_expander(preds_multi)
    render_prediction_expander(preds)
    render_mm1_expander(counter_configs)
    render_what_if_panel(readings, wait_times, counter_configs, service_rate)


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_login_page(theme: str) -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
#MainMenu,footer,header{visibility:hidden!important;}
[data-testid="stHeader"]{display:none!important;}
[data-testid="stSidebar"]{display:none!important;}
.stApp,[data-testid="stAppViewContainer"]{background:#0D1117!important;min-height:100vh;}
.block-container{padding:0!important;max-width:100%!important;}
[data-testid="stHorizontalBlock"]{gap:0!important;min-height:100vh;align-items:stretch;}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child{
  background:linear-gradient(145deg,#060C13 0%,#0D1117 45%,#111820 75%,#0C1829 100%)!important;
  padding:0!important;overflow:hidden;position:relative;}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{
  background:#FFFFFF!important;padding:0!important;display:flex;align-items:center;justify-content:center;}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child>div{
  width:100%;max-width:380px;padding:48px 40px;margin:0 auto;}
.stTextInput>div>div{border:1.5px solid #D0D7DE!important;border-radius:7px!important;background:#FFFFFF!important;}
.stTextInput input{color:#1F2328!important;background:#FFFFFF!important;caret-color:#0969DA!important;font-size:14px!important;}
.stTextInput input::placeholder{color:#8C959F!important;opacity:1!important;}
.stTextInput>div>div:focus-within{border-color:#0969DA!important;box-shadow:0 0 0 3px rgba(9,105,218,.12)!important;}
.stButton>button{
  background:linear-gradient(135deg,#0757BA,#0969DA,#0A79F5)!important;
  color:#fff!important;border:none!important;border-radius:7px!important;
  font-weight:600!important;font-size:14px!important;padding:12px 0!important;
  box-shadow:0 3px 14px rgba(9,105,218,.35)!important;transition:all .18s!important;}
.stButton>button:hover{box-shadow:0 6px 24px rgba(9,105,218,.50)!important;transform:translateY(-1px)!important;}
</style>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
<div class="lp-bg"></div>
<div class="lp-orb"></div>
<div class="lp-wrap">
  <div class="lp-logo">Q</div>
  <div class="lp-brand">QueueIQ</div>
  <div class="lp-tagline">Queue Intelligence Platform</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;position:relative;z-index:1;margin-bottom:32px">
    <div class="lp-stat"><div class="lp-sv">3</div><div class="lp-sl">Modes</div></div>
    <div class="lp-stat"><div class="lp-sv">AI</div><div class="lp-sl">Forecast</div></div>
    <div class="lp-stat"><div class="lp-sv">Live</div><div class="lp-sl">CCTV</div></div>
    <div class="lp-stat"><div class="lp-sv">M/M/1</div><div class="lp-sl">Model</div></div>
  </div>
  <div style="position:relative;z-index:1;width:100%;max-width:260px">
    <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.12em;color:rgba(47,129,247,.55);text-align:center;margin-bottom:12px">Data Flow</div>
    <div style="display:flex;flex-direction:column;gap:5px;align-items:center">
      <div style="background:rgba(47,129,247,.09);border:1px solid rgba(47,129,247,.20);border-radius:5px;padding:6px 20px;font-size:11px;color:rgba(139,148,158,.8)">Arrivals 1.8/min</div>
      <div style="font-size:11px;color:rgba(47,129,247,.3)">↓</div>
      <div style="background:rgba(47,129,247,.12);border:1px solid rgba(47,129,247,.28);border-radius:5px;padding:6px 20px;font-size:11px;color:#79C0FF">Queue · 27 waiting</div>
      <div style="font-size:11px;color:rgba(47,129,247,.3)">↓</div>
      <div style="display:flex;gap:8px">
        <div style="text-align:center">
          <div style="background:rgba(248,81,73,.08);border:1px solid rgba(248,81,73,.28);border-radius:5px;padding:4px 10px;font-size:10px;color:#FF7B72">C-01 HIGH</div>
        </div>
        <div style="text-align:center">
          <div style="background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.28);border-radius:5px;padding:4px 10px;font-size:10px;color:#56D364">C-02 LOW</div>
        </div>
        <div style="text-align:center">
          <div style="background:rgba(210,153,34,.08);border:1px solid rgba(210,153,34,.28);border-radius:5px;padding:4px 10px;font-size:10px;color:#E3B341">C-03 MED</div>
        </div>
      </div>
      <div style="font-size:11px;color:rgba(47,129,247,.3)">↓</div>
      <div style="background:rgba(47,129,247,.07);border:1px solid rgba(47,129,247,.15);border-radius:5px;padding:6px 20px;font-size:11px;color:rgba(139,148,158,.6)">AI Recommendation</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    with right:
        st.markdown(
            "<div class='rp-lbl'>Secure Access</div>"
            "<div class='rp-title' style='color:#1F2328'>Sign in to<br>QueueIQ</div>"
            "<div class='rp-sub' style='color:#57606A'>Access real-time queue intelligence,<br>AI forecasting, and live CCTV analysis.</div>"
            "<span style='font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.09em;color:#8C959F;margin-bottom:8px;display:block'>Admin Password</span>",
            unsafe_allow_html=True,
        )
        pwd = st.text_input("Password", type="password", placeholder="Enter password…",
                            key="login_pwd", label_visibility="collapsed")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Sign In →", key="login_btn", width='stretch'):
            if pwd == ADMIN_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            elif pwd:
                st.error("Incorrect password.")
        st.markdown(
            "<div class='rp-hint' style='color:#57606A'>Demo password: <code style='background:rgba(9,105,218,.10);color:#0969DA;padding:1px 6px;border-radius:4px;font-size:11px'>queueiq2024</code></div>"
            "<div class='rp-foot'>QueueIQ v2.0 <span style='margin:0 6px;color:#CFD4DB'>·</span> AI Operations <span style='margin:0 6px;color:#CFD4DB'>·</span> Secured</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DEMO SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
_DEMO_SCENARIOS = {
    "Normal (low traffic)":  {"preset":"Bank","n_counters":3,"overrides":[5,4,6]},
    "Rush Hour":             {"preset":"Bank","n_counters":3,"overrides":[27,22,18]},
    "Sudden Surge":          {"preset":"Bank","n_counters":3,"overrides":[38,7,12]},
    "Counter Failure":       {"preset":"Bank","n_counters":2,"overrides":[34,28]},
    "All Clear":             {"preset":"Bank","n_counters":4,"overrides":[3,2,4,1]},
}


def _load_demo_scenario(name: str) -> None:
    sc = _DEMO_SCENARIOS.get(name)
    if not sc:
        return
    _init_sim_state(sc["preset"], sc["n_counters"])
    import time as _t
    base_t = _t.time() - 600
    for i, c in enumerate(st.session_state["counters_state"]):
        target = sc["overrides"][i] if i < len(sc["overrides"]) else 5
        for ti in range(12):
            c["history"].append({
                "counter_id": c["id"], "name": c["name"],
                "people_count": int(round(target * (ti+1) / 12)),
                "timestamp_epoch": base_t + ti * 30,
                "timestamp_str": datetime.fromtimestamp(base_t + ti * 30).strftime("%H:%M"),
            })
    st.session_state["tick_count"] = 12
    st.toast(f"Loaded: {name}", icon="✓")


def _init_sim_state(preset: str, n_ctr: int) -> None:
    st.session_state.update({
        "counters_state": build_counters_state(preset, n_ctr),
        "tick_count": 0, "last_readings": [], "last_wait_times": [],
        "last_preds": [], "last_preds_multi": [], "last_rec": {}, "last_alert": None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# AUTH GATE
# ─────────────────────────────────────────────────────────────────────────────
_theme = st.session_state.get("theme", "dark")
inject_theme(_theme)

if not st.session_state.get("authenticated", False):
    render_login_page(_theme)
    st.stop()

# Restore sidebar after login page
st.markdown("""
<style>
section[data-testid="stSidebar"]{
  display:flex!important;visibility:visible!important;
  width:248px!important;min-width:248px!important;max-width:248px!important;
  transform:none!important;opacity:1!important;
}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Navigation separated from configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Brand ─────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="qiq-brand">
  <div class="qiq-logo">Q</div>
  <div>
    <div class="qiq-wordmark">QueueIQ</div>
    <div class="qiq-tagline">Queue Intelligence Platform</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Theme toggle (single full-width row) ───────────────────────────────────
    _is_dark  = _theme == "dark"
    _tog_lbl  = "Switch to Light Mode" if _is_dark else "Switch to Dark Mode"
    _tog_icon = "[Light]" if _is_dark else "[Dark]"
    st.markdown(
        "<div style='padding:8px 0 4px'></div>",
        unsafe_allow_html=True,
    )
    if st.button(
        _tog_lbl,
        key="theme_btn",
        help=_tog_lbl,
    ):
        st.session_state["theme"] = "light" if _is_dark else "dark"
        st.rerun()
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Data source navigation ────────────────────────────────────────────────
    st.markdown('<span class="sb-section">Data Source</span>', unsafe_allow_html=True)
    mode_choice = st.radio(
        "mode_radio",
        ["Simulation", "Upload CSV", "Camera / Vision"],
        label_visibility="collapsed",
        key="mode_choice",
    )
    is_simulation = (mode_choice == "Simulation")
    is_cv         = (mode_choice == "Camera / Vision")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Configuration (grouped separately from navigation) ────────────────────
    st.markdown('<span class="sb-section">Prediction Settings</span>', unsafe_allow_html=True)
    horizon_min = st.slider("Horizon (min)", 5, 60, 20, 5, key="horizon_min")
    threshold   = st.slider("Alert threshold (ppl)", 10, 60, 30, 5, key="threshold")

    if is_simulation:
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="sb-section">Queue Setup</span>', unsafe_allow_html=True)
        preset_choice = st.selectbox("Location", list(PRESETS.keys()), key="preset_choice",
                                     label_visibility="collapsed")
        n_counters = st.slider("Counters", 1, 6, 3, 1, key="n_counters")

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="sb-section">Scenarios</span>', unsafe_allow_html=True)
        demo_scenario = st.selectbox("Load scenario",
            ["— Manual —","Normal (low traffic)","Rush Hour","Sudden Surge","Counter Failure","All Clear"],
            key="demo_scenario", label_visibility="collapsed")
        if st.button("▶ Load Scenario", key="load_demo", width='stretch'):
            _load_demo_scenario(demo_scenario)

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="sb-section">Playback</span>', unsafe_allow_html=True)
        running      = st.toggle("Auto-play", value=False, key="running")
        _pa, _pb     = st.columns(2)
        with _pa: advance_tick = st.button("⏭ Tick",  width='stretch')
        with _pb: reset_sim    = st.button("↺ Reset", width='stretch')
        tick_speed = st.select_slider("Speed", [1,2,3,5,10], value=3,
                                      format_func=lambda x: f"{x}s")

    elif not is_cv:
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="sb-section">Service Settings</span>', unsafe_allow_html=True)
        service_rate = st.slider("Service rate (cust/min)", .5, 5.0, 1.0, .5, key="service_rate",
                                 label_visibility="collapsed")
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="sb-section">Sample Data</span>', unsafe_allow_html=True)
        st.caption("Download a ready-made CSV — Counter 1 peaks at 35 ppl.")
        st.download_button("↓ Sample CSV", data=generate_sample_csv(),
                           file_name="sample_queue_data.csv", mime="text/csv",
                           width='stretch')
        st.session_state.setdefault("_uf", None)

    elif is_cv:
        if not _CV_AVAILABLE:
            st.warning("YOLOv8 unavailable. Falling back to Simulation.", icon="⚠️")
        else:
            st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
            st.markdown('<span class="sb-section">Camera Settings</span>', unsafe_allow_html=True)
            cv_service_rate = st.slider("Service rate (cust/min)", .5, 5.0, 1.0, .5,
                                        key="cv_service_rate", label_visibility="collapsed")
            st.session_state.setdefault("_cv_files", [])
            st.session_state.setdefault("_cv_labels", [])

    # ── User section ──────────────────────────────────────────────────────────
    st.markdown('<div class="sb-divider" style="margin-top:12px"></div>', unsafe_allow_html=True)
    _ua, _ub = st.columns([4, 1])
    with _ua:
        st.markdown(
            '<div style="font-size:12px;font-weight:600;color:var(--text-primary)">'
            '<span class="sb-user-dot"></span>Admin</div>'
            '<div style="font-size:11px;color:var(--text-tertiary)">Active session</div>',
            unsafe_allow_html=True)
    with _ub:
        if st.button("⏏", key="logout_btn", help="Sign out"):
            st.session_state["authenticated"] = False
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SAFE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
advance_tick  = advance_tick  if is_simulation else False
reset_sim     = reset_sim     if is_simulation else False
running       = running       if is_simulation else False
tick_speed    = tick_speed    if is_simulation else 3
service_rate  = st.session_state.get("service_rate", 1.0)
preset_choice = st.session_state.get("preset_choice", list(PRESETS.keys())[0])
n_counters    = int(st.session_state.get("n_counters", 3))

readings = wait_times = preds = preds_multi = []
rec = {}; alert_res = None; counter_configs = []; header_info = ""


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION TICK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_sim_tick(counters_state, horizon, thresh):
    readings = generate_reading(counters_state, tick_seconds=30)
    st.session_state["last_readings"] = readings
    st.session_state["tick_count"]   += 1
    tick = st.session_state["tick_count"]

    wts = [estimate_wait_time(readings[i]["people_count"], counters_state[i]["service_rate"])
           for i in range(len(counters_state))]
    st.session_state["last_wait_times"] = wts

    preds = []
    for i, c in enumerate(counters_state):
        p = predict_future_count(c["history"], horizon_minutes=horizon,
                                 counter_name=c["name"], threshold=thresh)
        p["counter_id"] = c["id"]
        preds.append(p)
    st.session_state["last_preds"] = preds

    preds_multi = []
    for c in counters_state:
        mh = predict_multi_horizon(c["history"], counter_name=c["name"], threshold=thresh)
        mh["name"] = c["name"]
        preds_multi.append(mh)
    st.session_state["last_preds_multi"] = preds_multi

    current = [{"counter_id": r["counter_id"], "name": r["name"], "people_count": r["people_count"]}
               for r in readings]
    sr  = counters_state[0]["service_rate"] if counters_state else 1.0
    rec = recommend_action(current, preds, wts, service_rate=sr)
    st.session_state["last_rec"] = rec

    if tick == 1 or tick % 3 == 0:
        states = [{"name": r["name"], "people_count": r["people_count"], "wait_time": wts[i]}
                  for i, r in enumerate(readings)]
        st.session_state["last_alert"] = generate_alert(states, preds, rec)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTATION — MODE ROUTING
# ─────────────────────────────────────────────────────────────────────────────

if is_simulation:
    prev_p = st.session_state.get("_prev_preset")
    prev_n = st.session_state.get("_prev_n_ctr")
    cfg_changed = (preset_choice != prev_p) or (n_counters != prev_n)

    if "counters_state" not in st.session_state or reset_sim:
        _init_sim_state(preset_choice, n_counters)
        st.session_state["_prev_preset"] = preset_choice
        st.session_state["_prev_n_ctr"]  = n_counters
    elif cfg_changed:
        _init_sim_state(preset_choice, n_counters)
        st.session_state["_prev_preset"] = preset_choice
        st.session_state["_prev_n_ctr"]  = n_counters

    counters_state  = st.session_state["counters_state"]
    counter_configs = counters_state
    _tc_now = st.session_state.get("tick_count", 0)
    if advance_tick or running or (_tc_now == 0 and not reset_sim):
        run_sim_tick(counters_state, horizon_min, threshold)

    readings     = st.session_state.get("last_readings",    [])
    wait_times   = st.session_state.get("last_wait_times",  [])
    preds        = st.session_state.get("last_preds",       [])
    preds_multi  = st.session_state.get("last_preds_multi", [])
    rec          = st.session_state.get("last_rec",         {})
    alert_res    = st.session_state.get("last_alert")
    tick_n       = st.session_state.get("tick_count", 0)
    header_info  = f"Tick #{tick_n} · {preset_choice} · {len(counters_state)} counter{'s' if len(counters_state)!=1 else ''}"

    if running:
        import time as _t
        _t.sleep(tick_speed)
        st.rerun()

elif is_cv:
    cv_files  = st.session_state.get("_cv_files", [])
    cv_labels = st.session_state.get("_cv_labels", [])
    cv_sr     = st.session_state.get("cv_service_rate", 1.0)

    if not _CV_AVAILABLE:
        is_simulation = True; is_cv = False
        header_info   = "Camera unavailable — using Simulation"
        if "counters_state" not in st.session_state:
            _init_sim_state("Bank", 3)
        counters_state  = st.session_state["counters_state"]
        counter_configs = counters_state
        run_sim_tick(counters_state, horizon_min, threshold)
        readings    = st.session_state.get("last_readings",   [])
        wait_times  = st.session_state.get("last_wait_times", [])
        preds       = st.session_state.get("last_preds",      [])
        rec         = st.session_state.get("last_rec",        {})
        alert_res   = st.session_state.get("last_alert")
    elif not cv_files:
        header_info = "Camera mode — upload footage below"
    else:
        cv_cache_key = "_".join(f"{f.name}{f.size}" for f in cv_files) + str(cv_sr)
        if st.session_state.get("_cv_cache_key") != cv_cache_key:
            try:
                yolo_model = _get_yolo_model()
                with st.spinner("Running YOLOv8 person detection…"):
                    for f in cv_files: f.seek(0)
                    labels_to_use = cv_labels or [f"Counter {i+1}" for i in range(len(cv_files))]
                    cv_configs = _cvd.build_cv_counter_configs(cv_files, labels_to_use, cv_sr, model=yolo_model)
                if cv_configs:
                    st.session_state.update({"_cv_configs": cv_configs, "_cv_cache_key": cv_cache_key, "_cv_groq_key": None})
                else:
                    st.warning("No people detected. Try a clearer image."); cv_configs = []
            except Exception:
                st.warning("Camera mode unavailable — falling back to Simulation."); cv_configs = []
        else:
            cv_configs = st.session_state.get("_cv_configs", [])

        if cv_configs:
            counter_configs = cv_configs
            readings   = [c["history"][-1] for c in counter_configs if c["history"]]
            wait_times = [estimate_wait_time(r["people_count"], counter_configs[i]["service_rate"])
                          for i, r in enumerate(readings)]
            preds = []
            for i, c in enumerate(counter_configs):
                p = predict_future_count(c["history"], horizon_minutes=horizon_min,
                                         counter_name=c["name"], threshold=threshold)
                p["counter_id"] = c["id"]; preds.append(p)
            current  = [{"counter_id": r["counter_id"], "name": r["name"], "people_count": r["people_count"]} for r in readings]
            rec      = recommend_action(current, preds, wait_times)
            ck       = cv_cache_key + rec.get("action","")
            if st.session_state.get("_cv_groq_key") != ck:
                states   = [{"name": r["name"], "people_count": r["people_count"], "wait_time": wait_times[i]} for i, r in enumerate(readings)]
                alert_res = generate_alert(states, preds, rec)
                st.session_state.update({"_cv_alert": alert_res, "_cv_groq_key": ck})
            else:
                alert_res = st.session_state.get("_cv_alert")
            frames = sum(len(c["history"]) for c in counter_configs)
            header_info = f"{len(counter_configs)} counters · {frames} frames · svc {cv_sr}/min"

            def _build_cv_csv(configs):
                import io as _io
                rows = []
                for cfg in configs:
                    for h in cfg.get("history", []):
                        rows.append({"timestamp": h.get("timestamp_str", ""), "counter_name": h.get("name",""), "people_count": h.get("people_count",0)})
                if not rows:
                    for r in readings:
                        rows.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "counter_name": r.get("name",""), "people_count": r.get("people_count",0)})
                buf = _io.BytesIO(); pd.DataFrame(rows).to_csv(buf, index=False); return buf.getvalue()

            st.session_state["_cv_export_csv"]  = _build_cv_csv(counter_configs)
            st.session_state["_cv_export_name"] = f"cctv_queue_{len(counter_configs)}_counters.csv"

else:  # Upload CSV
    uf = st.session_state.get("_uf")
    if uf is not None:
        fk = f"{uf.name}_{uf.size}"
        if st.session_state.get("_csv_key") != fk:
            try:
                df = pd.read_csv(uf)
                configs_raw = parse_csv_raw(df)
                st.session_state.update({"_configs_raw": configs_raw, "_csv_key": fk, "_groq_key": None})
            except Exception as e:
                st.exception(e); configs_raw = []
        else:
            configs_raw = st.session_state.get("_configs_raw", [])

        if configs_raw:
            counter_configs = apply_service_rate(configs_raw, service_rate)
            readings   = [c["history"][-1] for c in counter_configs if c["history"]]
            wait_times = [estimate_wait_time(r["people_count"], counter_configs[i]["service_rate"]) for i, r in enumerate(readings)]
            preds = []
            for i, c in enumerate(counter_configs):
                p = predict_future_count(c["history"], horizon_minutes=horizon_min, counter_name=c["name"], threshold=threshold)
                p["counter_id"] = c["id"]; preds.append(p)
            current = [{"counter_id": r["counter_id"], "name": r["name"], "people_count": r["people_count"]} for r in readings]
            rec     = recommend_action(current, preds, wait_times)
            gk      = f"{fk}_{service_rate}_{rec.get('action')}"
            if st.session_state.get("_groq_key") != gk:
                states    = [{"name": r["name"], "people_count": r["people_count"], "wait_time": wait_times[i]} for i, r in enumerate(readings)]
                alert_res = generate_alert(states, preds, rec)
                st.session_state.update({"_real_alert": alert_res, "_groq_key": gk})
            else:
                alert_res = st.session_state.get("_real_alert")
            total_rows  = sum(len(c["history"]) for c in counter_configs)
            header_info = f"{len(counter_configs)} counters · {total_rows} readings · svc {service_rate}/min"
        else:
            header_info = "CSV parse failed — check column names"
    else:
        header_info = "No file uploaded"


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
mode_str = "simulation" if is_simulation else ("cv" if is_cv else "real_data")
render_header(mode_str, header_info, _theme)

# ── Upload empty state + uploader ─────────────────────────────────────────────
if not is_simulation and not is_cv:
    if not readings:
        st.markdown("""
<div class="drop-zone">
  <div class="dz-icon">📂</div>
  <div class="dz-title">Upload your queue data CSV</div>
  <div class="dz-sub">
    Import operational data to generate AI forecasts, wait-time estimates,
    risk classifications, and actionable recommendations.<br><br>
    <strong>Required columns:</strong> timestamp · counter_name · people_count
  </div>
  <div class="dz-tags">
    <span class="dz-tag">timestamp</span>
    <span class="dz-tag">counter_name</span>
    <span class="dz-tag">people_count</span>
    <span class="dz-tag">.csv</span>
  </div>
</div>
<div style="height:12px"></div>""", unsafe_allow_html=True)
    _up = st.file_uploader("Upload queue data CSV", type=["csv"],
                            help="Required: timestamp, counter_name, people_count",
                            key="csv_uploader")
    st.session_state["_uf"] = _up

# ── Camera workspace ───────────────────────────────────────────────────────────
elif is_cv and _CV_AVAILABLE:
    _t1, _t2 = st.tabs(["📁 Upload Footage", "📸 Live Camera"])

    with _t1:
        if not readings:
            st.markdown("""
<div class="drop-zone">
  <div class="dz-icon">📷</div>
  <div class="dz-title">Upload footage to detect people with AI</div>
  <div class="dz-sub">
    One file per counter — JPG/PNG image or MP4/AVI/MOV video.<br>
    YOLOv8n detects people and feeds the count into the queue engine.<br>
    <em>Processes uploaded footage — not a live real-time feed.</em>
  </div>
  <div class="dz-tags">
    <span class="dz-tag">.jpg</span><span class="dz-tag">.png</span>
    <span class="dz-tag">.mp4</span><span class="dz-tag">.avi</span><span class="dz-tag">.mov</span>
  </div>
</div>
<div style="height:12px"></div>""", unsafe_allow_html=True)
        _cv_files = st.file_uploader("Upload footage — one file per counter",
                                      type=["jpg","jpeg","png","mp4","avi","mov"],
                                      accept_multiple_files=True, key="cv_uploader",
                                      help="One file = one counter")
        st.session_state["_cv_files"] = _cv_files or []
        if _cv_files:
            st.markdown("**Name each counter:**")
            cv_lbls = []
            for i, f in enumerate(st.columns(min(len(_cv_files), 3))):
                if i < len(_cv_files):
                    with f:
                        cv_lbls.append(st.text_input(f"File {i+1}: {_cv_files[i].name}",
                                                      value=f"Counter {i+1}", key=f"cv_lbl_{i}"))
            st.session_state["_cv_labels"] = cv_lbls

    with _t2:
        from datetime import datetime as _dtl
        import io as _iol
        st.markdown("""
<div style="background:var(--accent-muted);border:1px solid var(--accent-border);
border-radius:var(--r-md);padding:12px 16px;margin-bottom:14px">
  <div style="font-size:13px;font-weight:600;color:var(--accent);margin-bottom:3px">📷 Live Camera Counter</div>
  <div style="font-size:12px;color:var(--text-secondary);line-height:1.6">
    Capture frames from your camera. AI counts people and records the data.
    Capture multiple frames to build a time-series — then export as CSV.
  </div>
</div>""", unsafe_allow_html=True)
        _lcn = st.text_input("Counter name", value=st.session_state.get("_live_ctr_name","Live Camera"),
                             key="_live_ctr_nm", placeholder="e.g. Main Hall Counter 1")
        st.session_state["_live_ctr_name"] = _lcn
        _cam = st.camera_input("Point camera at queue → Take Photo", key="live_cam")
        _lc1, _lc2 = st.columns(2)
        with _lc1: _proc = st.button("🔍 Detect & Record", key="live_det",
                                      width='stretch', disabled=(_cam is None))
        with _lc2: _clr  = st.button("↺ Clear",          key="live_clr", width='stretch')
        if _clr:
            st.session_state["_live_history"] = []; st.session_state["_live_export_csv"] = None
            st.toast("History cleared.", icon="↺")
        if _proc and _cam:
            try:
                _cam.seek(0); _bytes = _cam.read()
                import tempfile, os as _os
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as _tmp:
                    _tmp.write(_bytes); _tp = _tmp.name
                class _FF:
                    def __init__(self,path,name):
                        self.name=name; self._d=open(path,"rb").read(); self._p=0; self.size=len(self._d)
                    def read(self,n=-1):
                        d=self._d[self._p:] if n==-1 else self._d[self._p:self._p+n]
                        self._p+=len(d); return d
                    def seek(self,p): self._p=p
                ym = _get_yolo_model()
                with st.spinner("Running YOLOv8…"):
                    _lc = _cvd.build_cv_counter_configs([_FF(_tp,f"{_lcn}.jpg")],[_lcn],
                                st.session_state.get("cv_service_rate",1.0),model=ym)
                if _os.path.exists(_tp): _os.unlink(_tp)
                if _lc and _lc[0].get("history"):
                    _dc = _lc[0]["history"][-1]["people_count"]
                    _ts = _dtl.now().strftime("%Y-%m-%d %H:%M:%S")
                    _h  = st.session_state.get("_live_history",[])
                    _h.append({"timestamp":_ts,"counter_name":_lcn,"people_count":_dc})
                    st.session_state["_live_history"] = _h
                    _b = _iol.BytesIO(); pd.DataFrame(_h).to_csv(_b,index=False)
                    st.session_state["_live_export_csv"] = _b.getvalue()
                    st.success(f"✅ **{_dc} people** detected at {_ts}")
                    st.session_state.update({"_cv_configs":_lc,"_cv_cache_key":f"live_{_ts}","_cv_groq_key":None})
                else:
                    st.warning("No people detected — try better lighting.")
            except Exception as _e:
                st.error(f"Detection error: {_e}")
        _lh = st.session_state.get("_live_history",[])
        if _lh:
            st.markdown(f"**{len(_lh)} reading(s)**")
            st.dataframe(pd.DataFrame(_lh), width='stretch', height=180)
            _m1,_m2,_m3 = st.columns(3)
            _m1.metric("Latest", _lh[-1]["people_count"])
            _m2.metric("Peak",   max(r["people_count"] for r in _lh))
            _m3.metric("Avg",    f"{sum(r['people_count'] for r in _lh)/len(_lh):.1f}")
            st.download_button("↓ Export CSV", data=st.session_state["_live_export_csv"],
                               file_name=f"live_{_lcn.replace(' ','_')}.csv", mime="text/csv",
                               key="live_csv_exp", width='stretch')

# ── Simulation placeholder ─────────────────────────────────────────────────────
elif is_simulation and not readings:
    st.markdown("""
<div class="sim-ready">
  <div style="font-size:28px;opacity:.5;margin-bottom:12px">◉</div>
  <div style="font-size:14px;font-weight:600;color:var(--text-primary);margin-bottom:5px">Simulation Ready</div>
  <div style="font-size:12px;color:var(--text-secondary);line-height:1.6">
    Press <strong>⏭ Tick</strong> in the sidebar to generate the first reading,<br>
    enable <strong>Auto-play</strong> for continuous simulation,<br>
    or load a <strong>Demo Scenario</strong> to jump to an interesting state.
  </div>
</div>""", unsafe_allow_html=True)

# ── Dashboard ──────────────────────────────────────────────────────────────────
if readings:
    _sr = service_rate if not is_simulation else (
        st.session_state["counters_state"][0]["service_rate"]
        if st.session_state.get("counters_state") else 1.0
    )
    render_full_dashboard(
        readings, wait_times, preds, counter_configs, rec, alert_res, horizon_min,
        threshold=threshold,
        preds_multi=preds_multi or None,
        service_rate=_sr,
        theme=_theme,
    )

    # ── CCTV export panel ──────────────────────────────────────────────────────
    if is_cv and st.session_state.get("_cv_export_csv"):
        st.markdown("---")
        st.markdown("""
<div style="background:var(--accent-muted);border:1px solid var(--accent-border);
border-radius:var(--r-md);padding:14px 18px">
  <div style="font-size:13px;font-weight:600;color:var(--accent);margin-bottom:3px">↓ Export Detected Data</div>
  <div style="font-size:12px;color:var(--text-secondary)">
    Download AI-detected queue data as CSV.<br>
    Upload to <strong>📂 Upload CSV</strong> for full analysis.
  </div>
</div>""", unsafe_allow_html=True)
        _ec1, _ec2, _ = st.columns([2, 2, 1])
        with _ec1:
            st.download_button("↓ Download CSV",
                data=st.session_state["_cv_export_csv"],
                file_name=st.session_state.get("_cv_export_name","cctv_queue.csv"),
                mime="text/csv", key="cv_csv_exp", width='stretch')
        with _ec2:
            st.caption("Columns: timestamp · counter_name · people_count")
