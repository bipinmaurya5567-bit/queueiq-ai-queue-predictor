"""
app.py  ──  QueueIQ Dashboard
==============================
Three operating modes:
  &bull; Simulation Mode    &mdash; Poisson-based counter simulation (unchanged)
  &bull; Upload Real Data   &mdash; CSV upload feeds queue_math / predictor / recommender
  &bull; Camera/Video Feed  &mdash; YOLOv8n person detection on uploaded images/video
                          (Beta) Fully isolated; failure falls back gracefully.

Shared render_full_dashboard() drives all three modes from identical data
structures &mdash; no separate code paths beyond data sourcing.
"""

import io
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from simulator   import build_counters_state, generate_reading, PRESETS
from queue_math  import estimate_wait_time, estimate_wait_time_mm1
from predictor   import predict_future_count, predict_multi_horizon, MIN_READINGS
from recommender import recommend_action
from groq_alerts import generate_alert
from risk_engine import classify_risk, classify_facility_risk
from what_if     import scenario_open_counter, scenario_redirect_customers, scenario_increase_service
from config      import FORECAST_HORIZONS, RISK_LABELS

# ── Camera/Video mode &mdash; fully isolated import ──────────────────────────────
# If cv_detector or its deps are missing, _CV_AVAILABLE stays False and
# the rest of the app continues working with zero impact.
_CV_AVAILABLE = False
try:
    import cv_detector as _cvd
    _CV_AVAILABLE = _cvd._YOLO_AVAILABLE
except Exception:
    pass

@st.cache_resource(show_spinner=False)
def _get_yolo_model():
    """Load YOLOv8n once; cached by Streamlit so reruns are free."""
    if not _CV_AVAILABLE:
        return None
    try:
        return _cvd.load_yolo_model()
    except Exception:
        return None

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QueueIQ &mdash; Real-time Queue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# DESIGN SYSTEM  ── CSS variables + utility classes
# ─────────────────────────────────────────────────────────
COUNTER_COLORS = ["#00C2FF","#10B981","#F59E0B","#EF4444","#A78BFA","#34D399"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════════════════════
   QUEUEIQ COMMAND CENTER &mdash; DARK OPERATIONAL THEME
   ═══════════════════════════════════════════════════════ */
:root {
  /* Backgrounds &mdash; layered dark navy */
  --bg-primary:   #070D1A;
  --bg-secondary: #0C1526;
  --surface:      #101E33;
  --surface-el:   #152338;
  --surface-hi:   #1A2B42;

  /* Borders */
  --border:       rgba(255,255,255,.07);
  --border-accent:rgba(0,194,255,.20);

  /* Brand / Accent */
  --accent:       #00C2FF;
  --accent2:      #0080FF;
  --accent-dim:   rgba(0,194,255,.12);
  --accent-glow:  0 0 20px rgba(0,194,255,.18);

  /* Semantic */
  --success:      #10B981;
  --success-dim:  rgba(16,185,129,.10);
  --success-bd:   rgba(16,185,129,.30);
  --warning:      #F59E0B;
  --warning-dim:  rgba(245,158,11,.10);
  --warning-bd:   rgba(245,158,11,.30);
  --danger:       #EF4444;
  --danger-dim:   rgba(239,68,68,.08);
  --danger-bd:    rgba(239,68,68,.28);
  --critical:     #DC2626;
  --critical-dim: rgba(220,38,38,.08);
  --critical-bd:  rgba(220,38,38,.28);

  /* Text */
  --tx:   #E2EAF8;
  --tx2:  #8899BB;
  --tx3:  #4A5A78;
  --tx4:  #2A3A58;

  /* Counter accent colors */
  --c0: #00C2FF; --c1: #10B981; --c2: #F59E0B;
  --c3: #EF4444; --c4: #A78BFA; --c5: #34D399;

  /* Radii */
  --r6:6px; --r8:8px; --r10:10px; --r12:12px; --r16:16px;
}

/* ═══ RESET + BASE ═══ */
*,*::before,*::after { box-sizing:border-box; margin:0; }

html, body, [class*="css"], .stApp {
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
  background: var(--bg-primary) !important;
  color: var(--tx) !important;
}

[data-testid="stAppViewContainer"] { background: var(--bg-primary) !important; }
[data-testid="stHeader"]           { display:none !important; }
[data-testid="stToolbar"]          { display:none !important; }
#MainMenu, footer                  { visibility:hidden !important; }

/* Scrollbar */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track  { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb  { background: var(--surface-hi); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ═══ SIDEBAR &mdash; COMMAND NAV ═══ */
[data-testid="stSidebar"] {
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border) !important;
  min-width: 260px !important;
  max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 0 !important;
  overflow-x: hidden !important;
}
[data-testid="stSidebar"] * { color: var(--tx) !important; }
[data-testid="stSidebar"] .stMarkdown p { color: var(--tx2) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* Sidebar collapse button &mdash; make visible */
[data-testid="collapsedControl"] {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border) !important;
  color: var(--tx) !important;
}
[data-testid="stSidebarCollapseButton"] {
  color: var(--tx2) !important;
}

/* ═══ MAIN CONTENT AREA ═══ */
[data-testid="stMain"] {
  background: var(--bg-primary) !important;
  padding: 0 !important;
}

.block-container {
  background: var(--bg-primary) !important;
  padding: 20px 24px 40px !important;
  max-width: 100% !important;
}

/* ═══ WIDGETS ═══ */
.stSelectbox > div > div,
.stTextInput > div > div,
.stTextArea > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--tx) !important;
  border-radius: var(--r8) !important;
}
.stTextInput input,
.stTextArea textarea,
.stSelectbox select,
.stNumberInput input {
  color: var(--tx) !important;
  background: var(--surface) !important;
  caret-color: var(--accent) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
  color: var(--tx3) !important; opacity:1 !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px var(--accent-dim) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label,
.stSlider label, .stNumberInput label, .stCheckbox label,
.stRadio label, .stMultiSelect label, .stDateInput label {
  color: var(--tx2) !important; font-size:.78rem !important;
}
.stCheckbox span, .stRadio span { color: var(--tx2) !important; }
.stSlider > div > div > div > div { background: var(--accent) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: var(--r8) !important;
  gap: 2px !important;
  padding: 3px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--tx2) !important;
  border-radius: var(--r6) !important;
  font-size: .8rem !important;
  font-weight: 500 !important;
  padding: 6px 14px !important;
}
.stTabs [aria-selected="true"] {
  background: var(--surface-hi) !important;
  color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-border"] { display:none !important; }

/* Buttons */
.stButton > button {
  background: var(--surface-hi) !important;
  color: var(--tx) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r8) !important;
  font-weight: 600 !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: .84rem !important;
  transition: all .16s ease;
  letter-spacing: .01em;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  background: var(--accent-dim) !important;
  color: var(--accent) !important;
  transform: translateY(-1px);
}
.stDownloadButton > button {
  background: linear-gradient(135deg, rgba(0,194,255,.15), rgba(0,128,255,.15)) !important;
  color: var(--accent) !important;
  border: 1px solid var(--border-accent) !important;
  border-radius: var(--r8) !important;
  font-weight: 700 !important;
}
.stDownloadButton > button:hover { opacity:.85; }

/* Expander */
.stExpander {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r10) !important;
}
.stExpander summary { color: var(--tx2) !important; font-size:.85rem !important; }
.stExpander summary:hover { color: var(--tx) !important; }
.stExpander [data-testid="stExpanderDetails"] p { color: var(--tx2) !important; }

/* Alerts */
.stInfo    { background: rgba(0,194,255,.06) !important; border-color: rgba(0,194,255,.20) !important; border-radius:var(--r8) !important; }
.stInfo p  { color: var(--tx) !important; }
.stWarning { background: var(--warning-dim) !important; border-color: var(--warning-bd) !important; border-radius:var(--r8) !important; }
.stWarning p { color: #FDE68A !important; }
.stSuccess { background: var(--success-dim) !important; border-color: var(--success-bd) !important; border-radius:var(--r8) !important; }
.stSuccess p { color: #A7F3D0 !important; }
.stError   { background: var(--danger-dim) !important; border-color: var(--danger-bd) !important; border-radius:var(--r8) !important; }
.stError p { color: #FCA5A5 !important; }

/* Markdown */
[data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li { color: var(--tx2) !important; }
[data-testid="stMarkdown"] h1, [data-testid="stMarkdown"] h2,
[data-testid="stMarkdown"] h3, [data-testid="stMarkdown"] h4 { color: var(--tx) !important; }
h1,h2,h3,h4 { color: var(--tx) !important; font-family:'IBM Plex Sans',sans-serif; }
p { color: var(--tx2); }
li { color: var(--tx2); }
td,th { color: var(--tx2) !important; }
code {
  background: var(--surface-hi) !important; color: var(--accent) !important;
  border-radius: 4px; padding: 2px 6px;
  font-family: 'IBM Plex Mono', monospace !important;
}
hr { border-color: var(--border) !important; }

.stCaption, [data-testid="stCaptionContainer"] p {
  color: var(--tx3) !important; font-size:.71rem !important;
}

/* Dataframe / Chart */
.stDataFrame, .stDataFrame [data-testid="stDataFrameResizable"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r10) !important;
}
[data-testid="stVegaLiteChart"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r10) !important;
  padding:8px;
}

/* Metric cards */
div[data-testid="metric-container"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r10);
  padding: 14px 18px;
  transition: border-color .2s;
}
div[data-testid="metric-container"]:hover { border-color: var(--border-accent); }
div[data-testid="metric-container"] label {
  color: var(--tx3) !important; font-size:.67rem;
  text-transform:uppercase; letter-spacing:.09em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--tx) !important; font-size:1.45rem; font-weight:700;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-size:.75rem;
}

/* File uploader */
.stFileUploader {
  background: var(--surface) !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--r12) !important;
}
.stFileUploader:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploadDropzone"] { background: transparent !important; }
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small { color: var(--tx2) !important; }

/* Toggle */
.stToggle > label { color: var(--tx2) !important; }
div[data-testid="stRadioButton"] label { color: var(--tx2) !important; }
div[data-testid="stRadioButton"] p { color: var(--tx2) !important; }

/* ═══ QUEUEIQ NAV BRAND ═══ */
.qiq-brand {
  display:flex; align-items:center; gap:10px;
  padding:18px 16px 14px; border-bottom:1px solid var(--border);
  margin-bottom:8px;
}
.qiq-logo {
  width:30px; height:30px; border-radius:7px;
  background:linear-gradient(135deg,#0080FF,#00C2FF);
  display:flex; align-items:center; justify-content:center;
  font-size:14px; font-weight:800; color:#fff;
  letter-spacing:-1px; flex-shrink:0;
}
.qiq-name {
  font-size:.97rem; font-weight:700; color:var(--tx) !important;
  letter-spacing:-.2px; line-height:1;
}
.qiq-tagline {
  font-size:.6rem; color:var(--tx3) !important;
  text-transform:uppercase; letter-spacing:.1em;
  margin-top:2px; line-height:1;
}
.sb-section {
  font-size:.58rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.12em; color:var(--tx4) !important;
  padding:12px 16px 6px; display:block;
}
.sb-lbl {
  font-size:.61rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.12em; color:var(--tx3);
  margin:4px 0 8px; padding-bottom:6px;
  border-bottom:1px solid var(--border);
}

/* ═══ TOP BAR ═══ */
.qiq-topbar {
  display:flex; align-items:center; justify-content:space-between;
  padding:12px 0 16px; margin-bottom:4px;
  border-bottom:1px solid var(--border);
}
.qiq-topbar-left { display:flex; align-items:center; gap:16px; }
.qiq-page-title {
  font-size:1.15rem; font-weight:700; color:var(--tx);
  letter-spacing:-.3px;
}
.qiq-page-sub {
  font-size:.75rem; color:var(--tx3); margin-top:1px;
}
.qiq-topbar-right {
  display:flex; align-items:center; gap:10px;
}
.status-pill {
  display:inline-flex; align-items:center; gap:6px;
  padding:4px 12px; border-radius:20px;
  font-size:.7rem; font-weight:700; letter-spacing:.04em;
  white-space:nowrap;
}
.status-pill.ok {
  background:var(--success-dim); border:1px solid var(--success-bd);
  color:var(--success);
}
.status-pill.warn {
  background:var(--warning-dim); border:1px solid var(--warning-bd);
  color:var(--warning);
}
.status-pill.crit {
  background:var(--danger-dim); border:1px solid var(--danger-bd);
  color:var(--danger);
}
.status-pill.sim {
  background:rgba(0,128,255,.10); border:1px solid rgba(0,128,255,.25);
  color:#60A5FA;
}
.pulse-dot {
  width:7px; height:7px; border-radius:50%;
  background:currentColor;
  animation:pulse-dot 2.2s ease-in-out infinite;
  flex-shrink:0;
}
.mode-chip {
  display:inline-flex; align-items:center; gap:5px;
  background:var(--surface); border:1px solid var(--border);
  color:var(--tx2); padding:4px 11px; border-radius:20px;
  font-size:.68rem; font-weight:500;
}
.ts-chip {
  font-size:.67rem; color:var(--tx3);
  font-family:'IBM Plex Mono',monospace;
}

/* ═══ FACILITY STATUS HERO ═══ */
.fac-hero {
  border-radius:var(--r12); padding:22px 26px;
  border:1px solid var(--border);
  background:var(--surface);
  margin-bottom:16px;
  position:relative; overflow:hidden;
}
.fac-hero::before {
  content:''; position:absolute;
  top:0; left:0; right:0; height:3px;
}
.fac-hero.ok::before    { background:var(--success); }
.fac-hero.warn::before  { background:var(--warning); }
.fac-hero.crit::before  { background:var(--danger); }
.fac-hero.sim::before   { background:var(--accent); }

.fac-hero-row { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px; }
.fac-status-block { flex:1; min-width:200px; }
.fac-label {
  font-size:.6rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.14em; color:var(--tx3); margin-bottom:8px;
}
.fac-level {
  font-size:1.6rem; font-weight:800; letter-spacing:-.5px;
  line-height:1; margin-bottom:4px;
}
.fac-summary { font-size:.82rem; color:var(--tx2); line-height:1.5; }
.fac-metrics { display:flex; gap:24px; flex-wrap:wrap; }
.fac-metric { text-align:center; }
.fac-metric-val {
  font-size:1.8rem; font-weight:800; line-height:1;
  font-variant-numeric: tabular-nums;
}
.fac-metric-lbl {
  font-size:.62rem; color:var(--tx3); text-transform:uppercase;
  letter-spacing:.08em; margin-top:3px;
}
.fac-divider {
  width:1px; background:var(--border); align-self:stretch;
  margin:0 4px;
}

/* ═══ FOUR-QUESTION CARDS ═══ */
.q-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--r10); padding:16px 18px; height:100%;
  border-top:3px solid;
}
.q-num {
  font-size:.58rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.1em; margin-bottom:7px;
}
.q-val {
  font-size:1.05rem; font-weight:700; color:var(--tx); margin-bottom:4px;
  line-height:1.3;
}
.q-sub { font-size:.79rem; color:var(--tx2); line-height:1.5; }

/* ═══ SECTION HEADER ═══ */
.sec-hdr {
  font-size:.6rem; font-weight:700; color:var(--tx3);
  text-transform:uppercase; letter-spacing:.14em;
  margin:20px 0 10px; display:flex; align-items:center; gap:10px;
}
.sec-hdr::after { content:''; flex:1; height:1px; background:var(--border); }

/* ═══ COUNTER CARDS (OPERATIONAL) ═══ */
.counter-card {
  background:var(--surface); border:1px solid var(--border);
  border-top:3px solid; border-radius:var(--r10);
  padding:16px 18px 14px; transition:border-color .2s,transform .15s;
  position:relative; overflow:hidden;
}
.counter-card:hover {
  transform:translateY(-2px);
  border-color:var(--border-accent);
}
.cc-0 { border-top-color:var(--c0); }
.cc-1 { border-top-color:var(--c1); }
.cc-2 { border-top-color:var(--c2); }
.cc-3 { border-top-color:var(--c3); }
.cc-4 { border-top-color:var(--c4); }
.cc-5 { border-top-color:var(--c5); }

.c-id   { font-size:.58rem; font-weight:700; text-transform:uppercase; letter-spacing:.12em; color:var(--tx3); margin-bottom:2px; }
.c-name { font-size:.83rem; font-weight:600; color:var(--tx); margin-bottom:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.c-cnt  { font-size:2.6rem; font-weight:800; color:var(--tx); line-height:1; font-variant-numeric:tabular-nums; }
.c-unit { font-size:.6rem; color:var(--tx3); text-transform:uppercase; letter-spacing:.1em; margin:2px 0 10px; }
.c-row  { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
.c-wait { font-size:.74rem; color:var(--tx2); font-weight:500; }
.c-pred { font-size:.7rem; color:var(--tx3); }
.c-pred strong { color:var(--accent); font-weight:600; }
.c-trend { font-size:.7rem; font-weight:600; }
.c-trend.up   { color:var(--danger); }
.c-trend.down { color:var(--success); }
.c-trend.flat { color:var(--tx3); }

/* Utilization bar */
.util-bar-wrap {
  background:var(--surface-hi); border-radius:3px;
  height:4px; margin-top:10px; overflow:hidden;
}
.util-bar-fill {
  height:100%; border-radius:3px;
  transition:width .4s ease;
}
.util-pct { font-size:.62rem; color:var(--tx3); margin-top:4px; text-align:right; }

/* Status badges */
.lbadge {
  display:inline-flex; align-items:center; gap:4px;
  padding:2px 9px; border-radius:20px; font-size:.65rem;
  font-weight:700; letter-spacing:.03em; border:1px solid;
}
.lbadge.low      { background:var(--success-dim); border-color:var(--success-bd); color:var(--success); }
.lbadge.medium   { background:var(--warning-dim); border-color:var(--warning-bd); color:var(--warning); }
.lbadge.high     { background:var(--danger-dim);  border-color:var(--danger-bd);  color:var(--danger); }
.lbadge.critical { background:var(--critical-dim);border-color:var(--critical-bd);color:var(--critical); }

/* ═══ AI ACTION CENTER ═══ */
.ai-action-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--r12); padding:20px 22px;
  border-left:4px solid;
}
.ai-action-card.no_action    { border-left-color:var(--success); }
.ai-action-card.redirect     { border-left-color:var(--warning); }
.ai-action-card.open_counter { border-left-color:var(--danger); }
.ai-action-card.monitor      { border-left-color:var(--warning); }

.aac-header {
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom:12px;
}
.aac-label {
  font-size:.6rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.12em; color:var(--tx3);
}
.aac-severity {
  display:inline-flex; align-items:center; gap:5px;
  padding:2px 10px; border-radius:20px; font-size:.66rem; font-weight:700;
}
.aac-title { font-size:1.05rem; font-weight:700; color:var(--tx); margin-bottom:6px; }
.aac-msg   { font-size:.85rem; color:var(--tx2); line-height:1.6; margin-bottom:12px; }
.aac-why   {
  font-size:.78rem; color:var(--tx3); padding:10px 14px;
  background:var(--surface-hi); border-radius:var(--r8);
  margin-bottom:10px; border-left:2px solid var(--border-accent);
}
.aac-impact {
  display:flex; gap:16px; flex-wrap:wrap; margin-top:10px;
}
.aac-impact-item { text-align:center; }
.aac-impact-val {
  font-size:1.2rem; font-weight:800; color:var(--success);
  font-variant-numeric:tabular-nums;
}
.aac-impact-lbl { font-size:.62rem; color:var(--tx3); text-transform:uppercase; letter-spacing:.07em; margin-top:1px; }

/* ═══ AI ALERT PANEL ═══ */
.ai-box {
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--r10); padding:18px 20px 14px;
  position:relative;
}
.ai-box.high   { border-color:var(--danger-bd);  background:var(--danger-dim); }
.ai-box.medium { border-color:var(--warning-bd); background:var(--warning-dim); }
.ai-box.low    { border-color:var(--success-bd); background:var(--success-dim); }
.ai-tag {
  position:absolute; top:-10px; left:16px;
  background:var(--bg-secondary); border:1px solid var(--border-accent);
  color:var(--accent); padding:2px 10px; border-radius:20px;
  font-size:.58rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
}
.ai-text { font-size:.9rem; color:var(--tx); line-height:1.7; margin:4px 0 0; }
.ai-src  { font-size:.65rem; color:var(--tx3); margin-top:10px; text-align:right; }

/* ═══ UPLOAD PLACEHOLDER ═══ */
.up-placeholder {
  background:var(--surface); border:2px dashed var(--border);
  border-radius:var(--r16); padding:52px 32px; text-align:center;
  transition:border-color .2s;
}
.up-placeholder:hover { border-color:var(--accent); }
.up-icon  { font-size:2.4rem; margin-bottom:12px; }
.up-title { font-size:1rem; font-weight:600; color:var(--tx); margin-bottom:6px; }
.up-sub   { font-size:.83rem; color:var(--tx2); }

/* ═══ SCENARIO LAB ═══ */
.scenario-compare {
  display:grid; grid-template-columns:1fr auto 1fr;
  gap:12px; align-items:center; margin:12px 0;
}
.scenario-state {
  background:var(--surface-hi); border:1px solid var(--border);
  border-radius:var(--r10); padding:16px 18px;
}
.scenario-state-lbl {
  font-size:.6rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.1em; color:var(--tx3); margin-bottom:8px;
}
.scenario-state-val {
  font-size:2rem; font-weight:800; color:var(--tx);
  font-variant-numeric:tabular-nums;
}
.scenario-state-sub { font-size:.75rem; color:var(--tx2); margin-top:4px; }
.scenario-arrow { font-size:1.5rem; color:var(--tx3); text-align:center; }
.impact-delta {
  display:flex; gap:12px; flex-wrap:wrap;
  padding:12px 16px; background:var(--success-dim);
  border:1px solid var(--success-bd); border-radius:var(--r8);
  margin-top:10px;
}
.impact-delta.negative {
  background:var(--danger-dim); border-color:var(--danger-bd);
}
.impact-tag { font-size:.83rem; font-weight:700; color:var(--success); }

/* ═══ QUEUE FLOW DIAGRAM ═══ */
.qflow-wrap {
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--r12); padding:20px 24px;
}
.qflow-node {
  display:inline-flex; flex-direction:column; align-items:center;
  gap:4px;
}
.qflow-counter {
  width:54px; height:54px; border-radius:10px; border:2px solid;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  font-size:11px; font-weight:700; transition:.2s;
}
.qflow-counter.ok   { border-color:var(--success); color:var(--success); background:var(--success-dim); }
.qflow-counter.warn { border-color:var(--warning); color:var(--warning); background:var(--warning-dim); }
.qflow-counter.crit { border-color:var(--danger);  color:var(--danger);  background:var(--danger-dim); }
.qflow-counter.off  { border-color:var(--tx4); color:var(--tx4); background:var(--surface-hi); }

/* ═══ LOGIN OVERRIDE ═══ */
.lw-left-col {
  min-height:100vh; background:linear-gradient(160deg,#030810 0%,#070D1A 30%,#0C1526 60%,#091428 100%);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  padding:48px 32px; position:relative; overflow:hidden;
}
.lw-left-col::before {
  content:''; position:absolute; width:400px; height:400px; border-radius:50%;
  border:1px solid rgba(0,194,255,.06); top:-100px; right:-100px;
}
.lw-left-col::after {
  content:''; position:absolute; width:220px; height:220px; border-radius:50%;
  border:1px solid rgba(0,194,255,.05); bottom:-60px; left:-60px;
}
.lw-logo  { font-size:3rem; text-align:center; margin-bottom:8px; position:relative; z-index:1; }
.lw-brand {
  font-size:2.1rem; font-weight:800; color:#E2EAF8;
  text-align:center; margin-bottom:8px; position:relative; z-index:1;
  letter-spacing:-.3px;
}
.lw-desc {
  font-size:.86rem; color:rgba(136,153,187,.85); text-align:center;
  line-height:1.65; max-width:250px; margin-bottom:32px; position:relative; z-index:1;
}
.lw-stats { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; position:relative; z-index:1; }
.lw-stat  {
  text-align:center; background:rgba(0,194,255,.06);
  border:1px solid rgba(0,194,255,.15); border-radius:10px; padding:12px 18px;
}
.lw-sn { font-size:1.4rem; font-weight:800; color:#E2EAF8; }
.lw-sl { font-size:.6rem; color:rgba(136,153,187,.75); text-transform:uppercase; letter-spacing:.08em; margin-top:2px; }
.lw-badge {
  display:inline-flex; align-items:center; gap:6px;
  background:rgba(0,194,255,.08); border:1px solid rgba(0,194,255,.22);
  color:var(--accent); padding:5px 13px; border-radius:20px;
  font-size:.7rem; font-weight:700; letter-spacing:.05em; margin-bottom:22px;
}
.lw-title { font-size:1.85rem; font-weight:800; color:#E2EAF8; margin:0 0 5px; letter-spacing:-.4px; }
.lw-sub   { font-size:.85rem; color:#8899BB; margin-bottom:18px; line-height:1.55; }
.lw-divider { height:1px; background:rgba(255,255,255,.07); margin:14px 0 18px; }
.lw-flabel  { font-size:.67rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--tx3); margin-bottom:6px; display:block; }
.lw-hint { font-size:.69rem; color:var(--tx3); margin-top:12px; text-align:center; }
.lw-hint code { background:var(--surface-hi); color:var(--accent); padding:2px 7px; border-radius:4px; }
.lw-footer { font-size:.68rem; color:var(--tx3); text-align:center; margin-top:28px; padding-top:16px; border-top:1px solid var(--border); }

/* Login widget overrides */
.stTextInput > div > div {
  border: 1.5px solid var(--border) !important;
  border-radius: 10px !important;
  background: var(--surface) !important;
}
.stTextInput input {
  color: var(--tx) !important;
  background: var(--surface) !important;
  caret-color: var(--accent) !important;
  font-size:.94rem !important;
}
.stTextInput input::placeholder { color: var(--tx3) !important; opacity:1 !important; }
.stTextInput > div > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
}

/* ═══ ANIMATIONS ═══ */
@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); }
  50%     { opacity:.45; transform:scale(.82); }
}
@keyframes fadeInUp {
  from { opacity:0; transform:translateY(8px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes scanline {
  0% { transform:translateY(-100%); }
  100% { transform:translateY(100vh); }
}
.counter-card { animation:fadeInUp .24s ease both; }
.fac-hero     { animation:fadeInUp .20s ease both; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# THEME  &mdash;  light (DAY) is default  /  dark = night override
# ─────────────────────────────────────────────────────────

def inject_theme(theme: str) -> None:
    """
    LIGHT/DAY  = clean professional white theme (like a fintech report).
    DARK/NIGHT = command center dark navy (default base CSS).
    Both modes are fully distinct and readable.
    """
    # ── Always restore sidebar visibility (login page hides it) ─────────────
    st.markdown(
        "<style>"
        "[data-testid='stSidebar']{display:flex !important;visibility:visible !important;}"
        "[data-testid='stSidebarContent']{display:flex !important;}"
        "</style>",
        unsafe_allow_html=True,
    )

    if theme == "light":
        # ── DAY / LIGHT MODE ─────────────────────────────────────────────────
        # Override the dark base CSS with clean professional whites
        st.markdown("""
        <style>
        :root {
          --bg-primary:   #F4F6FA;
          --bg-secondary: #EBEEF5;
          --surface:      #FFFFFF;
          --surface-el:   #F0F2F8;
          --surface-hi:   #E8EBF4;
          --border:       rgba(0,0,0,.09);
          --border-accent:rgba(11,114,133,.30);
          --accent:       #0B7285;
          --accent2:      #0369A1;
          --accent-dim:   rgba(11,114,133,.08);
          --accent-glow:  0 0 16px rgba(11,114,133,.15);
          --success:      #059669;
          --success-dim:  rgba(5,150,105,.08);
          --success-bd:   rgba(5,150,105,.28);
          --warning:      #D97706;
          --warning-dim:  rgba(217,119,6,.08);
          --warning-bd:   rgba(217,119,6,.28);
          --danger:       #DC2626;
          --danger-dim:   rgba(220,38,38,.07);
          --danger-bd:    rgba(220,38,38,.28);
          --critical:     #991B1B;
          --critical-dim: rgba(153,27,27,.07);
          --critical-bd:  rgba(153,27,27,.28);
          --tx:   #0B1320;
          --tx2:  #3D4A5C;
          --tx3:  #7A8A9E;
          --tx4:  #B0BBC8;
          --c0:#0B7285; --c1:#059669; --c2:#D97706;
          --c3:#DC2626; --c4:#7C3AED; --c5:#0369A1;
        }
        html,body,[class*="css"],.stApp {
          background: #F4F6FA !important;
          color: #0B1320 !important;
        }
        [data-testid="stAppViewContainer"] { background:#F4F6FA !important; }
        [data-testid="stMain"]             { background:#F4F6FA !important; }
        .block-container                   { background:#F4F6FA !important; }

        /* Sidebar &mdash; keep dark for contrast in day mode too */
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg,#0B1320 0%,#131F30 100%) !important;
          border-right: 1px solid rgba(255,255,255,.07) !important;
        }
        [data-testid="stSidebar"] * { color:#E2E8F0 !important; }
        [data-testid="stSidebar"] .stMarkdown p { color:#94A3B8 !important; }

        /* Scrollbar light */
        ::-webkit-scrollbar-track { background:#EBEEF5; }
        ::-webkit-scrollbar-thumb { background:#D1D5E0; }
        ::-webkit-scrollbar-thumb:hover { background:#0B7285; }

        /* Widget wrappers */
        .stSelectbox>div>div,.stTextInput>div>div,.stTextArea>div>div {
          background:#FFFFFF !important; border:1px solid #D8DEF0 !important;
          color:#0B1320 !important; border-radius:8px !important;
        }
        .stTextInput input,.stTextArea textarea,.stNumberInput input {
          color:#0B1320 !important; background:#FFFFFF !important;
          caret-color:#0B7285 !important;
        }
        .stTextInput input::placeholder,.stTextArea textarea::placeholder {
          color:#9AA5B4 !important; opacity:1 !important;
        }
        .stSelectbox>div>div:focus-within,.stTextInput>div>div:focus-within {
          border-color:#0B7285 !important;
          box-shadow:0 0 0 3px rgba(11,114,133,.10) !important;
        }
        .stTextInput label,.stTextArea label,.stSelectbox label,
        .stSlider label,.stNumberInput label,.stCheckbox label,
        .stRadio label,.stMultiSelect label,.stDateInput label {
          color:#3D4A5C !important; font-size:.78rem !important;
        }
        .stCheckbox span,.stRadio span { color:#3D4A5C !important; }
        div[data-testid="stRadioButton"] p { color:#0B1320 !important; }
        div[data-testid="stRadioButton"] label { color:#0B1320 !important; }
        .stToggle>label { color:#0B1320 !important; }
        .stSlider>div>div>div>div { background:#0B7285 !important; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
          background:#EBEEF5 !important;
          border:1px solid #D8DEF0 !important;
        }
        .stTabs [data-baseweb="tab"] {
          color:#3D4A5C !important;
        }
        .stTabs [aria-selected="true"] {
          background:#FFFFFF !important;
          color:#0B7285 !important;
        }

        /* Buttons */
        .stButton>button {
          background:#FFFFFF !important; color:#0B1320 !important;
          border:1px solid #D8DEF0 !important;
        }
        .stButton>button:hover {
          border-color:#0B7285 !important;
          background:rgba(11,114,133,.07) !important;
          color:#0B7285 !important;
        }
        .stDownloadButton>button {
          background:linear-gradient(135deg,rgba(11,114,133,.12),rgba(14,116,144,.12)) !important;
          color:#0B7285 !important;
          border:1px solid rgba(11,114,133,.30) !important;
        }

        /* File uploader */
        .stFileUploader {
          background:#FFFFFF !important;
          border:2px dashed #D8DEF0 !important;
        }
        [data-testid="stFileUploadDropzone"] { background:#FFFFFF !important; }
        [data-testid="stFileUploadDropzone"] p,
        [data-testid="stFileUploadDropzone"] small { color:#3D4A5C !important; }

        /* Expander */
        .stExpander {
          background:#FFFFFF !important;
          border:1px solid #D8DEF0 !important;
        }
        .stExpander summary { color:#0B1320 !important; }
        .stExpander [data-testid="stExpanderDetails"] p { color:#3D4A5C !important; }

        /* Alert boxes */
        .stInfo    { background:rgba(11,114,133,.06) !important; border-color:rgba(11,114,133,.22) !important; }
        .stInfo p  { color:#0B1320 !important; }
        .stWarning { background:rgba(217,119,6,.07) !important; border-color:rgba(217,119,6,.22) !important; }
        .stWarning p { color:#7C4A00 !important; }
        .stSuccess { background:rgba(5,150,105,.07) !important; border-color:rgba(5,150,105,.22) !important; }
        .stSuccess p { color:#064E3B !important; }
        .stError   { background:rgba(220,38,38,.06) !important; border-color:rgba(220,38,38,.22) !important; }
        .stError p { color:#7F1D1D !important; }

        /* Text */
        [data-testid="stMarkdown"] p,[data-testid="stMarkdown"] li { color:#3D4A5C !important; }
        [data-testid="stMarkdown"] h1,[data-testid="stMarkdown"] h2,
        [data-testid="stMarkdown"] h3,[data-testid="stMarkdown"] h4 { color:#0B1320 !important; }
        h1,h2,h3,h4 { color:#0B1320 !important; }
        p { color:#3D4A5C; } li { color:#3D4A5C; } td,th { color:#3D4A5C !important; }
        code { background:#EDF0F7 !important; color:#0B7285 !important; }
        hr { border-color:#D8DEF0 !important; }
        .stCaption,[data-testid="stCaptionContainer"] p { color:#7A8A9E !important; }

        /* Dataframe / Chart */
        .stDataFrame,.stDataFrame [data-testid="stDataFrameResizable"] {
          background:#FFFFFF !important; border:1px solid #D8DEF0 !important;
        }
        [data-testid="stVegaLiteChart"] {
          background:#FFFFFF !important; border:1px solid #D8DEF0 !important;
        }

        /* Metrics */
        div[data-testid="metric-container"] {
          background:#FFFFFF !important; border:1px solid #D8DEF0 !important;
        }
        div[data-testid="metric-container"]:hover { border-color:#0B7285 !important; }
        div[data-testid="metric-container"] label { color:#7A8A9E !important; }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#0B1320 !important; }

        /* Component classes */
        .qiq-topbar { border-bottom-color:#D8DEF0 !important; }
        .qiq-page-title { color:#0B1320 !important; }
        .qiq-page-sub   { color:#7A8A9E !important; }
        .mode-chip { background:#FFFFFF !important; border-color:#D8DEF0 !important; color:#3D4A5C !important; }
        .ts-chip   { color:#9AA5B4 !important; }
        .sec-hdr   { color:#9AA5B4 !important; }
        .sec-hdr::after { background:#D8DEF0 !important; }

        /* Facility hero */
        .fac-hero   { background:#FFFFFF !important; border-color:#D8DEF0 !important; }
        .fac-label  { color:#9AA5B4 !important; }
        .fac-summary{ color:#3D4A5C !important; }
        .fac-metric-lbl { color:#9AA5B4 !important; }
        .fac-divider { background:#D8DEF0 !important; }

        /* Counter cards */
        .counter-card { background:#FFFFFF !important; border-color:#D8DEF0 !important; }
        .counter-card:hover { border-color:#0B7285 !important; }
        .c-id,.c-unit,.util-pct { color:#9AA5B4 !important; }
        .c-name  { color:#0B1320 !important; }
        .c-cnt   { color:#0B1320 !important; }
        .c-wait  { color:#3D4A5C !important; }
        .c-pred  { color:#7A8A9E !important; }
        .c-pred strong { color:#0B7285 !important; }
        .util-bar-wrap { background:#E8EBF4 !important; }

        /* AI Action Card */
        .ai-action-card { background:#FFFFFF !important; border-color:#D8DEF0 !important; }
        .aac-label  { color:#9AA5B4 !important; }
        .aac-title  { color:#0B1320 !important; }
        .aac-msg    { color:#3D4A5C !important; }
        .aac-why    { background:#F0F2F8 !important; border-left-color:rgba(11,114,133,.30) !important; color:#3D4A5C !important; }

        /* AI Alert box */
        .ai-box  { background:#FFFFFF !important; border-color:#D8DEF0 !important; }
        .ai-text { color:#0B1320 !important; }
        .ai-src  { color:#9AA5B4 !important; }
        .ai-tag  {
          background:#FFFFFF !important;
          border-color:rgba(11,114,133,.30) !important;
          color:#0B7285 !important;
        }

        /* Q-cards */
        .q-card { background:#FFFFFF !important; border-color:#D8DEF0 !important; }
        .q-val  { color:#0B1320 !important; }
        .q-sub  { color:#3D4A5C !important; }

        /* Placeholders */
        .up-placeholder { background:#FFFFFF !important; border-color:#D8DEF0 !important; }
        .up-title { color:#0B1320 !important; }
        .up-sub   { color:#7A8A9E !important; }
        .sb-lbl   { color:#9AA5B4 !important; border-bottom-color:rgba(255,255,255,.10) !important; }
        </style>
        """, unsafe_allow_html=True)
    else:
        # ── NIGHT / DARK MODE ─────────────────────────────────────────────────
        # Reinforce the command center dark theme (base CSS already dark,
        # but re-apply key tokens in case light mode was previously injected)
        st.markdown("""
        <style>
        :root {
          --bg-primary:   #070D1A;
          --bg-secondary: #0C1526;
          --surface:      #101E33;
          --surface-el:   #152338;
          --surface-hi:   #1A2B42;
          --border:       rgba(255,255,255,.07);
          --border-accent:rgba(0,194,255,.20);
          --accent:       #00C2FF;
          --accent2:      #0080FF;
          --accent-dim:   rgba(0,194,255,.12);
          --success:      #10B981;
          --success-dim:  rgba(16,185,129,.10);
          --success-bd:   rgba(16,185,129,.30);
          --warning:      #F59E0B;
          --warning-dim:  rgba(245,158,11,.10);
          --warning-bd:   rgba(245,158,11,.30);
          --danger:       #EF4444;
          --danger-dim:   rgba(239,68,68,.08);
          --danger-bd:    rgba(239,68,68,.28);
          --tx:   #E2EAF8;
          --tx2:  #8899BB;
          --tx3:  #4A5A78;
          --tx4:  #2A3A58;
        }
        html,body,[class*="css"],.stApp {
          background: #070D1A !important; color: #E2EAF8 !important;
        }
        [data-testid="stAppViewContainer"] { background:#070D1A !important; }
        [data-testid="stMain"]             { background:#070D1A !important; }
        .block-container                   { background:#070D1A !important; }
        </style>
        """, unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────

def classify_load(wait_min) -> str:
    """Map wait time (minutes) &#8594; 'low' | 'medium' | 'high'."""
    if wait_min is None or wait_min < 12:
        return "low"
    return "high" if wait_min >= 25 else "medium"


@st.cache_data(show_spinner=False)
def generate_sample_csv() -> bytes:
    """
    3 counters, 4-hour window, 5-min intervals (49 readings per counter).

    Design goals:
      Counter 1 (Main Desk)   &mdash; ramps 5 &#8594; ~35 people &mdash; last reading HIGH wait
      Counter 2 (Express)     &mdash; steady 7&ndash;13           &mdash; LOW wait (redirect target)
      Counter 3 (Inquiry)     &mdash; starts 30, drops to 5 &mdash; tapering, LOW by end

    With default service_rate=1.0 cust/min:
      Counter 1 wait ≈ 35 min > 25 &#8594; REDIRECT fires &#8594; Groq alert references it
    """
    base = datetime.now().replace(second=0, microsecond=0) - timedelta(hours=4)
    rng  = np.random.default_rng(42)
    rows = []

    for i in range(49):                      # 0..48  &#8594; 0..240 min in 5-min steps
        t    = base + timedelta(minutes=5 * i)
        frac = i / 48.0

        # Main Desk: sigmoid-like ramp to peak ≈ 35 around 75% of window
        ramp = min(1.0, frac * 1.5)
        c1   = int(max(3, 4 + 33 * ramp * (1 - 0.12 * ramp) + rng.integers(-2, 3)))

        # Express: low-medium, mild sinusoidal variation
        c2   = int(max(1, 7 + 6 * np.sin(np.pi * frac) + rng.integers(-2, 3)))

        # Inquiry: linear drop from 30 to 5
        c3   = int(max(1, 30 - 25 * frac + rng.integers(-3, 4)))

        rows += [
            {"timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
             "counter_name": "Counter 1 (Main Desk)",    "people_count": c1},
            {"timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
             "counter_name": "Counter 2 (Express)",       "people_count": c2},
            {"timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
             "counter_name": "Counter 3 (Inquiry)",       "people_count": c3},
        ]

    buf = io.BytesIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()


def parse_csv_raw(df: pd.DataFrame) -> list[dict]:
    """
    Parse uploaded CSV into counter history dicts &mdash; WITHOUT baking in service_rate.
    service_rate is injected separately so slider changes don't force a full re-parse.

    Required columns: timestamp, counter_name, people_count
    Returns: list of {id, name, history, arrival_rate=None}
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    required = {"timestamp", "counter_name", "people_count"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {sorted(missing)}.\n"
            f"Required: timestamp, counter_name, people_count"
        )

    df["timestamp"]    = pd.to_datetime(df["timestamp"])
    df["people_count"] = (
        pd.to_numeric(df["people_count"], errors="coerce")
        .fillna(0).clip(lower=0).astype(int)
    )
    df = df.sort_values("timestamp")

    configs = []
    for idx, (name, grp) in enumerate(
        sorted(df.groupby("counter_name"), key=lambda x: x[0])
    ):
        grp     = grp.sort_values("timestamp")
        history = [
            {
                "counter_id":      idx + 1,
                "name":            str(name),
                "people_count":    int(row["people_count"]),
                "timestamp_epoch": row["timestamp"].timestamp(),
                "timestamp_str":   row["timestamp"].strftime("%H:%M"),
            }
            for _, row in grp.iterrows()
        ]
        configs.append({
            "id":           idx + 1,
            "name":         str(name),
            "history":      history,
            "arrival_rate": None,   # unknown from real data
        })
    return configs


def apply_service_rate(configs_raw: list[dict], service_rate: float) -> list[dict]:
    """Attach a service_rate to every counter config (shallow-merges)."""
    return [{**c, "service_rate": service_rate} for c in configs_raw]


# ─────────────────────────────────────────────────────────
# RENDER FUNCTIONS  (shared by BOTH modes)
# ─────────────────────────────────────────────────────────

def _lbadge(load: str) -> str:
    """Return a coloured load-level HTML badge."""
    dot   = "●"
    label = {"low": "Low", "medium": "Medium", "high": "High"}[load]
    return f'<span class="lbadge {load}">{dot} {label}</span>'


def render_header(mode: str, info: str) -> None:
    from datetime import datetime as _dt_hdr
    icon_map  = {"simulation": "⟳", "real_data": "↑", "cv": "◉"}
    label_map = {"simulation": "Simulation", "real_data": "CSV Data", "cv": "CCTV"}
    status_map = {
        "simulation": ("sim",  "SIMULATION RUNNING"),
        "real_data":  ("ok",   "DATA LOADED"),
        "cv":         ("ok",   "CAMERA ANALYSIS"),
    }
    icon   = icon_map.get(mode, "◈")
    label  = label_map.get(mode, mode)
    scls, stxt = status_map.get(mode, ("ok", "OPERATIONAL"))
    ts = _dt_hdr.now().strftime("%H:%M:%S")
    st.markdown(f'''
    <div class="qiq-topbar">
      <div class="qiq-topbar-left">
        <div>
          <div class="qiq-page-title">Queue Intelligence Platform</div>
          <div class="qiq-page-sub">{info}</div>
        </div>
      </div>
      <div class="qiq-topbar-right">
        <span class="mode-chip">{icon} {label}</span>
        <span class="status-pill {scls}">
          <span class="pulse-dot"></span>{stxt}
        </span>
        <span class="ts-chip">{ts}</span>
      </div>
    </div>
    ''', unsafe_allow_html=True)


def render_metrics(readings, wait_times, preds, rec) -> None:
    if not readings:
        return
    total  = sum(r["people_count"] for r in readings)
    valid  = [w for w in wait_times if w is not None]
    avg_wt = sum(valid) / len(valid) if valid else 0
    max_wt = max(valid) if valid else 0
    alerts = sum(1 for p in preds if p.get("alert"))
    sev_ic = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get("severity", "low"), "⚪")
    action = rec.get("action", "&mdash;").replace("_", " ").title()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total People",   total)
    c2.metric("⏱ Avg Wait",        f"{avg_wt:.0f} min")
    c3.metric("🔴 Max Wait",       f"{max_wt:.0f} min")
    c4.metric("&#9888;️ Active Alerts",  f"{alerts}/{len(preds)}")
    c5.metric("🎯 Recommendation", f"{sev_ic} {action}")


def render_counter_table(readings, wait_times, preds, horizon: int) -> None:
    load_label = {"low": "🟢 Low", "medium": "🟡 Medium", "high": "🔴 High"}
    rows = []
    for i, r in enumerate(readings):
        wt   = wait_times[i] if i < len(wait_times) else None
        pred = preds[i] if i < len(preds) else {}
        load = classify_load(wt)
        pc   = pred.get("predicted_count")
        rows.append({
            "Counter":                r["name"],
            "People Now":             r["people_count"],
            "Est. Wait":              f"{wt:.0f} min" if wt is not None else "&mdash;",
            f"Predicted +{horizon}m": f"{pc} ppl" if pc is not None else "...",
            "Load Level":             load_label[load],
        })
    st.dataframe(
        pd.DataFrame(rows),
        width='stretch',
        hide_index=True,
        column_config={
            "People Now": st.column_config.NumberColumn(format="%d 👤"),
        },
    )


def render_counter_cards(readings, wait_times, preds) -> None:
    cols = st.columns(max(1, min(len(readings), 4)))
    for i, r in enumerate(readings):
        col = cols[i % len(cols)]
        wt    = wait_times[i] if i < len(wait_times) else None
        pred  = preds[i]      if i < len(preds)       else {}
        load  = classify_load(wt)
        pc    = pred.get("predicted_count")
        cnt   = r["people_count"]

        # Utilization estimate (cap at 100%)
        util = min(int((cnt / 30) * 100), 100) if cnt else 0
        util_color = {
            "low":      "var(--success)",
            "medium":   "var(--warning)",
            "high":     "var(--danger)",
        }.get(load, "var(--tx3)")

        # Trend direction from prediction
        if pc is not None and cnt is not None:
            delta = pc - cnt
            if delta > 1:
                trend_cls, trend_sym = "up", "↗ +{} forecast".format(int(delta))
            elif delta < -1:
                trend_cls, trend_sym = "down", "↘ {} forecast".format(int(delta))
            else:
                trend_cls, trend_sym = "flat", "&#8594; stable"
        else:
            trend_cls, trend_sym = "flat", "collecting data"

        wait_str = f"{wt:.0f} min" if wt is not None else "&mdash;"
        pred_str = f"{pc}" if pc is not None else "&mdash;"

        with col:
            st.markdown(f'''
            <div class="counter-card cc-{i % 6}">
              <div class="c-id">Counter {r['counter_id']}</div>
              <div class="c-name">{r['name']}</div>
              <div class="c-cnt">{cnt}</div>
              <div class="c-unit">people waiting</div>
              <div class="c-row">
                {_lbadge(load)}
                <span class="c-wait">⏱ {wait_str}</span>
              </div>
              <div class="c-row" style="margin-bottom:0">
                <span class="c-pred">&#8594; <strong>{pred_str} ppl</strong> in +{pred.get("predicted_in_min", "?"):.0f}m</span>
                <span class="c-trend {trend_cls}">{trend_sym}</span>
              </div>
              <div class="util-bar-wrap">
                <div class="util-bar-fill" style="width:{util}%;background:{util_color}"></div>
              </div>
              <div class="util-pct">{util}% utilization</div>
            </div>
            ''', unsafe_allow_html=True)


def render_trend_chart(counter_configs: list[dict]) -> None:
    data = {}
    for c in counter_configs:
        if c.get("history"):
            data[c["name"]] = [h["people_count"] for h in c["history"]]
    if not data:
        st.info("No history to display yet.")
        return
    min_len = min(len(v) for v in data.values())
    st.line_chart(
        pd.DataFrame({k: v[-min_len:] for k, v in data.items()}),
        width='stretch',
        height=230,
    )


def render_recommendation_banner(rec: dict) -> None:
    action = rec.get("action", "no_action")
    titles = {
        "redirect":     "Redirect Customers to Adjacent Counter",
        "open_counter": "Activate an Additional Counter",
        "no_action":    "Queue Operating Within Normal Parameters",
        "monitor":      "Monitor Queue Closely &mdash; Elevated Activity",
    }
    labels = {
        "redirect":     "ACTION REQUIRED",
        "open_counter": "URGENT ACTION",
        "no_action":    "SYSTEM NORMAL",
        "monitor":      "WATCH REQUIRED",
    }
    sev_map = {
        "redirect":     ("warn", "MEDIUM"),
        "open_counter": ("crit", "HIGH"),
        "no_action":    ("ok",   "LOW"),
        "monitor":      ("warn", "MODERATE"),
    }
    sev_cls, sev_txt = sev_map.get(action, ("ok", "LOW"))
    sev_colors = {
        "ok":   "var(--success)", "warn": "var(--warning)", "crit": "var(--danger)"
    }
    sev_color = sev_colors[sev_cls]

    # Why text
    avs = rec.get("arrival_vs_service", "")
    why_html = ""
    if avs:
        why_html = f'<div class="aac-why">📊 {avs}</div>'

    # Impact
    impact = rec.get("impact", {})
    impact_html = ""
    bw = impact.get("before_wait")
    aw = impact.get("after_wait")
    pm = impact.get("people_moved", "")
    if bw and aw:
        delta_t = bw - aw
        impact_html = f"""
        <div class="aac-impact">
          <div class="aac-impact-item">
            <div class="aac-impact-val" style="color:var(--tx2)">{bw:.0f} min</div>
            <div class="aac-impact-lbl">Current Wait</div>
          </div>
          <div style="font-size:1.2rem;color:var(--tx3);align-self:center">&#8594;</div>
          <div class="aac-impact-item">
            <div class="aac-impact-val">{aw:.0f} min</div>
            <div class="aac-impact-lbl">After Action</div>
          </div>
          <div class="aac-impact-item" style="margin-left:8px">
            <div class="aac-impact-val">−{delta_t:.0f} min</div>
            <div class="aac-impact-lbl">Time Saved</div>
          </div>
          {"<div class=\'aac-impact-item\'><div class=\'aac-impact-val\'>" + str(pm) + "</div><div class=\'aac-impact-lbl\'>Customers Moved</div></div>" if pm else ""}
        </div>"""

    st.markdown(f'''
    <div class="ai-action-card {action}">
      <div class="aac-header">
        <div class="aac-label">AI ACTION CENTER</div>
        <span class="aac-severity" style="background:rgba(0,0,0,.2);border:1px solid {sev_color}40;color:{sev_color}">
          <span style="width:6px;height:6px;border-radius:50%;background:{sev_color};display:inline-block"></span>
          {sev_txt} PRIORITY
        </span>
      </div>
      <div class="aac-title">{rec.get("title", titles.get(action, "Status"))}</div>
      <div class="aac-msg">{rec.get("message", "")}</div>
      {why_html}
      {impact_html}
    </div>
    ''', unsafe_allow_html=True)


def render_groq_alert(alert_res: dict | None, rec: dict) -> None:
    if not alert_res:
        st.markdown('''
        <div class="ai-box" style="text-align:center;padding:24px">
          <div class="ai-tag">AI ALERT</div>
          <div style="font-size:.85rem;color:var(--tx3);margin-top:8px">
            Alert will appear once queue data is available.
          </div>
        </div>
        ''', unsafe_allow_html=True)
        return
    action   = rec.get("action", "no_action")
    severity = {"redirect":"medium","open_counter":"high","no_action":"low","monitor":"medium"}.get(action,"low")
    source   = ("llama-3.3-70b via Groq" if alert_res.get("source") == "groq"
                else "Template (offline fallback)")
    text     = alert_res.get("alert_text", "&mdash;")
    sev_colors = {"high":"var(--danger)","medium":"var(--warning)","low":"var(--success)"}
    sev_labels = {"high":"HIGH RISK","medium":"MODERATE","low":"NORMAL"}
    sc = sev_colors.get(severity,"var(--tx3)")
    sl = sev_labels.get(severity,"INFO")
    st.markdown(f'''
    <div class="ai-box {severity}">
      <div class="ai-tag">AI NARRATIVE ALERT</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;margin-top:4px">
        <span style="display:inline-flex;align-items:center;gap:4px;background:rgba(0,0,0,.2);
               border:1px solid {sc}40;color:{sc};padding:2px 9px;border-radius:20px;
               font-size:.6rem;font-weight:700;letter-spacing:.06em">
          <span style="width:5px;height:5px;border-radius:50%;background:{sc};display:inline-block"></span>
          {sl}
        </span>
      </div>
      <p class="ai-text">{text}</p>
      <div class="ai-src">Model: {source}</div>
    </div>
    ''', unsafe_allow_html=True)
    if alert_res.get("error"):
        st.caption(f"ℹ️ {alert_res['error']}")


def render_prediction_expander(preds: list[dict]) -> None:
    with st.expander("🔍 Prediction Analysis", expanded=False):
        for p in preds:
            if p.get("sentence"):
                if p.get("capped"):
                    st.warning(f"&#9888;️ {p['sentence']}")
                elif p.get("alert"):
                    st.error(f"🔴 {p['sentence']}")
                elif p.get("predicted_count") is None:
                    st.info(f"⏳ {p['sentence']}")
                else:
                    st.info(f"ℹ️ {p['sentence']}")
            if p.get("slope") is not None:
                capped_note = "  |  &#9888;️ capped" if p.get("capped") else ""
                st.caption(
                    f"  Trend: {p['slope']:+.2f} ppl/min  |  R² = {p.get('r_squared', '?')}{capped_note}"
                )


def render_mm1_expander(counter_configs: list[dict]) -> None:
    with st.expander("📐 Queueing Theory &mdash; M/M/1 Model", expanded=False):
        st.markdown("""
**M/M/1 Queue** &mdash; single server, Poisson arrivals (λ), exponential service times (μ).

| Symbol | Formula | Meaning |
|--------|---------|---------|
| ρ = λ/μ | Traffic intensity | Must be < 1 for stability |
| Lq = ρ²/(1−ρ) | Mean queue length | Avg customers waiting |
| **Wq = λ/(μ(μ−λ))** | **Mean wait time** | Full M/M/1 formula (minutes) |
| Wq ≈ N/μ | Simplified estimate | Used in live dashboard table |
        """)
        rows = []
        for c in counter_configs:
            if c.get("arrival_rate"):
                mm1 = estimate_wait_time_mm1(c["arrival_rate"], c["service_rate"])
                rows.append({
                    "Counter":  c["name"],
                    "λ/min":    c["arrival_rate"],
                    "μ/min":    c["service_rate"],
                    "ρ":        mm1.get("rho", "&mdash;"),
                    "Wq (min)": mm1.get("Wq_minutes", "∞"),
                    "Stable":   "&#10003;" if mm1.get("stable") else "&#10007; Overloaded",
                })
            else:
                rows.append({
                    "Counter":  c["name"],
                    "λ/min":    "N/A (real data)",
                    "μ/min":    c["service_rate"],
                    "ρ":        "&mdash;",
                    "Wq (min)": "&mdash;",
                    "Stable":   "&mdash;",
                })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def render_hero_metric(readings, wait_times, preds, rec, threshold) -> None:
    """Big, obvious facility status card &mdash; judges see this in 2 seconds."""
    if not readings:
        return

    # Compute per-counter risks
    counter_risks = []
    for i, r in enumerate(readings):
        wt = wait_times[i] if i < len(wait_times) else None
        pc = preds[i].get("predicted_count") if i < len(preds) else None
        counter_risks.append(classify_risk(
            current_count=r["people_count"],
            predicted_count=pc,
            wait_time=wt,
            threshold=threshold,
        ))
    facility = classify_facility_risk(counter_risks)

    total   = sum(r["people_count"] for r in readings)
    valid_w = [w for w in wait_times if w is not None]
    max_w   = max(valid_w) if valid_w else 0
    avg_w   = sum(valid_w) / len(valid_w) if valid_w else 0

    lvl   = facility["level"]
    icon  = RISK_LABELS[lvl][0]
    label = RISK_LABELS[lvl][1]
    color = RISK_LABELS[lvl][2]

    rec_title = rec.get("title", rec.get("action", "&mdash;").replace("_", " ").title())

    # Map risk level to CSS class
    css_cls = {"normal":"ok","moderate":"warn","high":"warn","critical":"crit"}.get(lvl,"ok")
    level_colors = {"ok":"var(--success)","warn":"var(--warning)","crit":"var(--danger)"}
    lc = level_colors.get(css_cls,"var(--tx2)")

    # Total predicted (sum of worst predictions)
    total_pred = sum(
        (preds[i].get("predicted_count") or r["people_count"])
        for i,r in enumerate(readings)
    )

    st.markdown(f'''
    <div class="fac-hero {css_cls}">
      <div class="fac-hero-row">
        <div class="fac-status-block">
          <div class="fac-label">FACILITY STATUS</div>
          <div class="fac-level" style="color:{lc}">{label}</div>
          <div class="fac-summary">{facility['summary']}</div>
        </div>
        <div class="fac-divider"></div>
        <div class="fac-metrics">
          <div class="fac-metric">
            <div class="fac-metric-val" style="color:var(--tx)">{total}</div>
            <div class="fac-metric-lbl">Current Queue</div>
          </div>
          <div class="fac-metric">
            <div class="fac-metric-val" style="color:{lc}">{total_pred}</div>
            <div class="fac-metric-lbl">Predicted</div>
          </div>
          <div class="fac-metric">
            <div class="fac-metric-val" style="color:var(--tx)">{avg_w:.0f} min</div>
            <div class="fac-metric-lbl">Avg Wait</div>
          </div>
          <div class="fac-metric">
            <div class="fac-metric-val" style="color:{lc}">{max_w:.0f} min</div>
            <div class="fac-metric-lbl">Peak Wait</div>
          </div>
          <div class="fac-metric" style="max-width:180px;text-align:left">
            <div style="font-size:1rem;font-weight:700;color:var(--tx);line-height:1.25">{rec_title}</div>
            <div class="fac-metric-lbl">Recommended Action</div>
          </div>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)
    return counter_risks, facility


def render_four_questions(readings, wait_times, preds, rec, horizon) -> None:
    """The 4 core questions every operations manager needs answered."""
    if not readings:
        return

    total      = sum(r["people_count"] for r in readings)
    valid_w    = [w for w in wait_times if w is not None]
    avg_wait   = sum(valid_w)/len(valid_w) if valid_w else 0

    # What will happen next &mdash; worst predicted counter
    worst_pred = max(preds, key=lambda p: p.get("predicted_count") or 0) if preds else {}
    pred_count = worst_pred.get("predicted_count")
    pred_min   = worst_pred.get("predicted_in_min", horizon)

    # Why &mdash; arrival vs service
    why_text = rec.get("arrival_vs_service") or "Queue dynamics are being monitored."

    # What to do
    what_text = rec.get("message", "No action needed.")
    what_title = rec.get("title", "Monitor")

    pred_str = f"{pred_count} people in +{int(pred_min)} min" if pred_count else "Collecting data..."

    st.markdown('<div class="sec-hdr">Situational Intelligence</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    q_colors = ["var(--accent)","var(--accent2)","var(--warning)","var(--success)"]
    q_titles = ["① NOW","② NEXT","③ WHY","④ ACTION"]
    q_vals   = [
        f"{total} waiting",
        pred_str,
        why_text[:80] + ("..." if len(why_text) > 80 else ""),
        what_title,
    ]
    q_subs   = [
        f"Avg wait: {avg_wait:.0f} min",
        (worst_pred.get("sentence","")[:55] + "...") if worst_pred.get("sentence") else "",
        "",
        (what_text[:80] + "...") if len(what_text) > 80 else what_text,
    ]
    for col, qc, qt, qv, qs in zip([c1,c2,c3,c4], q_colors, q_titles, q_vals, q_subs):
        with col:
            st.markdown(f'''
            <div class="q-card" style="border-top-color:{qc}">
              <div class="q-num" style="color:{qc}">{qt}</div>
              <div class="q-val">{qv}</div>
              <div class="q-sub">{qs}</div>
            </div>''', unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


def render_forecast_chart(counter_configs, preds, threshold, horizon) -> None:
    """Line chart: historical + forecast + threshold. Uses Streamlit native (no Plotly dep)."""
    st.markdown('<div class="sec-hdr">Queue Trend + Forecast</div>', unsafe_allow_html=True)

    rows = []
    max_hist = max((len(c.get("history", [])) for c in counter_configs), default=0)
    window = min(max_hist, 20)  # show last 20 readings

    for c in counter_configs:
        hist = c.get("history", [])
        if not hist:
            continue
        recent = hist[-window:]
        for idx, h in enumerate(recent):
            rows.append({
                "tick": idx,
                "counter": c["name"],
                "people_count": h["people_count"],
                "type": "Historical",
            })

    # Forecast points appended after historical
    forecast_rows = []
    for i, c in enumerate(counter_configs):
        if i >= len(preds):
            continue
        pc = preds[i].get("predicted_count")
        if pc is None:
            continue
        forecast_rows.append({
            "tick": window,
            "counter": c["name"] + " (forecast)",
            "people_count": pc,
            "type": "Forecast",
        })

    # Threshold line
    threshold_rows = [
        {"tick": t, "counter": f"&#9888; Threshold ({threshold})", "people_count": threshold, "type": "Threshold"}
        for t in range(window + 1)
    ]

    all_rows = rows + forecast_rows + threshold_rows
    if not all_rows:
        st.info("No data for chart yet.")
        return

    df_chart = pd.DataFrame(all_rows)
    # Pivot so each counter is a column
    df_pivot = df_chart.pivot_table(index="tick", columns="counter",
                                    values="people_count", aggfunc="first")
    st.line_chart(df_pivot, height=240)
    st.caption(
        f"📈 Solid lines = historical | 🔮 Forecast = predicted +{horizon}m | "
        f"&#9888; Threshold = {threshold} people"
    )


def render_what_if_panel(readings, wait_times, counter_configs, service_rate) -> None:
    """Interactive What-If scenario simulator."""
    with st.expander("🔬 What-If Simulator &mdash; Test Scenarios Before Acting", expanded=False):
        st.markdown(
            "Simulate the effect of interventions using the actual queue model. "
            "Numbers are calculated &mdash; not estimated."
        )
        scenario = st.selectbox(
            "Choose a what-if scenario:",
            ["Open Additional Counter", "Redirect Customers", "Increase Service Rate"],
            key="what_if_choice",
        )

        counters_simple = [
            {"name": c["name"], "people_count": c["history"][-1]["people_count"]
             if c.get("history") else 0}
            for c in counter_configs
        ]

        if scenario == "Open Additional Counter":
            absorption = st.slider("Load absorbed by new counter (%)", 20, 70, 45, 5,
                                   key="wi_absorption") / 100
            result = scenario_open_counter(counters_simple, wait_times, service_rate,
                                           absorption=absorption)
        elif scenario == "Redirect Customers":
            result = scenario_redirect_customers(counters_simple, wait_times, service_rate)
        else:
            factor = st.slider("Service rate multiplier", 1.2, 3.0, 1.5, 0.1, key="wi_factor")
            result = scenario_increase_service(counters_simple, wait_times, service_rate, factor=factor)

        if "error" in result:
            st.warning(result["error"])
        else:
            st.success(f"**{result['scenario_label']}**")
            st.info(result["summary"])

            before_w = result.get("before_wait") or result.get("wait_from_before")
            after_w  = result.get("after_wait")  or result.get("wait_from_after")
            if before_w and after_w:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Before", f"{before_w:.0f} min", label_visibility="visible")
                col_b.metric("After",  f"{after_w:.0f} min",
                             delta=f"{after_w - before_w:.0f} min",
                             delta_color="inverse")
                col_c.metric("Reduction", f"{before_w - after_w:.0f} min saved")


def render_multi_horizon_expander(preds_multi) -> None:
    """Show +5/+10/+15/+20 min forecast table."""
    if not preds_multi:
        return
    with st.expander("🔮 Multi-Horizon Forecast (+5/+10/+15/+20 min)", expanded=False):
        rows = []
        for pm in preds_multi:
            if not pm.get("has_data"):
                continue
            row = {"Counter": pm.get("name", "?")}
            for h in FORECAST_HORIZONS:
                hp = pm["horizons"].get(h, {})
                pc = hp.get("predicted_count")
                rl = hp.get("range_low")
                rh = hp.get("range_high")
                if pc is not None:
                    row[f"+{h}m"] = f"{pc} [{rl}&ndash;{rh}]"
                else:
                    row[f"+{h}m"] = "&mdash;"
            row["Trend"] = f"{pm.get('slope', 0):+.2f} ppl/min" if pm.get("slope") else "&mdash;"
            row["R²"]    = str(pm.get("r_squared", "&mdash;"))
            row["Method"] = pm.get("forecast_method", "&mdash;")
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("Range shown as [low&ndash;high] = ±1 std of historical residuals (~68% empirical interval).")
        else:
            st.info(f"Collecting data... need at least {MIN_READINGS} readings per counter.")


def render_login_page(theme: str) -> None:
    """Login page: dark navy left panel with queue-flow, clean white right form."""

    st.markdown(
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&display=swap');"
        "#MainMenu,footer,header{visibility:hidden !important;}"
        "[data-testid='stHeader']{display:none !important;}"
        "[data-testid='stSidebar']{display:none !important;}"
        ".stApp,[data-testid='stAppViewContainer']{background:#0A0F1E !important;min-height:100vh;}"
        ".block-container{padding:0 !important;max-width:100% !important;}"
        "[data-testid='stHorizontalBlock']{gap:0 !important;min-height:100vh;align-items:stretch;}"
        "[data-testid='stHorizontalBlock'] > [data-testid='stColumn']:first-child{"
          "background:linear-gradient(145deg,#04070F 0%,#071428 40%,#0A1F3E 75%,#0C1A34 100%) !important;"
          "padding:0 !important;overflow:hidden;position:relative;}"
        "[data-testid='stHorizontalBlock'] > [data-testid='stColumn']:last-child{"
          "background:#F7F8FC !important;padding:0 !important;"
          "display:flex;align-items:center;justify-content:center;}"
        "[data-testid='stHorizontalBlock'] > [data-testid='stColumn']:last-child > div{"
          "width:100%;max-width:400px;padding:48px 40px;margin:0 auto;}"
        ".lp-wrap{min-height:100vh;display:flex;flex-direction:column;align-items:center;"
          "justify-content:center;padding:52px 40px;position:relative;z-index:2;}"
        ".lp-grid{position:absolute;inset:0;z-index:0;"
          "background-image:linear-gradient(rgba(0,194,255,.04) 1px,transparent 1px),"
          "linear-gradient(90deg,rgba(0,194,255,.04) 1px,transparent 1px);"
          "background-size:40px 40px;}"
        ".lp-orb{position:absolute;width:360px;height:360px;border-radius:50%;"
          "background:radial-gradient(circle,rgba(0,194,255,.08) 0%,transparent 70%);"
          "top:-80px;right:-100px;z-index:0;}"
        ".lp-orb2{position:absolute;width:260px;height:260px;border-radius:50%;"
          "background:radial-gradient(circle,rgba(0,128,255,.07) 0%,transparent 70%);"
          "bottom:-60px;left:-60px;z-index:0;}"
        ".lp-logo{width:52px;height:52px;border-radius:14px;"
          "background:linear-gradient(135deg,#0050CC,#00C2FF);"
          "display:flex;align-items:center;justify-content:center;"
          "font-size:1.6rem;font-weight:900;color:#fff;"
          "margin-bottom:20px;position:relative;z-index:1;"
          "box-shadow:0 8px 24px rgba(0,194,255,.30);}"
        ".lp-brand{font-family:'IBM Plex Sans',sans-serif;font-size:2.4rem;font-weight:700;"
          "color:#E2EAF8;letter-spacing:-.5px;margin-bottom:6px;position:relative;z-index:1;text-align:center;}"
        ".lp-brand span{color:#00C2FF;}"
        ".lp-tagline{font-size:.78rem;color:rgba(136,153,187,.75);text-align:center;"
          "letter-spacing:.06em;margin-bottom:36px;position:relative;z-index:1;"
          "text-transform:uppercase;font-weight:500;}"
        ".lp-flow{position:relative;z-index:1;width:100%;max-width:300px;margin-bottom:32px;}"
        ".lp-ftitle{font-size:.58rem;font-weight:700;text-transform:uppercase;"
          "letter-spacing:.14em;color:rgba(0,194,255,.55);margin-bottom:14px;text-align:center;}"
        ".flow-row{display:flex;align-items:center;justify-content:center;gap:6px;margin-bottom:8px;}"
        ".flow-node{background:rgba(0,194,255,.07);border:1px solid rgba(0,194,255,.18);"
          "border-radius:8px;padding:6px 14px;font-size:.72rem;font-weight:600;"
          "color:#8BAFC8;white-space:nowrap;}"
        ".flow-node.hi{background:rgba(0,194,255,.14);border-color:rgba(0,194,255,.42);"
          "color:#00C2FF;box-shadow:0 0 10px rgba(0,194,255,.12);}"
        ".flow-arrow{color:rgba(0,194,255,.30);font-size:.9rem;}"
        ".fdot{width:8px;height:8px;border-radius:50%;display:inline-block;margin:0 2px;}"
        ".fdot.ok{background:#10B981;} .fdot.warn{background:#F59E0B;} .fdot.crit{background:#EF4444;}"
        ".lp-stats{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;"
          "position:relative;z-index:1;}"
        ".lp-stat{text-align:center;background:rgba(255,255,255,.04);"
          "border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:10px 16px;min-width:68px;}"
        ".lp-sn{font-size:1.25rem;font-weight:800;color:#E2EAF8;}"
        ".lp-sl{font-size:.56rem;color:rgba(136,153,187,.65);text-transform:uppercase;"
          "letter-spacing:.09em;margin-top:2px;}"
        ".rp-lbl{font-size:.6rem;font-weight:700;text-transform:uppercase;"
          "letter-spacing:.12em;color:#0B7285;margin-bottom:22px;"
          "display:flex;align-items:center;gap:8px;}"
        ".rp-lbl::before{content:'';display:inline-block;width:18px;height:2px;"
          "background:#0B7285;border-radius:2px;}"
        ".rp-title{font-family:'IBM Plex Sans',sans-serif;font-size:2rem;font-weight:700;"
          "color:#0B1320;letter-spacing:-.5px;line-height:1.15;margin-bottom:6px;}"
        ".rp-sub{font-size:.86rem;color:#5A6A7A;margin-bottom:28px;line-height:1.6;}"
        ".rp-flbl{font-size:.68rem;font-weight:700;text-transform:uppercase;"
          "letter-spacing:.1em;color:#7A8A9E;margin-bottom:8px;display:block;}"
        ".rp-hint{margin-top:14px;padding:10px 14px;background:#EDF2FA;border-radius:8px;"
          "border-left:3px solid #0B7285;font-size:.74rem;color:#5A6A7A;line-height:1.55;}"
        ".rp-hint code{background:#D8E6F0;color:#0B7285;padding:1px 6px;"
          "border-radius:4px;font-size:.73rem;font-weight:600;}"
        ".rp-foot{margin-top:26px;padding-top:16px;border-top:1px solid #E4EAF0;"
          "font-size:.66rem;color:#9AA5B4;text-align:center;"
          "display:flex;align-items:center;justify-content:center;gap:8px;}"
        ".rp-dot{width:3px;height:3px;border-radius:50%;background:#C5CED8;display:inline-block;}"
        ".stTextInput > div > div{border:1.5px solid #D8DEF0 !important;"
          "border-radius:10px !important;background:#FFFFFF !important;color:#0B1320 !important;}"
        ".stTextInput input{color:#0B1320 !important;background:#FFFFFF !important;"
          "caret-color:#0B7285 !important;font-size:.95rem !important;}"
        ".stTextInput input::placeholder{color:#A0AABB !important;opacity:1 !important;}"
        ".stTextInput > div > div:focus-within{border-color:#0B7285 !important;"
          "box-shadow:0 0 0 3px rgba(11,114,133,.12) !important;}"
        ".stButton > button{background:linear-gradient(135deg,#085E72,#0B7285,#0E8FA5) !important;"
          "color:#fff !important;border:none !important;border-radius:10px !important;"
          "font-weight:700 !important;font-size:.95rem !important;padding:14px 0 !important;"
          "box-shadow:0 4px 18px rgba(11,114,133,.38) !important;"
          "transition:all .18s ease !important;}"
        ".stButton > button:hover{box-shadow:0 8px 28px rgba(11,114,133,.52) !important;"
          "transform:translateY(-2px) !important;}"
        ".stAlert{border-radius:10px !important;}"
        "</style>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown(
            "<div class='lp-grid'></div>"
            "<div class='lp-orb'></div>"
            "<div class='lp-orb2'></div>"
            "<div class='lp-wrap'>"
              "<div class='lp-logo'>Q</div>"
              "<div class='lp-brand'>Queue<span>IQ</span></div>"
              "<div class='lp-tagline'>Queue Intelligence Platform</div>"
              "<div class='lp-flow'>"
                "<div class='lp-ftitle'>Live Queue Flow</div>"
                "<div class='flow-row'><div class='flow-node hi'>&#8595; Arrivals &nbsp;1.8/min</div></div>"
                "<div class='flow-row'><span class='flow-arrow'>&#8595;</span></div>"
                "<div class='flow-row'>"
                  "<div class='flow-node' style='padding:8px 20px'>"
                    "Waiting &nbsp;<span style='color:#F59E0B;font-weight:800'>27</span>"
                  "</div>"
                "</div>"
                "<div class='flow-row'><span class='flow-arrow'>&#8595;</span></div>"
                "<div class='flow-row' style='gap:8px;align-items:flex-start'>"
                  "<div style='text-align:center'>"
                    "<div class='flow-node' style='border-color:rgba(239,68,68,.38);color:#F87171'>C-01</div>"
                    "<div style='margin-top:5px'><span class='fdot crit'></span></div>"
                    "<div style='font-size:.57rem;color:rgba(136,153,187,.4);margin-top:3px'>HIGH</div>"
                  "</div>"
                  "<div style='text-align:center'>"
                    "<div class='flow-node' style='border-color:rgba(16,185,129,.38);color:#6EE7B7'>C-02</div>"
                    "<div style='margin-top:5px'><span class='fdot ok'></span></div>"
                    "<div style='font-size:.57rem;color:rgba(136,153,187,.4);margin-top:3px'>LOW</div>"
                  "</div>"
                  "<div style='text-align:center'>"
                    "<div class='flow-node' style='border-color:rgba(245,158,11,.38);color:#FCD34D'>C-03</div>"
                    "<div style='margin-top:5px'><span class='fdot warn'></span></div>"
                    "<div style='font-size:.57rem;color:rgba(136,153,187,.4);margin-top:3px'>MED</div>"
                  "</div>"
                "</div>"
                "<div class='flow-row'><span class='flow-arrow'>&#8595;</span></div>"
                "<div class='flow-row'><div class='flow-node' style='color:rgba(136,153,187,.5)'>&#8594; Exit</div></div>"
              "</div>"
              "<div class='lp-stats'>"
                "<div class='lp-stat'><div class='lp-sn'>3</div><div class='lp-sl'>Modes</div></div>"
                "<div class='lp-stat'><div class='lp-sn'>AI</div><div class='lp-sl'>Forecast</div></div>"
                "<div class='lp-stat'><div class='lp-sn'>Live</div><div class='lp-sl'>CCTV</div></div>"
                "<div class='lp-stat'><div class='lp-sn'>M/M/1</div><div class='lp-sl'>Model</div></div>"
              "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            "<div class='rp-lbl'>Secure Access</div>"
            "<div class='rp-title'>Sign in to<br>QueueIQ</div>"
            "<div class='rp-sub'>Access real-time queue intelligence, AI crowd forecasting, and live CCTV analysis.</div>"
            "<span class='rp-flbl'>&#128273; Admin Password</span>",
            unsafe_allow_html=True,
        )

        pwd = st.text_input(
            "Password", type="password",
            placeholder="Enter your password...",
            key="login_pwd_input",
            label_visibility="collapsed",
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        login_btn = st.button("Sign In \u2192", key="login_btn", use_container_width=True)

        if login_btn:
            if pwd == ADMIN_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            elif pwd:
                st.error("\u274c Incorrect password.")

        st.markdown(
            "<div class='rp-hint'>Demo password: <code>queueiq2024</code></div>"
            "<div class='rp-foot'>"
              "QueueIQ v2.0<span class='rp-dot'></span>"
              "AI Operations Platform<span class='rp-dot'></span>Secured"
            "</div>",
            unsafe_allow_html=True,
        )


def render_full_dashboard(
    readings:        list[dict],
    wait_times:      list,
    preds:           list[dict],
    counter_configs: list[dict],
    rec:             dict,
    alert_res:       dict | None,
    horizon:         int,
    threshold:       int = 30,
    preds_multi:     list[dict] | None = None,
    service_rate:    float = 1.0,
) -> None:
    """Single entry-point used by ALL three modes."""
    # 1. Hero Metric
    render_hero_metric(readings, wait_times, preds, rec, threshold)
    # 2. Situational intelligence
    render_four_questions(readings, wait_times, preds, rec, horizon)
    # 3. Metrics row
    render_metrics(readings, wait_times, preds, rec)
    st.markdown("---")
    # 4. Counter cards + table
    st.markdown('<div class="sec-hdr">Live Counter Status</div>', unsafe_allow_html=True)
    render_counter_table(readings, wait_times, preds, horizon)
    st.markdown("")
    render_counter_cards(readings, wait_times, preds)
    # 5. Forecast chart
    render_forecast_chart(counter_configs, preds, threshold, horizon)
    # 6. Recommendation
    st.markdown('<div class="sec-hdr">AI Recommendation</div>', unsafe_allow_html=True)
    render_recommendation_banner(rec)
    # 7. AI Alert
    st.markdown('<div class="sec-hdr">AI Narrative Alert</div>', unsafe_allow_html=True)
    render_groq_alert(alert_res, rec)
    # 8. Multi-horizon table
    if preds_multi:
        render_multi_horizon_expander(preds_multi)
    # 9. Prediction expander
    render_prediction_expander(preds)
    # 10. What-If
    render_what_if_panel(readings, wait_times, counter_configs, service_rate)


ADMIN_PASSWORD = "queueiq2024"


# ── Auth gate &mdash; runs before anything else ────────────────
_theme = st.session_state.get("theme", "dark")   # NIGHT MODE default (command center)
inject_theme(_theme)

if not st.session_state.get("authenticated", False):
    render_login_page(_theme)
    st.stop()

# ── RESTORE SIDEBAR — login page hides it, force it back ────────────────
st.markdown(
    "<style>"
    "section[data-testid='stSidebar']{"
      "display:flex !important;"
      "visibility:visible !important;"
      "width:260px !important;"
      "min-width:260px !important;"
      "max-width:260px !important;"
      "transform:none !important;"
      "opacity:1 !important;"
    "}"
    "section[data-testid='stSidebar'] > div:first-child{"
      "display:flex !important;"
      "flex-direction:column !important;"
    "}"
    "</style>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# DEMO SCENARIO LOADER
# ─────────────────────────────────────────────────────────
_DEMO_SCENARIOS = {
    "Normal (low traffic)": {
        "preset": "Bank",
        "n_counters": 3,
        "overrides": [5, 4, 6],       # people counts to inject
        "desc": "Normal operations &mdash; low load, no intervention needed.",
    },
    "Rush Hour": {
        "preset": "Bank",
        "n_counters": 3,
        "overrides": [27, 22, 18],
        "desc": "Rush hour &mdash; multiple counters approaching threshold.",
    },
    "Sudden Surge": {
        "preset": "Bank",
        "n_counters": 3,
        "overrides": [38, 7, 12],
        "desc": "Counter 1 is surging &mdash; redirect or open new counter recommended.",
    },
    "Counter Failure": {
        "preset": "Bank",
        "n_counters": 2,
        "overrides": [34, 28],
        "desc": "Counter 3 offline &mdash; remaining counters overloaded.",
    },
    "All Clear": {
        "preset": "Bank",
        "n_counters": 4,
        "overrides": [3, 2, 4, 1],
        "desc": "Post-rush &mdash; all counters clear, monitoring only.",
    },
}


def _load_demo_scenario(scenario_name: str) -> None:
    """Load a pre-configured demo scenario into session_state."""
    sc = _DEMO_SCENARIOS.get(scenario_name)
    if sc is None:
        return
    preset    = sc["preset"]
    n_ctr     = sc["n_counters"]
    overrides = sc.get("overrides", [])

    # Reset state fresh
    _init_sim_state(preset, n_ctr)

    # Inject override counts into counter history so predictors have data
    cs = st.session_state["counters_state"]
    import time as _time
    base_t = _time.time() - 600   # 10 min of synthetic history
    for i, c in enumerate(cs):
        target = overrides[i] if i < len(overrides) else 5
        # Build a realistic ramp to the target count
        for tick_i in range(12):
            ramp = int(round(target * (tick_i + 1) / 12))
            reading = {
                "counter_id":      c["id"],
                "name":            c["name"],
                "people_count":    ramp,
                "timestamp_epoch": base_t + tick_i * 30,
                "timestamp_str":   datetime.fromtimestamp(base_t + tick_i * 30).strftime("%H:%M"),
            }
            c["history"].append(reading)

    st.session_state["_demo_desc"] = sc["desc"]
    st.session_state["tick_count"] = 12  # pretend 12 ticks have run
    st.toast(f"🎬 Loaded: {scenario_name}", icon="&#10003;")


# ─────────────────────────────────────────────────────────
# SESSION STATE HELPERS  (simulation mode)
# ─────────────────────────────────────────────────────────

def _init_sim_state(preset: str, n_ctr: int) -> None:
    st.session_state["counters_state"]  = build_counters_state(preset, n_ctr)
    st.session_state["tick_count"]      = 0
    st.session_state["last_readings"]   = []
    st.session_state["last_wait_times"] = []
    st.session_state["last_preds"]      = []
    st.session_state["last_rec"]        = {}
    st.session_state["last_alert"]      = None



with st.sidebar:
    # ── App title + Theme toggle ──────────────────────────
    _title_col, _toggle_col = st.columns([3, 1])
    with _title_col:
        st.markdown("### &#128202; QueueIQ")
        st.caption("Real-time queue intelligence")
    with _toggle_col:
        st.markdown("<div style='padding-top:14px'></div>", unsafe_allow_html=True)
        _is_dark = _theme == "dark"
        if st.button(
            "☀️" if _is_dark else "🌙",
            key="theme_btn",
            help="Switch to Light Mode" if _is_dark else "Switch to Dark Mode",
        ):
            st.session_state["theme"] = "light" if _is_dark else "dark"
            st.rerun()
    st.markdown("---")

    # ── Admin badge + Logout ──────────────────────────────
    _lcol, _rcol = st.columns([3, 1])
    with _lcol:
        st.markdown(
            '<div style="font-size:.72rem;color:#A78BFA;font-weight:600;">🔐 Admin Session Active</div>',
            unsafe_allow_html=True,
        )
    with _rcol:
        if st.button("⏏️", key="logout_btn", help="Logout"):
            st.session_state["authenticated"] = False
            st.session_state.pop("_auto_login", None)
            st.rerun()
    st.markdown("---")


    # ── Mode toggle &mdash; TOP of sidebar ─────────────────────
    st.markdown('<div class="sb-lbl">Data Source</div>', unsafe_allow_html=True)
    mode_choice = st.radio(
        "mode_radio",
        ["🔁 Simulation Mode", "&#128194; Upload Real Data", "📹 AI CCTV Analysis"],
        label_visibility="collapsed",
        key="mode_choice",
    )
    is_simulation = mode_choice.startswith("🔁")
    is_cv         = mode_choice.startswith("📹")

    # ── Demo Mode scenarios ─────────────────────────────────
    if is_simulation:
        st.markdown('---')
        st.markdown('<div class="sb-lbl">🎬 Demo Scenarios</div>', unsafe_allow_html=True)
        demo_scenario = st.selectbox(
            "Load scenario",
            ["&mdash; Manual &mdash;", "Normal (low traffic)", "Rush Hour", "Sudden Surge",
             "Counter Failure", "All Clear"],
            key="demo_scenario",
            label_visibility="collapsed",
        )
        if st.button("▶ Load Scenario", key="load_demo", use_container_width=True):
            _load_demo_scenario(demo_scenario)
    st.markdown("---")

    # ── Shared controls ───────────────────────────────────
    st.markdown('<div class="sb-lbl">Prediction Settings</div>', unsafe_allow_html=True)
    horizon_min = st.slider("🔮 Prediction Horizon (min)", 5, 60, 20, 5, key="horizon_min")
    threshold   = st.slider("&#9888;️ Alert Threshold (people)", 10, 60, 30, 5, key="threshold")
    st.markdown("---")

    if is_simulation:
        # ── Simulation-specific controls ──────────────────
        st.markdown('<div class="sb-lbl">Location & Setup</div>', unsafe_allow_html=True)
        preset_choice = st.selectbox(
            "📍 Location Preset",
            list(PRESETS.keys()),
            key="preset_choice",
        )
        n_counters = st.slider("🏢 Number of Counters", 1, 6, 3, 1, key="n_counters")
        st.markdown("---")

        st.markdown('<div class="sb-lbl">Simulation Controls</div>', unsafe_allow_html=True)
        running      = st.toggle("▶️ Auto-play Simulation", value=False, key="running")
        ca, cb       = st.columns(2)
        with ca:
            advance_tick = st.button("⏭ Advance\nTick", width='stretch')
        with cb:
            reset_sim = st.button("🔄 Reset\nSim", width='stretch')
        tick_speed = st.select_slider(
            "⚡ Tick Speed",
            options=[1, 2, 3, 5, 10],
            value=3,
            format_func=lambda x: f"{x}s",
        )

        st.markdown("---")
        st.markdown("**System Pipeline**")
        st.markdown(
            "1. 📡 **Detect** &mdash; queue state\n"
            "2. 📐 **Estimate** &mdash; wait time, utilisation\n"
            "3. 🔮 **Forecast** &mdash; +5/10/15/20 min\n"
            "4. 🚨 **Risk** &mdash; LOW/MEDIUM/HIGH/CRITICAL\n"
            "5. 🎯 **Recommend** &mdash; action + impact\n"
            "6. 🤖 **AI Alert** &mdash; Groq LLM narrative"
        )
    elif not is_cv:
        # ── Upload-specific controls ───────────────────────
        st.markdown('<div class="sb-lbl">Real Data Settings</div>', unsafe_allow_html=True)
        service_rate = st.slider(
            "&#9881;️ Service Rate (cust/min)",
            min_value=0.5, max_value=5.0, value=1.0, step=0.5,
            key="service_rate",
            help="How many customers each counter can serve per minute. "
                 "Adjust based on your location type.",
        )
        st.markdown("---")

        st.markdown('<div class="sb-lbl">Sample Data</div>', unsafe_allow_html=True)
        st.caption(
            "No real data? Download the template &mdash; Counter 1 peaks at 35 people "
            "and triggers a live alert."
        )
        st.download_button(
            "&#128229; Download Sample CSV",
            data=generate_sample_csv(),
            file_name="sample_queue_data.csv",
            mime="text/csv",
            width='stretch',
        )
        st.markdown("---")
        # CSV uploader lives in the MAIN AREA (see RENDER section below)
        # &mdash; kept here only as a comment so the sidebar layout is clear.
        st.session_state.setdefault("_uf", None)

    elif is_cv:
        # ── Camera/Video Feed controls ──────────────────────────────
        if not _CV_AVAILABLE:
            st.warning(
                "YOLOv8 unavailable in this environment.\n"
                "Falling back to Simulation Mode.",
                icon="&#9888;️",
            )
        else:
            st.markdown('<div class="sb-lbl">Camera Settings</div>', unsafe_allow_html=True)
            cv_service_rate = st.slider(
                "Service Rate (cust/min)",
                min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                key="cv_service_rate",
                help="How fast each counter serves customers.",
            )
            st.markdown("---")
            # File uploader + counter-name inputs live in the MAIN AREA
            # (see RENDER section below) &mdash; sidebar only has the service rate slider.
            st.session_state.setdefault("_cv_files", [])
            st.session_state.setdefault("_cv_labels", [])



def run_sim_tick(counters_state: list, horizon: int, thresh: int) -> None:
    """Advance simulation by one tick; update all session_state keys."""
    readings = generate_reading(counters_state, tick_seconds=30)
    st.session_state["last_readings"] = readings
    st.session_state["tick_count"]   += 1
    tick = st.session_state["tick_count"]

    # Wait-time estimates
    wait_times = [
        estimate_wait_time(readings[i]["people_count"], counters_state[i]["service_rate"])
        for i in range(len(counters_state))
    ]
    st.session_state["last_wait_times"] = wait_times

    # Single-horizon predictions (backward-compat display)
    preds = []
    for i, counter in enumerate(counters_state):
        pred = predict_future_count(
            counter["history"],
            horizon_minutes=horizon,
            counter_name=counter["name"],
            threshold=thresh,
        )
        pred["counter_id"] = counter["id"]
        preds.append(pred)
    st.session_state["last_preds"] = preds

    # Multi-horizon predictions (+5/+10/+15/+20 min)
    preds_multi = []
    for counter in counters_state:
        mh = predict_multi_horizon(
            counter["history"],
            counter_name=counter["name"],
            threshold=thresh,
        )
        mh["name"] = counter["name"]
        preds_multi.append(mh)
    st.session_state["last_preds_multi"] = preds_multi

    # Recommendation (with service_rate for what-if calculation)
    current_summary = [
        {"counter_id": r["counter_id"], "name": r["name"], "people_count": r["people_count"]}
        for r in readings
    ]
    sr = counters_state[0]["service_rate"] if counters_state else 1.0
    rec = recommend_action(current_summary, preds, wait_times, service_rate=sr)
    st.session_state["last_rec"] = rec

    # Groq alert &mdash; call on tick 1 and every 3 ticks (frequent enough for live demo)
    if tick == 1 or tick % 3 == 0:
        alert_states = [
            {"name": r["name"], "people_count": r["people_count"], "wait_time": wait_times[i]}
            for i, r in enumerate(readings)
        ]
        st.session_state["last_alert"] = generate_alert(alert_states, preds, rec)


# ─────────────────────────────────────────────────────────
# COMPUTATION  ── mode-routing
# ─────────────────────────────────────────────────────────
# Safe defaults &mdash; ONLY for non-simulation modes (overwritten by sidebar when is_simulation=True)
# IMPORTANT: Do NOT derive preset_choice/n_counters from session_state here;
# the sidebar widget assignments (lines ~861-866) own these values.
advance_tick  = False
reset_sim     = False
running       = False
tick_speed    = 3
service_rate  = st.session_state.get("service_rate", 1.0)
# These two are set by the sidebar when is_simulation=True; the session_state
# fallback is only used if somehow we reach the computation block without the
# sidebar branch running (shouldn't happen, but defensively safe).
preset_choice = st.session_state.get("preset_choice",  list(PRESETS.keys())[0])
n_counters    = int(st.session_state.get("n_counters",   3))

readings        = []
wait_times      = []
preds           = []
preds_multi     = []
rec             = {}
alert_res       = None
counter_configs = []
header_info     = ""

# ── SIMULATION MODE ───────────────────────────────────────
if is_simulation:
    # ── Detect config changes using the LIVE widget values ──
    # (preset_choice and n_counters are now set by the sidebar above, not guessed)
    prev_preset = st.session_state.get("_prev_preset")
    prev_n_ctr  = st.session_state.get("_prev_n_ctr")
    cfg_changed = (preset_choice != prev_preset) or (n_counters != prev_n_ctr)

    if "counters_state" not in st.session_state or reset_sim:
        # First load or explicit reset &#8594; full reinitialise
        _init_sim_state(preset_choice, n_counters)
        st.session_state["_prev_preset"] = preset_choice
        st.session_state["_prev_n_ctr"]  = n_counters

    elif cfg_changed:
        # Preset or counter-count changed &mdash; rebuild counters, clear old data
        _init_sim_state(preset_choice, n_counters)
        st.session_state["_prev_preset"]  = preset_choice
        st.session_state["_prev_n_ctr"]   = n_counters

    counters_state  = st.session_state["counters_state"]
    counter_configs = counters_state

    # Run tick when:
    #   (a) "Advance Tick" button clicked      &#8594; advance_tick=True
    #   (b) Auto-play toggle is ON             &#8594; running=True
    #   (c) First ever load (tick_count==0)    &#8594; auto-start so data shows immediately
    #       EXCEPT when Reset was just clicked &#8594; reset_sim=True prevents auto-fire
    #       so Reset correctly returns to the empty-state placeholder.
    _tick_count_now = st.session_state.get("tick_count", 0)
    if advance_tick or running or (_tick_count_now == 0 and not reset_sim):
        run_sim_tick(counters_state, horizon_min, threshold)

    readings        = st.session_state.get("last_readings",    [])
    wait_times      = st.session_state.get("last_wait_times",  [])
    preds           = st.session_state.get("last_preds",       [])
    preds_multi     = st.session_state.get("last_preds_multi", [])
    rec             = st.session_state.get("last_rec",         {})
    alert_res       = st.session_state.get("last_alert")
    tick_n          = st.session_state.get("tick_count", 0)
    n_display       = len(counters_state)
    header_info     = f"Tick #{tick_n} &middot; {preset_choice} &middot; {n_display} counter{'s' if n_display != 1 else ''}"

# ── CAMERA / VIDEO FEED MODE ──────────────────────────────────────────────
elif is_cv:
    cv_files  = st.session_state.get("_cv_files", [])
    cv_labels = st.session_state.get("_cv_labels", [])
    cv_sr     = st.session_state.get("cv_service_rate", 1.0)

    if not _CV_AVAILABLE:
        # Graceful degradation &mdash; switch to simulation instead of crashing
        is_simulation = True
        is_cv         = False
        header_info   = "Camera mode unavailable &mdash; using Simulation Mode"
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
        header_info = "Camera mode ready &mdash; upload footage in the sidebar"

    else:
        # Cache detection by (file identities + service rate) to avoid
        # re-running YOLO on every Streamlit rerun (it's slow on CPU).
        cv_cache_key = "_".join(f"{f.name}{f.size}" for f in cv_files) + str(cv_sr)
        if st.session_state.get("_cv_cache_key") != cv_cache_key:
            try:
                yolo_model = _get_yolo_model()
                with st.spinner("Running person detection... (YOLOv8n, CPU)"):
                    # reset file pointers &mdash; Streamlit may have read them already
                    for f in cv_files:
                        f.seek(0)
                    labels_to_use = cv_labels or [f"Counter {i+1}" for i in range(len(cv_files))]
                    cv_configs = _cvd.build_cv_counter_configs(
                        cv_files, labels_to_use, cv_sr, model=yolo_model
                    )
                if cv_configs:
                    st.session_state["_cv_configs"]    = cv_configs
                    st.session_state["_cv_cache_key"]  = cv_cache_key
                    st.session_state["_cv_groq_key"]   = None
                else:
                    st.warning(
                        "Person detection returned no results. "
                        "Try a clearer image or a different file."
                    )
                    cv_configs = []
            except Exception as _cv_err:
                st.warning(
                    "Camera/Video mode unavailable in this environment &mdash; "
                    "falling back to Simulation Mode."
                )
                cv_configs = []
        else:
            cv_configs = st.session_state.get("_cv_configs", [])

        if cv_configs:
            counter_configs = cv_configs
            readings = [c["history"][-1] for c in counter_configs if c["history"]]
            wait_times = [
                estimate_wait_time(r["people_count"], counter_configs[i]["service_rate"])
                for i, r in enumerate(readings)
            ]
            preds = []
            for i, c in enumerate(counter_configs):
                pred = predict_future_count(
                    c["history"],
                    horizon_minutes=horizon_min,
                    counter_name=c["name"],
                    threshold=threshold,
                )
                pred["counter_id"] = c["id"]
                preds.append(pred)

            current_summary = [
                {"counter_id": r["counter_id"], "name": r["name"],
                 "people_count": r["people_count"]}
                for r in readings
            ]
            rec = recommend_action(current_summary, preds, wait_times)

            cv_groq_key = cv_cache_key + rec.get("action", "")
            if st.session_state.get("_cv_groq_key") != cv_groq_key:
                alert_states = [
                    {"name": r["name"], "people_count": r["people_count"],
                     "wait_time": wait_times[i]}
                    for i, r in enumerate(readings)
                ]
                alert_res = generate_alert(alert_states, preds, rec)
                st.session_state["_cv_alert"]    = alert_res
                st.session_state["_cv_groq_key"] = cv_groq_key
            else:
                alert_res = st.session_state.get("_cv_alert")

            total_frames = sum(len(c["history"]) for c in counter_configs)
            header_info  = (
                f"{len(counter_configs)} counters &middot; {total_frames} frames analysed &middot; "
                f"svc rate {cv_sr}/min"
            )

            # ── Build exportable CSV from CCTV detected data ──────────
            def _build_cv_csv(configs: list) -> bytes:
                """
                Converts CCTV detection output into a CSV matching the
                'Upload Real Data' format: timestamp, counter_name, people_count
                """
                import io as _io
                from datetime import datetime as _dt
                rows = []
                for cfg in configs:
                    for h in cfg.get("history", []):
                        rows.append({
                            "timestamp":    h.get("timestamp_str", _dt.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "counter_name": h.get("name", cfg.get("name", "Counter")),
                            "people_count": h.get("people_count", 0),
                        })
                # If no history rows, use the last single reading
                if not rows:
                    for r in readings:
                        rows.append({
                            "timestamp":    _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "counter_name": r.get("name", "Counter"),
                            "people_count": r.get("people_count", 0),
                        })
                buf = _io.BytesIO()
                pd.DataFrame(rows).to_csv(buf, index=False)
                return buf.getvalue()

            st.session_state["_cv_export_csv"]  = _build_cv_csv(counter_configs)
            st.session_state["_cv_export_name"] = (
                f"cctv_queue_{len(counter_configs)}_counters.csv"
            )

# ── REAL DATA MODE ────────────────────────────────────────────────

else:
    uf = st.session_state.get("_uf")

    if uf is not None:
        # ── Step 1: parse CSV (cached by file identity) ──
        file_key = f"{uf.name}_{uf.size}"
        if st.session_state.get("_csv_key") != file_key:
            try:
                df = pd.read_csv(uf)
                configs_raw = parse_csv_raw(df)
                st.session_state["_configs_raw"] = configs_raw
                st.session_state["_csv_key"]     = file_key
                st.session_state["_groq_key"]    = None   # force Groq refresh on new file
            except Exception as e:
                st.exception(e)   # full traceback visible in UI &mdash; no silent failures
                configs_raw = []
        else:
            configs_raw = st.session_state.get("_configs_raw", [])

        if configs_raw:
            # ── Step 2: apply current service_rate (fast, no re-parse) ──
            counter_configs = apply_service_rate(configs_raw, service_rate)

            # ── Step 3: engine functions &mdash; same as simulation mode ──
            readings   = [c["history"][-1] for c in counter_configs if c["history"]]
            wait_times = [
                estimate_wait_time(r["people_count"], counter_configs[i]["service_rate"])
                for i, r in enumerate(readings)
            ]

            preds = []
            for i, c in enumerate(counter_configs):
                pred = predict_future_count(
                    c["history"],
                    horizon_minutes=horizon_min,
                    counter_name=c["name"],
                    threshold=threshold,
                )
                pred["counter_id"] = c["id"]
                preds.append(pred)

            current_summary = [
                {"counter_id": r["counter_id"], "name": r["name"],
                 "people_count": r["people_count"]}
                for r in readings
            ]
            rec = recommend_action(current_summary, preds, wait_times)

            # ── Step 4: Groq alert &mdash; cached by (file + service_rate + action) ──
            groq_cache_key = f"{file_key}_{service_rate}_{rec.get('action')}"
            if st.session_state.get("_groq_key") != groq_cache_key:
                alert_states = [
                    {"name": r["name"], "people_count": r["people_count"],
                     "wait_time": wait_times[i]}
                    for i, r in enumerate(readings)
                ]
                alert_res = generate_alert(alert_states, preds, rec)
                st.session_state["_real_alert"] = alert_res
                st.session_state["_groq_key"]   = groq_cache_key
            else:
                alert_res = st.session_state.get("_real_alert")

            total_rows  = sum(len(c["history"]) for c in counter_configs)
            header_info = (
                f"{len(counter_configs)} counters &middot; {total_rows} readings &middot; "
                f"svc rate {service_rate}/min"
            )
        else:
            header_info = "CSV parse failed &mdash; check column names"
    else:
        header_info = "No file uploaded"


# ─────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────
if is_simulation:
    mode_str = "simulation"
elif is_cv:
    mode_str = "cv"
else:
    mode_str = "real_data"
render_header(mode_str, header_info)

# ── Upload Real Data mode &mdash; uploader renders on EVERY rerun ──────────────────
# CRITICAL: file_uploader MUST NOT be gated behind "if not readings".
# Streamlit only returns the uploaded file while the widget is rendered.
# If we hide the widget once data loads, session_state["_uf"] is cleared on
# the next rerun and the parsed data disappears &#8594; dashboard vanishes.
if not is_simulation and not is_cv:
    if not readings:
        # Empty-state card (shown before first upload)
        st.markdown('''
        <div class="up-placeholder" style="padding-bottom:8px">
          <div class="up-icon">&#128194;</div>
          <div class="up-title">Upload your queue data CSV to get started</div>
          <div class="up-sub">
            Use the <strong>Download Sample CSV</strong> button in the sidebar
            to get a realistic test file &mdash; Counter&nbsp;1 peaks at
            35&nbsp;people and immediately triggers the redirect recommendation
            and Groq alert.
          </div>
        </div>
        ''', unsafe_allow_html=True)

    # Uploader always present so Streamlit keeps returning the file every rerun
    _uploader_result = st.file_uploader(
        "Upload queue data CSV",
        type=["csv"],
        label_visibility="visible",
        help="Required columns: timestamp, counter_name, people_count",
        key="csv_uploader",
    )
    st.session_state["_uf"] = _uploader_result
    if not readings:
        st.caption("`timestamp` &middot; `counter_name` &middot; `people_count`")

# ── Camera/Video mode &mdash; uploader renders on EVERY rerun ──────────────────────
elif is_cv and _CV_AVAILABLE:

    # ── Sub-mode toggle ───────────────────────────────────────────────────────
    _cam_tab1, _cam_tab2 = st.tabs(["📁 Upload Image / Video", "📸 Live Camera Scan"])

    # ════════════════════════════════════════════════════════════
    # TAB 1: Upload (existing behaviour)
    # ════════════════════════════════════════════════════════════
    with _cam_tab1:
        if not readings:
            st.markdown('''
            <div class="up-placeholder" style="padding-bottom:8px">
              <div class="up-icon">&#128247;</div>
              <div class="up-title">Upload footage to count people with AI</div>
              <div class="up-sub">
                One file per counter (JPG / PNG image <em>or</em> MP4 / AVI video).
                YOLOv8n detects people and feeds the count into the queue engine.
                <br><em>Processes uploaded footage &mdash; not a live real-time feed.</em>
              </div>
            </div>
            ''', unsafe_allow_html=True)

        _cv_files = st.file_uploader(
            "Upload image(s) or video(s) &mdash; one per counter",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
            accept_multiple_files=True,
            label_visibility="visible",
            key="cv_uploader",
            help="One file = one counter. Add counter names below after uploading.",
        )
        st.session_state["_cv_files"] = _cv_files or []

        if _cv_files:
            st.markdown("**Name each counter:**")
            cv_labels = []
            cols = st.columns(min(len(_cv_files), 3))
            for i, f in enumerate(_cv_files):
                with cols[i % len(cols)]:
                    lbl = st.text_input(
                        f"File {i+1}: {f.name}",
                        value=f"Counter {i+1}",
                        key=f"cv_label_{i}",
                    )
                    cv_labels.append(lbl)
            st.session_state["_cv_labels"] = cv_labels

    # ════════════════════════════════════════════════════════════
    # TAB 2: Live Camera Scan
    # ════════════════════════════════════════════════════════════
    with _cam_tab2:
        from datetime import datetime as _dt_live
        import io as _io_live

        st.markdown('''
        <div style="
          background:linear-gradient(135deg,rgba(11,114,133,.08),rgba(20,184,166,.04));
          border:1.5px solid rgba(11,114,133,.20);
          border-radius:12px; padding:16px 20px; margin-bottom:16px;
        ">
          <div style="font-size:1rem;font-weight:700;color:#0B7285;margin-bottom:4px;">
            &#128248; Live Camera People Counter
          </div>
          <div style="font-size:.82rem;color:#3D4A5C;line-height:1.6;">
            Click <strong>Take Photo</strong> to capture a frame from your camera.
            AI instantly counts people and records the data.<br>
            Capture multiple frames to build a time-series &mdash; then export as CSV.
          </div>
        </div>
        ''', unsafe_allow_html=True)

        # Counter name for live camera
        _live_ctr_name = st.text_input(
            "Counter / Camera name",
            value=st.session_state.get("_live_ctr_name", "Live Camera"),
            key="_live_ctr_name_input",
            placeholder="e.g. Main Hall Counter 1",
        )
        st.session_state["_live_ctr_name"] = _live_ctr_name

        # Live camera capture widget
        _cam_img = st.camera_input(
            "📷 Point camera at queue &#8594; Take Photo",
            key="live_cam_input",
            help="Allow browser camera access when prompted.",
        )

        _live_col1, _live_col2 = st.columns([1, 1])
        with _live_col1:
            _process_btn = st.button(
                "🔍 Detect People & Record",
                key="live_detect_btn",
                use_container_width=True,
                disabled=(_cam_img is None),
            )
        with _live_col2:
            _clear_btn = st.button(
                "🗑 Clear History",
                key="live_clear_btn",
                use_container_width=True,
            )

        if _clear_btn:
            st.session_state["_live_history"] = []
            st.session_state["_live_export_csv"] = None
            st.toast("History cleared.", icon="🗑")

        # ── Process captured frame with YOLO ──────────────────────
        if _process_btn and _cam_img is not None:
            try:
                _cam_img.seek(0)
                _cam_bytes = _cam_img.read()

                # Wrap as a UploadedFile-like object so _cvd can process it
                import tempfile, os as _os
                _tmp_path = None
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as _tmp:
                    _tmp.write(_cam_bytes)
                    _tmp_path = _tmp.name

                # Build a minimal file-like wrapper
                class _FakePseudoFile:
                    def __init__(self, path, name):
                        self.name = name
                        self._path = path
                        self._data = open(path, "rb").read()
                        self._pos = 0
                        self.size = len(self._data)
                    def read(self, n=-1):
                        if n == -1:
                            d = self._data[self._pos:]
                            self._pos = len(self._data)
                        else:
                            d = self._data[self._pos:self._pos+n]
                            self._pos += n
                        return d
                    def seek(self, pos):
                        self._pos = pos

                _fake_file = _FakePseudoFile(_tmp_path, f"{_live_ctr_name}.jpg")

                yolo_model = _get_yolo_model()
                with st.spinner("Running YOLOv8 person detection..."):
                    _live_configs = _cvd.build_cv_counter_configs(
                        [_fake_file], [_live_ctr_name],
                        st.session_state.get("cv_service_rate", 1.0),
                        model=yolo_model
                    )

                if _tmp_path and _os.path.exists(_tmp_path):
                    _os.unlink(_tmp_path)

                if _live_configs and _live_configs[0].get("history"):
                    _detected_count = _live_configs[0]["history"][-1]["people_count"]
                    _ts = _dt_live.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Append to running history
                    _hist = st.session_state.get("_live_history", [])
                    _hist.append({
                        "timestamp":    _ts,
                        "counter_name": _live_ctr_name,
                        "people_count": _detected_count,
                    })
                    st.session_state["_live_history"] = _hist

                    # Build export CSV
                    _buf = _io_live.BytesIO()
                    pd.DataFrame(_hist).to_csv(_buf, index=False)
                    st.session_state["_live_export_csv"] = _buf.getvalue()

                    # Show instant result
                    st.success(
                        f"✅ **{_detected_count} people** detected at {_ts}"
                    )

                    # Feed into session so dashboard can render
                    st.session_state["_cv_configs"] = _live_configs
                    st.session_state["_cv_cache_key"] = f"live_{_ts}"
                    st.session_state["_cv_groq_key"]  = None
                else:
                    st.warning(
                        "No people detected in frame &mdash; try better lighting or a clearer view."
                    )
            except Exception as _live_err:
                st.error(f"Detection error: {_live_err}")

        # ── Live history table ────────────────────────────────────
        _live_hist = st.session_state.get("_live_history", [])
        if _live_hist:
            st.markdown("---")
            st.markdown(f"**📊 Live Session History &mdash; {len(_live_hist)} reading(s)**")

            _hist_df = pd.DataFrame(_live_hist)
            st.dataframe(_hist_df, use_container_width=True, height=200)

            # Metrics row
            _m1, _m2, _m3 = st.columns(3)
            with _m1:
                st.metric("Latest Count", _live_hist[-1]["people_count"])
            with _m2:
                st.metric("Peak Count", max(r["people_count"] for r in _live_hist))
            with _m3:
                st.metric(
                    "Avg Count",
                    f"{sum(r['people_count'] for r in _live_hist)/len(_live_hist):.1f}"
                )

            # Export CSV
            st.markdown("---")
            _lexp_c1, _lexp_c2 = st.columns([2, 3])
            with _lexp_c1:
                st.download_button(
                    label="📥 Export Live Data CSV",
                    data=st.session_state["_live_export_csv"],
                    file_name=f"live_camera_{_live_ctr_name.replace(' ','_')}.csv",
                    mime="text/csv",
                    key="live_csv_export_btn",
                    use_container_width=True,
                    help="Upload this to 📂 Upload Real Data for full analysis",
                )
            with _lexp_c2:
                st.caption(
                    "💡 **Tip:** Download this CSV &#8594; switch to **📂 Upload Real Data** "
                    "&#8594; upload the file for full AI analysis, predictions, and alerts."
                )


# ── Simulation mode &mdash; show placeholder or dashboard ──────────────────────────
elif is_simulation and not readings:
    st.info("Advance the first tick to begin the simulation.")

# ── Dashboard &mdash; renders for ALL modes once readings are populated ─────────────
if readings:
    _sr = service_rate if not is_simulation else (
        st.session_state["counters_state"][0]["service_rate"]
        if st.session_state.get("counters_state") else 1.0
    )
    render_full_dashboard(
        readings, wait_times, preds, counter_configs, rec, alert_res, horizon_min,
        threshold=threshold,
        preds_multi=preds_multi if preds_multi else None,
        service_rate=_sr,
    )

    # ── CCTV CSV EXPORT PANEL ─────────────────────────────────────────────────
    if is_cv and st.session_state.get("_cv_export_csv"):
        st.markdown("---")
        st.markdown('''
        <div style="
          background:linear-gradient(135deg,rgba(11,114,133,.10),rgba(20,184,166,.06));
          border:1.5px solid rgba(11,114,133,.25);
          border-radius:14px; padding:20px 24px; margin-top:4px;
        ">
          <div style="font-size:1.05rem;font-weight:700;color:#0B7285;margin-bottom:4px;">
            &#128228; Export Detected Data as CSV
          </div>
          <div style="font-size:.84rem;color:#3D4A5C;line-height:1.6;">
            Download the AI-detected queue data in CSV format.<br>
            You can then upload it directly to
            <strong>&#128194; Upload Real Data</strong> mode for further analysis,
            historical tracking, or custom visualisations.
          </div>
        </div>
        ''', unsafe_allow_html=True)

        _exp_col1, _exp_col2, _exp_col3 = st.columns([2, 2, 1])
        with _exp_col1:
            st.download_button(
                label="📥  Download CSV",
                data=st.session_state["_cv_export_csv"],
                file_name=st.session_state.get("_cv_export_name", "cctv_queue_data.csv"),
                mime="text/csv",
                key="cv_csv_export_btn",
                use_container_width=True,
                help="CSV columns: timestamp, counter_name, people_count",
            )
        with _exp_col2:
            # Preview what's in the CSV
            import io as _prev_io
            _prev_df = pd.read_csv(
                _prev_io.BytesIO(st.session_state["_cv_export_csv"])
            )
            st.caption(
                f"📋 **{len(_prev_df)} rows** &middot; "
                f"**{_prev_df['counter_name'].nunique()} counter(s)** &middot; "
                f"Columns: `timestamp`, `counter_name`, `people_count`"
            )
        with _exp_col3:
            if st.button("👁 Preview", key="cv_csv_preview_btn", use_container_width=True):
                st.session_state["_show_csv_preview"] = not st.session_state.get("_show_csv_preview", False)

        if st.session_state.get("_show_csv_preview"):
            import io as _prev_io2
            _pv_df = pd.read_csv(_prev_io2.BytesIO(st.session_state["_cv_export_csv"]))
            st.dataframe(_pv_df.head(20), use_container_width=True)
            st.caption("Showing first 20 rows. Download the full file above.")


# ─────────────────────────────────────────────────────────
# AUTO-PLAY  (simulation mode only)
# ─────────────────────────────────────────────────────────
if is_simulation and st.session_state.get("running"):
    time.sleep(tick_speed)
    st.rerun()

