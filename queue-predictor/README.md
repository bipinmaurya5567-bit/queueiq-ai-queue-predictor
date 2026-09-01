<div align="center">

<h1>📊 QueueIQ — AI-Powered Queue Predictor</h1>

<p><strong>Real-time crowd intelligence · Predictive analytics · Smart staffing decisions</strong></p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Person%20Detection-0052CC?style=for-the-badge&logoColor=white)](https://ultralytics.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F55036?style=for-the-badge)](https://console.groq.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Forecasting-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

> **QueueIQ** is an intelligent queue management dashboard that predicts crowd build-up, classifies congestion risk, and recommends staffing actions — in real time — using computer vision, queueing theory, and a large language model.

</div>

---

## 🧠 What Problem Does It Solve?

Service centers — banks, hospitals, railway stations, college offices — routinely suffer from reactive queue management: staff only act *after* lines explode. **QueueIQ flips this.** It ingests live or simulated counter data, predicts where queues are heading over the next 5–20 minutes, and surfaces AI-generated alerts *before* the situation becomes critical.

---

## ✨ Key Features

| # | Feature | Detail |
|---|---|---|
| 1 | 🔁 **Simulation Mode** | Poisson-process queue simulation across 4 real-world presets — Bank, Hospital, College Office, Railway Station |
| 2 | 📂 **CSV Upload Mode** | Upload historical counter data (`timestamp, counter_name, people_count`) for full AI analysis |
| 3 | 📹 **AI CCTV Analysis** | Drag-and-drop images or video — YOLOv8n auto-detects and counts every person in frame |
| 4 | 📸 **Live Camera Scan** | Real-time webcam capture with instant headcount |
| 5 | 🧠 **Multi-Horizon Forecasting** | Linear regression predicts queue size at 5, 10, 15, and 20 minutes ahead |
| 6 | ⚠️ **4-Level Risk Engine** | Per-counter & facility-level risk scoring: **Low → Medium → High → Critical** |
| 7 | 🚨 **Groq LLM Alerts** | LLaMA-3.3-70B generates one-sentence natural-language alerts for operations managers |
| 8 | 🔀 **What-If Scenarios** | Simulate *"What if I open a new counter?"*, *"What if I redirect 40% of customers?"*, or *"What if I speed up service?"* |
| 9 | 📊 **M/M/1 Queue Theory** | Full queueing-theory wait-time calculation (ρ = λ/μ, Lq, Wq) alongside empirical estimates |
| 10 | 📤 **CSV Export** | Download any detected or simulated dataset for offline audit |
| 11 | 🌙 **Dark / ☀️ Light Mode** | One-click theme toggle in sidebar |

---

## 🏗️ Project Structure

```
queue-predictor/
│
├── app.py                      ← Streamlit dashboard entry point (all 3 modes)
├── config.py                   ← ⚙️  Central config — ALL thresholds & tuning knobs here
│
├── simulator.py                ← Poisson-based queue counter simulation engine
├── queue_math.py               ← M/M/1 queueing theory (ρ, Lq, Wq formulas)
├── predictor.py                ← Linear regression multi-horizon forecasting
├── recommender.py              ← Action recommendation engine (open/redirect/speed-up)
├── groq_alerts.py              ← Groq API → LLaMA-3.3-70B alert generation
├── cv_detector.py              ← YOLOv8n person detection (image / video / webcam)
├── risk_engine.py              ← Risk classifier: per-counter + facility aggregation
├── what_if.py                  ← What-if scenario simulation engine
│
├── tests/
│   └── test_all.py             ← pytest suite: math, predictions, risk, scenarios
│
├── requirements.txt            ← Python dependencies
├── .env.example                ← Environment variable template
├── .gitignore
└── README.md
```

### ⚙️ Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│              DATA SOURCES (choose one per session)          │
│   📹 Camera/Video Feed  │  📂 CSV Upload  │  🔁 Simulation  │
└──────────────────────┬──────────────────────────────────────┘
                       │ people_count per counter per tick
                       ▼
         ┌─────────────────────────────┐
         │       queue_math.py         │  M/M/1 wait time (Wq = λ/μ(μ−λ))
         │       predictor.py          │  5/10/15/20 min linear forecasts
         │       risk_engine.py        │  LOW | MEDIUM | HIGH | CRITICAL
         │       recommender.py        │  "Open Counter 3" / "Redirect..."
         │       groq_alerts.py        │  LLM one-sentence alert narrative
         └─────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    app.py       │  Streamlit real-time dashboard
              └─────────────────┘
```

---

## 📦 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI / Dashboard** | [Streamlit 1.35+](https://streamlit.io) | Interactive web app, no front-end code needed |
| **Computer Vision** | [YOLOv8n](https://ultralytics.com) + [OpenCV](https://opencv.org) | Real-time person detection in images & video |
| **LLM Alerts** | [Groq API](https://groq.com) · LLaMA-3.3-70B | Natural-language alert generation (< 150 tokens) |
| **ML Forecasting** | [scikit-learn](https://scikit-learn.org) | Linear regression for queue length prediction |
| **Queue Theory** | Custom (`queue_math.py`) | M/M/1 formulas: ρ, Lq, Wq |
| **Data Processing** | [pandas](https://pandas.pydata.org) · [NumPy](https://numpy.org) | Data wrangling, rolling windows |
| **Image Handling** | [Pillow](https://pillow.readthedocs.io) | Frame extraction and preprocessing |
| **Testing** | [pytest](https://pytest.org) | Automated unit & integration tests |
| **Env Config** | [python-dotenv](https://pypi.org/project/python-dotenv/) | Secure API key loading |

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.10+** installed
- A free **Groq API key** → [console.groq.com/keys](https://console.groq.com/keys)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/bipinmaurya5567-bit/queueiq-ai-queue-predictor.git
cd queueiq-ai-queue-predictor/queue-predictor
```

---

### Step 2 — Create & activate a virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** YOLOv8n model weights (`yolov8n.pt`, ~6 MB) are downloaded automatically by Ultralytics on first use. No manual download required.

---

### Step 4 — Configure your API key

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and replace the placeholder:

```env
# .env
GROQ_API_KEY=gsk_your_actual_key_here
```

Get your free key at **[console.groq.com/keys](https://console.groq.com/keys)** — no credit card required.

---

## ▶️ How to Run the App

```bash
streamlit run app.py
```

Then open **[http://localhost:8501](http://localhost:8501)** in your browser.

You will see three operating modes in the sidebar:

| Mode | When to use |
|---|---|
| 🔁 **Simulation** | Demo / testing — no data or camera needed |
| 📂 **Upload Real Data** | You have historical CSV exports from your counters |
| 📹 **AI CCTV Analysis** | You have images, video clips, or a live webcam feed |

---

## 📋 CSV Format Reference

When using **Upload Real Data** mode, your CSV must contain these columns:

```csv
timestamp,counter_name,people_count
2024-01-15 09:00:00,Counter 1 (Savings),12
2024-01-15 09:00:00,Counter 2 (Current),7
2024-01-15 09:05:00,Counter 1 (Savings),16
2024-01-15 09:05:00,Counter 2 (Current),4
```

| Column | Type | Notes |
|---|---|---|
| `timestamp` | `datetime` | Any pandas-parseable format (`YYYY-MM-DD HH:MM:SS`) |
| `counter_name` | `str` | Unique name per service window |
| `people_count` | `int >= 0` | Snapshot headcount at that moment |

---

## 🧪 Running Tests

```bash
# From inside queue-predictor/
python -m pytest tests/ -v
```

Or run directly (no pytest needed):

```bash
python tests/test_all.py
```

**Test coverage includes:**

- ✅ M/M/1 queue wait time calculations (stable & unstable systems)
- ✅ Multi-horizon prediction pipeline
- ✅ Risk classification — per-counter and facility-level aggregation
- ✅ Recommendation engine logic (open counter, redirect, speed-up)
- ✅ What-if scenario arithmetic
- ✅ Edge cases — zero queue, missing data, division guards

---

## ⚙️ Configuration Reference

All system thresholds live in `config.py`. Edit this file to tune QueueIQ to your venue **without touching any business logic**:

```python
# ── Queue thresholds (people count) ─────────────────────────
THRESHOLD_LOW      = 10    # Normal operation
THRESHOLD_MEDIUM   = 20    # Building up — monitor
THRESHOLD_HIGH     = 30    # Alert threshold (also configurable via UI)
THRESHOLD_CRITICAL = 45    # Override — always triggers Critical

# ── Wait time thresholds (minutes) ──────────────────────────
WAIT_LOW      =  8
WAIT_MEDIUM   = 15
WAIT_HIGH     = 25
WAIT_CRITICAL = 40

# ── Server utilisation rho = lambda/mu ──────────────────────
UTILIZATION_HIGH     = 0.85   # System stressed
UTILIZATION_CRITICAL = 0.95   # Near-saturation

# ── Forecast horizons ────────────────────────────────────────
FORECAST_HORIZONS = [5, 10, 15, 20]   # minutes ahead

# ── Groq LLM ────────────────────────────────────────────────
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS  = 150
GROQ_TIMEOUT_SEC = 6
```

---

## 🗺️ Roadmap

- [ ] 🔌 REST API layer — expose predictions for external dashboards
- [ ] 🐳 Docker + `docker-compose` one-command deployment
- [ ] 📈 Historical trend charts (day-over-day, week-over-week)
- [ ] 📍 Multi-branch / multi-location support
- [ ] 📱 SMS / email / WhatsApp alerts via Twilio
- [ ] 🔄 WebSocket live counter updates (no manual page refresh)
- [ ] 🗃️ Database persistence (SQLite / PostgreSQL) for audit history
- [ ] ☁️ Streamlit Cloud one-click deploy button

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. **Fork** the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "feat: add amazing feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a **Pull Request**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Bipin Maurya**

[![GitHub](https://img.shields.io/badge/GitHub-bipinmaurya5567--bit-181717?style=flat-square&logo=github)](https://github.com/bipinmaurya5567-bit)

---

<div align="center">

**⭐ If this project helped you, please give it a star — it means a lot!**

</div>
