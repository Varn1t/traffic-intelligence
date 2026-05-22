# 🚦 TraffiQ — AI-Powered Traffic Analysis System

A professional-grade real-time traffic analysis system built with **YOLOv8**, **ByteTrack**, and **Flask**. Detects and tracks vehicles across user-defined lanes, computes per-lane statistics, flags incidents, measures speeds, and displays everything in a live web dashboard with a purple-black gradient UI.

Featuring a premium, two-way interactive **Control Panel** allowing live parameter configuration without restarting the system.

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚙️ Interactive Control Panel | Collapsible glassmorphic sidebar for live web-based parameter configuration |
| 🎯 Multi-Object Tracking | ByteTrack algorithm with unique vehicle IDs |
| ⚡ Lane-Level Analytics | Per-lane counts, Level of Service (LOS) A–F grading, flow rate, queue length |
| 📷 Speed Camera | Pixel-to-meter calibrated speed estimation + dynamic speeding cameras + CSV logging |
| ⚠️ Incident Detection | Dynamic threshold incident tracking for stopped vehicles (>2s to 30s) |
| 🚨 Emergency Priority | Auto signal timer extension for emergency vehicles |
| 📊 Live Chart History | 2-minute rolling bar/line chart (Chart.js) |
| 🔥 Heatmap Overlay | Accumulative density heatmap via OpenCV |
| 📈 Trend Prediction | Rolling linear regression per lane |
| 💾 CSV Data Logging | Every detection and violation exported for offline analysis |
| 🔐 Auth-Protected Dashboard | Login-gated live dashboard with session management |

---

## ⚙️ Interactive Control Panel & Sidebar

TraffiQ transitions from a passive monitor to an active control center via its dynamic two-way **Control Panel**:
* **Collapsible Sidebar:** Open and close the control console via the hovering gear (`⚙️`) button or `Escape` key. Styled with blur backdrops and neon-purple glow themes.
* **Visualization Mode Toggles:** Switch between active processing modes (`Lanes`, `Heatmap`, `Speed`, `Timer`) dynamically. The underlying OpenCV pipeline renders the corresponding overlay instantly.
* **Dynamic Speed Trigger Limit:** Adjust speeding camera threshold (20 - 120 km/h) on the fly via slider control. Bounding boxes highlight speeding vehicles instantly.
* **Incident Timeout Configuration:** Slide the stopped vehicle detection threshold (2s - 30s) dynamically.
* **YOLO Confidence Slider:** Slide detection threshold (0.10 - 0.95) to filter out background noise or increase object detection density instantly.
* **Intelligent Cooldown Lockouts:** Features a robust 5-second interaction cooldown protection. It displays an amber-glass warning banner reading `Please wait 5 seconds before you change` when inputs are spammed, and automatically reverts sliders/modes to verified backend states on failure.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Varn1t/Traffic-Intelligence.git
cd "Traffic Intelligence"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your credentials
Copy the example env file and fill in your own values:
```bash
cp .env.example .env
```
Edit `.env`:
```
DASH_USER=your_username
DASH_PASS=your_password
```

### 4. Add your video file
Place your traffic video in the project folder and update `VIDEO_PATH` in `config.py`:
```python
VIDEO_PATH = "Traffic_Video.mp4"
```

### 5. Run the System
```bash
python main.py
```

1. **ROI Selector Box:** The application opens an interactive OpenCV selection window. Use your mouse to draw rectangular boundaries for each lane.
2. **Press ENTER** when lanes are set.
3. The YOLOv8 model runs in a background thread, starts local camera playback, and spins up the live dashboard.
4. Open **http://localhost:5050** in your browser, log in, and control the system dynamically.

---

## 🛠️ Tech Stack

- **Python** — core logic
- **YOLOv8** (Ultralytics) — vehicle detection
- **ByteTrack** (Supervision) — multi-object tracking
- **OpenCV** — video processing & heatmap overlays
- **Flask** — web server & REST API
- **Chart.js** — live dashboard charts
- **scikit-learn** — Random Forest traffic prediction
- **Vanilla CSS** — premium glassmorphism styling and animations

---

## 📁 Project Structure

```
traffiq/
├── main.py             # Main entry point & video processing loop
├── config.py           # Configuration variables and thresholds
├── state.py            # Thread-safe shared state & stats tracking
├── loggers.py          # Logging classes (CSV, SpeedCamera)
├── analytics.py        # Core logic: ML predictor, incident detection, flow rates
├── templates.py        # UI: Dashboard and landing page HTML
├── web_app.py          # Flask application logic
├── requirements.txt    # Python dependencies
├── .env.example        # Credential template (copy → .env)
├── .gitignore
└── README.md
```

> **Note:** Your `.env`, video files (`*.mp4`), model weights (`*.pt`), and CSV logs are excluded from the repo via `.gitignore`.

---

## 📄 License

All rights reserved by Varn1t

---

*Built by Varnit*
