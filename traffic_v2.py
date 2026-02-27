"""
Traffic Analysis System v2
==========================
Resume-highlight features added:
  1. ByteTrack multi-object tracking  (via supervision)
  2. Flask live dashboard             (real-time stats in browser)
  3. Incident detection               (stopped-vehicle alerts)
  4. CSV data logging                 (analytics-ready output)
  5. Speed calibration                (pixel-to-meter via reference line)

Run:   python traffic_v2.py
Dash:  http://localhost:5050
Deps:  pip install ultralytics supervision flask opencv-python numpy
"""

import cv2
import numpy as np
import time
import csv
import os
import threading
import webbrowser
from math import hypot
from datetime import datetime
from collections import defaultdict, deque

from ultralytics import YOLO
import supervision as sv
from flask import Flask, jsonify, render_template_string, session, request, redirect, url_for
from dotenv import load_dotenv

load_dotenv()  # loads credentials from .env file

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
VIDEO_PATH          = "Traffic_Video.mp4"
MODEL_NAME          = "yolov8n.pt"
CONF_THRESHOLD      = 0.40          # detection confidence
TRACK_CLASSES       = {"car", "bus", "truck", "motorbike"}
INCIDENT_TIMEOUT    = 5.0           # seconds a vehicle must stay still → incident
INCIDENT_DIST_PX    = 15            # pixels of movement tolerance
LOG_FILE            = "traffic_log.csv"
FLASK_PORT          = 5050
PIXEL_TO_METER      = 0.05          # ← change after calibration
SPEED_LIMIT_KMPH    = 50            # vehicles above this are flagged by speed camera
SPEEDER_LOG_FILE    = "speeders_log.csv"
EMERGENCY_CLASSES   = {"bus", "truck"}  # large vehicle proxy for ambulance/fire truck
EMERGENCY_SPEED_KMH = 40            # fast large vehicle = likely emergency vehicle

# ──────────────────────────────────────────────
# SHARED STATE  (Flask ↔ OpenCV thread-safe)
# ──────────────────────────────────────────────
shared_state = {
    "lane_counts": {},          # {lane_id: {class: count}}
    "vehicle_count": 0,
    "incidents": [],            # list of active incident dicts
    "mode": "lanes",
    "fps": 0.0,
    "frame_id": 0,
    "speed_data": {},           # {track_id: speed_kmh}
    "speeders": [],             # recent speeding events
    "emergency_active": False,
    "emergency_lane": None,
    "lane_trends":   {},   # {lane_id_str: "↑"/"↓"/"→"}
    "lane_los":      {},   # {lane_id_str: "A".."F"}
    "lane_flow":     {},   # {lane_id_str: veh/min}
    "lane_queue":    {},   # {lane_id_str: stopped count at RED}
    "wrong_way":     [],   # list of track_ids flagged this frame
    "tailgating":    [],   # list of {id_a, id_b, lane} this frame
}
state_lock = threading.Lock()

# ──────────────────────────────────────────────
# SPEED CAMERA LOGGER
# ──────────────────────────────────────────────
class SpeedCameraLogger:
    """Logs speeding vehicles once per overspeed event (not every frame)."""
    def __init__(self, path: str):
        self.path = path
        self.logged_ids = {}   # track_id → last logged speed bucket
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "frame_id", "track_id",
                                     "lane_id", "speed_kmh", "class"])

    def log(self, frame_id: int, track_id: int, lane_id, speed_kmh: float, label: str):
        bucket = int(speed_kmh // 10)  # only re-log if speed changes by 10 km/h
        if self.logged_ids.get(track_id) == bucket:
            return None
        self.logged_ids[track_id] = bucket
        ts = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([ts, frame_id, track_id, lane_id,
                                     round(speed_kmh, 1), label])
        return {"timestamp": ts, "track_id": track_id, "lane": lane_id,
                "speed_kmh": round(speed_kmh, 1), "class": label}

# ──────────────────────────────────────────────
# CSV LOGGER
# ──────────────────────────────────────────────
class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self._write_header()

    def _write_header(self):
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "frame_id", "lane_id",
                         "cars", "buses", "trucks", "motorbikes", "total", "incident"])

    def log(self, frame_id: int, lane_counts: dict, incidents: list):
        ts = datetime.now().isoformat(timespec="seconds")
        incident_lanes = {inc["lane"] for inc in incidents}
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            for lane_id, counts in lane_counts.items():
                w.writerow([
                    ts, frame_id, lane_id,
                    counts.get("car", 0),
                    counts.get("bus", 0),
                    counts.get("truck", 0),
                    counts.get("motorbike", 0),
                    sum(counts.values()),
                    int(lane_id in incident_lanes),
                ])

# ──────────────────────────────────────────────
# INCIDENT DETECTOR
# ──────────────────────────────────────────────
class IncidentDetector:
    """Flags vehicles that haven't moved for INCIDENT_TIMEOUT seconds."""
    def __init__(self):
        self.history = {}   # track_id → {"pos": (x,y), "still_since": float}

    def update(self, track_id: int, cx: int, cy: int, lane_id: int) -> bool:
        now = time.time()
        if track_id not in self.history:
            self.history[track_id] = {"pos": (cx, cy), "still_since": now, "lane": lane_id}
            return False
        prev = self.history[track_id]
        dist = hypot(cx - prev["pos"][0], cy - prev["pos"][1])
        if dist > INCIDENT_DIST_PX:
            # vehicle moved — reset timer
            self.history[track_id] = {"pos": (cx, cy), "still_since": now, "lane": lane_id}
            return False
        else:
            still_for = now - prev["still_since"]
            return still_for >= INCIDENT_TIMEOUT

    def cleanup(self, active_ids: set):
        self.history = {k: v for k, v in self.history.items() if k in active_ids}

# ──────────────────────────────────────────────
# LANE TREND TRACKER  (rule-based predictive optimisation)
# ──────────────────────────────────────────────
class LaneTrendTracker:
    """Rolling linear-regression slope per lane — purely maths, no ML.

    update() ingests one sample per frame.
    trend()  returns the slope (vehicles/sample):  +ve = rising, -ve = falling.
    label()  returns a display arrow: ↑  ↓  →
    """
    WINDOW = 20      # samples  (~10 s at typical processing speeds)
    THRESHOLD = 0.15 # slope magnitude to call a trend definite

    def __init__(self):
        self.history: dict = defaultdict(lambda: deque(maxlen=self.WINDOW))

    def update(self, lane_id: int, count: int):
        self.history[lane_id].append(count)

    def trend(self, lane_id: int) -> float:
        """Linear regression slope over the rolling window."""
        h = list(self.history[lane_id])
        n = len(h)
        if n < 3:
            return 0.0
        xs = list(range(n))
        mx, my = sum(xs) / n, sum(h) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, h))
        den = sum((x - mx) ** 2 for x in xs) or 1e-9
        return num / den

    def label(self, lane_id: int) -> str:
        """Unicode arrow — for the HTML dashboard."""
        s = self.trend(lane_id)
        if s >  self.THRESHOLD: return "\u2191"   # ↑
        if s < -self.THRESHOLD: return "\u2193"   # ↓
        return "\u2192"                           # →

    def label_ascii(self, lane_id: int) -> str:
        """ASCII arrow — for OpenCV putText which can't render Unicode."""
        s = self.trend(lane_id)
        if s >  self.THRESHOLD: return "^"
        if s < -self.THRESHOLD: return "v"
        return "-"

# ──────────────────────────────────────────────
# FLOW RATE TRACKER
# ──────────────────────────────────────────────
class FlowRateTracker:
    """Counts unique vehicle IDs entering each lane per sliding 60-second window."""
    WINDOW = 60.0   # seconds

    def __init__(self):
        # lane_id → deque of (track_id, timestamp)
        self.log: dict = defaultdict(lambda: deque())

    def record(self, lane_id: int, track_id: int):
        now = time.time()
        self.log[lane_id].append((track_id, now))

    def rate(self, lane_id: int) -> float:
        """Vehicles per minute for this lane over the last 60 s."""
        now = time.time()
        cutoff = now - self.WINDOW
        buf = self.log[lane_id]
        # drop old entries
        while buf and buf[0][1] < cutoff:
            buf.popleft()
        unique_ids = len({tid for tid, _ in buf})
        return round(unique_ids / (self.WINDOW / 60), 1)   # per-minute rate

# ──────────────────────────────────────────────
# LOS GRADE  (Highway Capacity Manual simplified)
# ──────────────────────────────────────────────
def los_grade(vehicle_count: int) -> tuple:
    """Return (grade, colour_hex, description) for a lane vehicle count."""
    if vehicle_count <= 3:  return ("A", "#4ade80",  "Free flow")
    if vehicle_count <= 6:  return ("B", "#a3e635",  "Reasonable free flow")
    if vehicle_count <= 10: return ("C", "#facc15",  "Stable flow")
    if vehicle_count <= 15: return ("D", "#fb923c",  "Approaching unstable")
    if vehicle_count <= 22: return ("E", "#f87171",  "Unstable flow")
    return                         ("F", "#dc2626",  "Forced / breakdown")

# ──────────────────────────────────────────────
# SESSION STATS  (for summary on quit)
# ──────────────────────────────────────────────
_ss_all_ids:       set  = set()
_ss_peak_count:    int  = 0
_ss_peak_time:     str  = ""
_ss_total_inc:     int  = 0
_ss_wrong_ids:     set  = set()
_ss_tailgate:      int  = 0
_ss_start:         str  = datetime.now().isoformat(timespec="seconds")

# Convenience namespace so callers use session_stats["key"] syntax
class _SessionStats:
    @property
    def all_ids(self):         return _ss_all_ids
    @property
    def peak_count(self):      return _ss_peak_count
    @peak_count.setter
    def peak_count(self, v):   global _ss_peak_count; _ss_peak_count = v
    @property
    def peak_time(self):       return _ss_peak_time
    @peak_time.setter
    def peak_time(self, v):    global _ss_peak_time; _ss_peak_time = v
    @property
    def total_incidents(self): return _ss_total_inc
    @total_incidents.setter
    def total_incidents(self, v): global _ss_total_inc; _ss_total_inc = v
    @property
    def wrong_way_ids(self):   return _ss_wrong_ids
    @property
    def tailgate_events(self): return _ss_tailgate
    @tailgate_events.setter
    def tailgate_events(self, v): global _ss_tailgate; _ss_tailgate = v
    @property
    def session_start(self):   return _ss_start
    def __getitem__(self, k):  return getattr(self, k)
    def __setitem__(self, k, v): setattr(self, k, v)

session_stats = _SessionStats()


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Traffic Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root{
      --bg:#08060e;--surface:#110e1a;--surface2:#181326;--border:#2a1f4a;
      --accent:#a855f7;--accent2:#7c3aed;--green:#22c55e;--red:#ef4444;
      --yellow:#f59e0b;--cyan:#c084fc;--text:#e2e8f0;--muted:#8b7fb0;
      --glow:rgba(168,85,247,0.15);
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}

    /* ── ANIMATED BACKGROUND ── */
    body::before{content:'';position:fixed;inset:0;
      background:radial-gradient(ellipse 80% 50% at 50% -20%,#2d1b6955,transparent),
                 radial-gradient(ellipse 50% 40% at 80% 110%,#4c1d9544,transparent),
                 radial-gradient(ellipse 30% 25% at 10% 60%,#7c3aed18,transparent);
      pointer-events:none;z-index:0}

    .wrap{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:28px 24px}

    /* ── HEADER ── */
    header{display:flex;align-items:center;justify-content:space-between;margin-bottom:32px;flex-wrap:wrap;gap:12px}
    .brand{display:flex;align-items:center;gap:14px}
    .brand-icon{width:46px;height:46px;border-radius:14px;
      background:linear-gradient(135deg,#a855f7,#7c3aed);
      display:flex;align-items:center;justify-content:center;font-size:1.4rem;
      box-shadow:0 0 30px #a855f740}
    .brand h1{font-size:1.45rem;font-weight:800;letter-spacing:-.02em;color:#f1f5f9}
    .brand p{font-size:.78rem;color:var(--muted);margin-top:2px}

    .header-controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
    .live-badge{display:flex;align-items:center;gap:7px;
      background:#0d1225;border:1px solid var(--border);border-radius:100px;
      padding:6px 14px;font-size:.75rem;font-weight:600;color:var(--green)}
    .live-dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse-dot 1.4s ease-in-out infinite}
    @keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.7)}}
    .badge-paused{border-color:#f59e0b40!important;color:#fcd34d!important}
    .badge-offline{border-color:#ef444440!important;color:#fca5a5!important;animation:pulse-offline 2s ease-in-out infinite}
    @keyframes pulse-offline{0%,100%{box-shadow:0 0 8px #ef444420}50%{box-shadow:0 0 16px #ef444440}}

    .btn{display:flex;align-items:center;gap:6px;padding:8px 16px;border-radius:10px;
      font-family:'Inter',sans-serif;font-size:.78rem;font-weight:600;cursor:pointer;
      border:1px solid var(--border);background:var(--surface2);color:var(--text);
      transition:all .18s;white-space:nowrap}
    .btn:hover{border-color:var(--accent);color:var(--accent);background:#1a0f30}
    .btn.active{border-color:var(--accent);background:#2d1b69;color:var(--accent)}
    .btn.danger{border-color:#ef444440;color:var(--red)}
    .btn.danger:hover{background:#2d0c0c;border-color:var(--red)}
    .btn-export{border-color:#22c55e40;color:var(--green)}
    .btn-export:hover{background:#0d2d1a;border-color:var(--green)}

    /* ── KPI GRID ── */
    .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:24px}
    .kpi{background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);border-radius:18px;
      padding:20px;position:relative;overflow:hidden;transition:all .3s ease;
      backdrop-filter:blur(8px)}
    .kpi:hover{border-color:var(--accent);transform:translateY(-4px);box-shadow:0 8px 30px rgba(168,85,247,0.15)}
    .kpi::before{content:'';position:absolute;inset:0;border-radius:18px;opacity:0;
      background:linear-gradient(135deg,rgba(168,85,247,0.08),rgba(124,58,237,0.05));transition:opacity .3s}
    .kpi:hover::before{opacity:1}
    .kpi-icon{font-size:1.3rem;margin-bottom:10px;opacity:.8}
    .kpi-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:600;margin-bottom:6px}
    .kpi-val{font-size:2rem;font-weight:800;line-height:1;letter-spacing:-.04em;transition:color .3s}
    .kpi-val.blue{color:var(--accent)}
    .kpi-val.green{color:var(--green)}
    .kpi-val.red{color:var(--red)}
    .kpi-val.yellow{color:var(--yellow)}
    .kpi-val.cyan{color:var(--cyan)}
    .kpi-sub{font-size:.7rem;color:var(--muted);margin-top:5px}
    .kpi-bar{position:absolute;bottom:0;left:0;height:3px;border-radius:0 0 18px 18px;
      background:linear-gradient(90deg,#a855f7,#7c3aed);width:60%;transition:width .6s}

    /* ── ALERTS ── */
    .alert-row{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}
    .alert-card{flex:1;min-width:200px;border-radius:14px;padding:14px 18px;
      display:flex;align-items:center;gap:12px;font-size:.83rem;font-weight:500;
      border:1px solid transparent;animation:slide-in .3s ease}
    @keyframes slide-in{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}
    .alert-em{background:#1a0a00;border-color:#f59e0b80;color:#fcd34d;animation:glow-yellow 1.5s ease-in-out infinite}
    @keyframes glow-yellow{0%,100%{box-shadow:0 0 10px #f59e0b20}50%{box-shadow:0 0 22px #f59e0b50}}
    .alert-icon{font-size:1.5rem;flex-shrink:0}
    .alert-title{font-weight:700;font-size:.88rem}
    .alert-sub{font-size:.75rem;opacity:.75;margin-top:2px}

    /* ── CHART ── */
    .chart-card{background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);border-radius:18px;
      padding:24px;margin-bottom:20px;backdrop-filter:blur(8px);transition:border-color .3s}
    .chart-card:hover{border-color:#a855f740}
    .chart-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:10px}
    .chart-title{font-size:.85rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
    .chart-controls{display:flex;gap:8px}
    .chart-btn{padding:5px 12px;border-radius:8px;font-size:.72rem;font-weight:600;
      font-family:'Inter',sans-serif;cursor:pointer;border:1px solid var(--border);
      background:var(--surface2);color:var(--muted);transition:all .18s}
    .chart-btn.active{background:#2d1b69;border-color:var(--accent);color:var(--accent)}

    /* ── TABLE ── */
    .table-card{background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);border-radius:18px;overflow:hidden;margin-bottom:20px;backdrop-filter:blur(8px)}
    .table-header{display:flex;align-items:center;justify-content:space-between;padding:18px 22px 0;flex-wrap:wrap;gap:10px;margin-bottom:14px}
    .section-title{font-size:.85rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
    .table-hint{font-size:.72rem;color:var(--muted);font-style:italic}
    table{width:100%;border-collapse:collapse}
    th{padding:10px 16px;text-align:left;font-size:.68rem;color:var(--muted);
      text-transform:uppercase;letter-spacing:.08em;font-weight:600;border-bottom:1px solid var(--border)}
    td{padding:12px 16px;border-bottom:1px solid #110e1a60;font-size:.84rem;
      transition:background .2s}
    tr:last-child td{border-bottom:none}
    tbody tr{cursor:pointer;transition:all .2s}
    tbody tr:hover td{background:#1f1535}
    tbody tr.lane-selected td{background:#2d1b69!important}
    tbody tr.lane-selected{outline:none}

    .los-badge{display:inline-flex;align-items:center;justify-content:center;
      width:28px;height:28px;border-radius:8px;font-size:.75rem;font-weight:800}
    .badge-green{background:#14532d40;color:#4ade80;border:1px solid #4ade8030}
    .badge-yellow{background:#713f1240;color:#fbbf24;border:1px solid #fbbf2430}
    .badge-orange{background:#7c2d1240;color:#fb923c;border:1px solid #fb923c30}
    .badge-red{background:#45090940;color:#f87171;border:1px solid #f8717130}

    .status-pill{display:inline-block;padding:3px 10px;border-radius:100px;font-size:.68rem;font-weight:700}
    .pill-clear{background:#14532d30;color:#4ade80;border:1px solid #4ade8025}
    .pill-moderate{background:#713f1230;color:#fbbf24;border:1px solid #fbbf2425}
    .pill-congested{background:#45090930;color:#f87171;border:1px solid #f8717125}

    .queue-val{color:var(--red);font-weight:700}
    .trend-up{color:var(--red);font-size:1rem;font-weight:800}
    .trend-down{color:var(--green);font-size:1rem;font-weight:800}
    .trend-flat{color:var(--muted);font-size:1rem;font-weight:800}

    /* ── SPEED TABLE ── */
    .speed-card{background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);border-radius:18px;overflow:hidden;margin-bottom:20px;backdrop-filter:blur(8px)}
    .speed-row-highlight{background:#2d0c0c}

    /* ── LEGEND ── */
    .legend{display:flex;gap:20px;flex-wrap:wrap;align-items:center;
      background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);border-radius:14px;
      padding:14px 20px;margin-bottom:20px;font-size:.75rem;backdrop-filter:blur(8px)}
    .legend-group{display:flex;align-items:center;gap:7px;color:var(--muted)}
    .legend-dot{width:8px;height:8px;border-radius:2px}
    .legend-sep{width:1px;height:16px;background:var(--border)}

    /* ── KEYS ── */
    .keys-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:8px}
    .key-item{background:var(--surface);border:1px solid var(--border);border-radius:12px;
      padding:12px 14px;display:flex;align-items:center;gap:10px}
    .key-cap{font-family:monospace;font-size:.95rem;font-weight:800;min-width:28px;
      text-align:center;background:#0d1530;border:1.5px solid var(--border);
      border-radius:6px;padding:2px 8px;color:var(--accent)}
    .key-desc{font-size:.75rem;color:var(--muted)}
    .key-name{font-weight:600;color:var(--text);font-size:.8rem;margin-bottom:2px}

    /* ── FOOTER ── */
    .footer{text-align:center;font-size:.72rem;color:var(--muted);margin-top:8px;padding-bottom:12px}
    .footer a{color:var(--accent);text-decoration:none}

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar{width:6px;height:6px}
    ::-webkit-scrollbar-track{background:var(--bg)}
    ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
    ::-webkit-scrollbar-thumb:hover{background:#a855f780}
  </style>
</head>
<body>
<div class="wrap">

  <!-- HEADER -->
  <header>
    <div class="brand">
      <div class="brand-icon">&#x1F6A6;</div>
      <div>
        <h1>Traffic Intelligence</h1>
        <p>Real-time analysis &middot; localhost:5050</p>
      </div>
    </div>
    <div class="header-controls">
      <div class="live-badge" id="live-badge">
        <span class="live-dot" id="live-dot"></span>
        <span id="live-txt">LIVE</span>
      </div>
      <button class="btn" id="btn-pause" onclick="togglePause()">&#x23F8;&#xFE0F; Pause</button>
      <button class="btn btn-export" onclick="exportData()">&#x2B07;&#xFE0F; Export JSON</button>
      <a href="/logout" class="btn danger" style="text-decoration:none">&#x1F6AA; Logout</a>
    </div>
  </header>

  <!-- KPI CARDS -->
  <div class="kpi-grid">
    <div class="kpi"><div class="kpi-icon">&#x1F697;</div><div class="kpi-label">Total Vehicles</div><div class="kpi-val blue" id="k-veh">—</div><div class="kpi-sub" id="k-veh-sub">this session</div><div class="kpi-bar" style="background:linear-gradient(90deg,#3b82f6,#06b6d4)"></div></div>
    <div class="kpi"><div class="kpi-icon">&#x1F4CD;</div><div class="kpi-label">Active Lanes</div><div class="kpi-val green" id="k-lanes">—</div><div class="kpi-sub">monitored</div><div class="kpi-bar" style="background:linear-gradient(90deg,#22c55e,#06b6d4);width:80%"></div></div>
    <div class="kpi"><div class="kpi-icon">&#x26A0;&#xFE0F;</div><div class="kpi-label">Incidents</div><div class="kpi-val green" id="k-inc">—</div><div class="kpi-sub">stopped vehicles</div><div class="kpi-bar" style="background:linear-gradient(90deg,#ef4444,#f59e0b);width:30%"></div></div>
    <div class="kpi"><div class="kpi-icon">&#x26A1;</div><div class="kpi-label">Mode</div><div class="kpi-val yellow" id="k-mode" style="font-size:1.1rem;letter-spacing:0">—</div><div class="kpi-sub">press keys to switch</div><div class="kpi-bar" style="background:linear-gradient(90deg,#f59e0b,#f97316);width:55%"></div></div>
    <div class="kpi"><div class="kpi-icon">&#x1F4F9;</div><div class="kpi-label">FPS</div><div class="kpi-val cyan" id="k-fps">—</div><div class="kpi-sub" id="k-frame-sub">frame —</div><div class="kpi-bar" style="background:linear-gradient(90deg,#06b6d4,#8b5cf6);width:70%"></div></div>
    <div class="kpi"><div class="kpi-icon">&#x1F6A8;</div><div class="kpi-label">Speeders</div><div class="kpi-val red" id="k-speeders">—</div><div class="kpi-sub" id="k-speed-limit">limit: — km/h</div><div class="kpi-bar" style="background:linear-gradient(90deg,#ef4444,#ec4899);width:40%"></div></div>
  </div>

  <!-- ALERT BANNERS -->
  <div id="em-banner" style="display:none;margin-bottom:16px">
    <div class="alert-card alert-em">
      <span class="alert-icon">&#x1F6A8;</span>
      <div><div class="alert-title">EMERGENCY VEHICLE</div><div class="alert-sub" id="em-txt">Lane — given priority</div></div>
    </div>
  </div>

  <!-- INCIDENTS -->
  <div id="inc-section" style="display:none;margin-bottom:16px">
    <div id="inc-list"></div>
  </div>

  <!-- CHART -->
  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">&#x1F4CA; Lane Traffic &mdash; Last 2&nbsp;min</span>
      <div class="chart-controls">
        <button class="chart-btn active" id="btn-bar" onclick="setChartType('bar')">Bar</button>
        <button class="chart-btn" id="btn-line" onclick="setChartType('line')">Line</button>
      </div>
    </div>
    <canvas id="lane-chart" height="72"></canvas>
  </div>

  <!-- LEGEND ROW -->
  <div class="legend">
    <span style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:var(--muted)">Level of Service (Congestion)</span>
    <div class="legend-group"><div class="legend-dot" style="background:#4ade80"></div>A–B: Smooth / No delays</div>
    <div class="legend-group"><div class="legend-dot" style="background:#fbbf24"></div>C: Stable traffic</div>
    <div class="legend-group"><div class="legend-dot" style="background:#fb923c"></div>D: Slowing down</div>
    <div class="legend-group"><div class="legend-dot" style="background:#f87171"></div>E–F: Stop & Go / Jam</div>
    <div class="legend-sep"></div>
    <span style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:var(--muted)">Trend</span>
    <div class="legend-group"><span style="color:var(--red);font-weight:700">&#x2191;</span> Increasing</div>
    <div class="legend-group"><span style="color:var(--green);font-weight:700">&#x2193;</span> Decreasing</div>
    <div class="legend-group"><span style="color:var(--muted);font-weight:700">&#x2192;</span> Stable</div>
    <div class="legend-sep"></div>
    <div class="legend-group" style="margin-left:auto;font-style:italic">&#x1F4A1; Click a lane row to highlight it</div>
  </div>

  <!-- LANE TABLE -->
  <div class="table-card">
    <div class="table-header">
      <span class="section-title">&#x1F6E3;&#xFE0F; Lane Summary</span>
      <span class="table-hint" id="filter-hint"></span>
    </div>
    <table>
      <thead><tr>
        <th>Lane</th><th>Cars</th><th>Buses</th><th>Trucks</th><th>Bikes</th>
        <th>Total</th><th>LOS</th><th>Flow/min</th><th>Queue@Red</th><th>Status</th><th>Trend</th>
      </tr></thead>
      <tbody id="lane-tbody"></tbody>
    </table>
  </div>

  <!-- SPEED CAMERA TABLE -->
  <div id="speed-section" style="display:none">
    <div class="speed-card">
      <div class="table-header">
        <span class="section-title">&#x1F4F7; Speed Camera &mdash; Recent Violations</span>
      </div>
      <table>
        <thead><tr><th>Time</th><th>Track ID</th><th>Lane</th><th>Speed</th><th>Class</th></tr></thead>
        <tbody id="speed-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- KEYBOARD CONTROLS -->
  <div style="margin-bottom:20px">
    <div style="font-size:.85rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:12px">&#x2328;&#xFE0F; Keyboard Controls <span style="font-weight:400;text-transform:none;letter-spacing:0">(press in video window)</span></div>
    <div class="keys-grid">
      <div class="key-item"><span class="key-cap">L</span><div><div class="key-name">Lanes</div><div class="key-desc">Show lane regions &amp; counts</div></div></div>
      <div class="key-item"><span class="key-cap">H</span><div><div class="key-name">Heatmap</div><div class="key-desc">Vehicle density overlay</div></div></div>
      <div class="key-item"><span class="key-cap">S</span><div><div class="key-name">Speed</div><div class="key-desc">Labels per vehicle</div></div></div>
      <div class="key-item"><span class="key-cap">T</span><div><div class="key-name">Timer</div><div class="key-desc">Adaptive signal timing</div></div></div>
      <div class="key-item"><span class="key-cap" style="color:var(--red);border-color:var(--red)">Q</span><div><div class="key-name" style="color:var(--red)">Quit</div><div class="key-desc">Stop &amp; save session</div></div></div>
    </div>
  </div>

  <div class="footer">Traffic Intelligence Dashboard &mdash; <a href="/api/stats">JSON API</a> &middot; <a href="/api/history">History API</a></div>
</div>

<script>
// ── CONSTANTS ──
const COLORS=['#a855f7','#22c55e','#f59e0b','#ef4444','#7c3aed','#c084fc','#f97316','#ec4899'];
const LOS_C={A:'badge-green',B:'badge-green',C:'badge-yellow',D:'badge-orange',E:'badge-red',F:'badge-red'};
const LOS_COL={A:'#4ade80',B:'#a3e635',C:'#fbbf24',D:'#fb923c',E:'#f87171',F:'#dc2626'};

// ── STATE ──
let paused=false, selectedLane=null, chartType='bar', lastData={};

// ── CHART INIT ──
const ctx=document.getElementById('lane-chart');
let chart=new Chart(ctx,{
  type:'bar',
  data:{labels:[],datasets:[]},
  options:{
    responsive:true,animation:{duration:300},
    plugins:{legend:{labels:{color:'#64748b',font:{family:'Inter',size:11},boxWidth:12,boxHeight:12,padding:16}}},
    scales:{
      x:{ticks:{color:'#475569',font:{family:'Inter',size:10}},grid:{color:'#0d122580'}},
      y:{ticks:{color:'#475569',font:{family:'Inter',size:10}},grid:{color:'#1e2d5080'},beginAtZero:true}
    }
  }
});

function setChartType(t){
  chartType=t;
  document.getElementById('btn-bar').className='chart-btn'+(t==='bar'?' active':'');
  document.getElementById('btn-line').className='chart-btn'+(t==='line'?' active':'');
  // Mutate type in-place — Chart.js v4 supports this without destroy/recreate
  chart.config.type = t;
  chart.data.datasets.forEach(ds=>{
    ds.backgroundColor = ds.borderColor+(t==='bar'?'66':'22');
    ds.borderWidth     = t==='bar'?0:2;
    ds.borderRadius    = t==='bar'?6:0;
    ds.tension         = 0.4;
    ds.fill            = t==='line';
    ds.pointRadius     = t==='line'?2:0;
    ds.pointHoverRadius= 5;
  });
  chart.update();
}

function togglePause(){
  paused=!paused;
  const btn=document.getElementById('btn-pause');
  const badge=document.getElementById('live-badge');
  const dot=document.getElementById('live-dot');
  const txt=document.getElementById('live-txt');
  btn.innerHTML=paused?'&#x25B6;&#xFE0F; Resume':'&#x23F8;&#xFE0F; Pause';
  btn.className='btn'+(paused?' active':'');
  if(paused){
    dot.style.animation='none';
    dot.style.background='#f59e0b';
    txt.style.color='#fcd34d';
    txt.textContent='PAUSED';
    badge.classList.add('badge-paused');
    badge.classList.remove('badge-offline');
  } else {
    dot.style.animation='pulse-dot 1.4s ease-in-out infinite';
    dot.style.background='#22c55e';
    txt.style.color='#86efac';
    txt.textContent='LIVE';
    badge.classList.remove('badge-paused','badge-offline');
  }
}

function exportData(){
  const blob=new Blob([JSON.stringify(lastData,null,2)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='traffic_snapshot_'+new Date().toISOString().slice(0,19).replace(/:/g,'-')+'.json';
  a.click();
}

function losClass(l){return LOS_C[l]||'badge-green';}
function statusPill(t){
  if(t<5)return '<span class="status-pill pill-clear">CLEAR</span>';
  if(t<15)return '<span class="status-pill pill-moderate">MODERATE</span>';
  return '<span class="status-pill pill-congested">CONGESTED</span>';
}
function trendEl(a){
  if(a==='\u2191')return '<span class="trend-up">\u2191</span>';
  if(a==='\u2193')return '<span class="trend-down">\u2193</span>';
  return '<span class="trend-flat">\u2192</span>';
}

function animateVal(el,newVal){
  const cur=parseFloat(el.textContent)||0;
  if(cur===newVal||el.textContent==='—')return el.textContent=newVal;
  el.style.transition='color .3s';
  el.style.color=newVal>cur?'var(--red)':'var(--green)';
  el.textContent=newVal;
  setTimeout(()=>el.style.color='',500);
}

function updateChart(hist){
  if(!hist.length)return;
  const labels=hist.map(h=>h.t);
  let lids=[...new Set(hist.flatMap(h=>Object.keys(h.lanes||{})))].sort();
  if (selectedLane) {
    lids = lids.filter(l => String(l) === String(selectedLane));
  }
  const newDatasets=lids.map((lid,i)=>({
    label:'Lane '+lid,
    data:hist.map(h=>(h.lanes||{})[lid]??0),
    backgroundColor:COLORS[i%COLORS.length]+(chartType==='bar'?'66':'22'),
    borderColor:COLORS[i%COLORS.length],
    borderWidth:chartType==='bar'?0:2,
    borderRadius:chartType==='bar'?6:0,
    tension:0.4,fill:chartType==='line',
    pointRadius:chartType==='line'?2:0,
    pointHoverRadius:5,
  }));
  chart.data.labels=labels;
  chart.data.datasets=newDatasets;
  chart.update('none');
}

function renderData(s, hist) {
  // KPIs
  animateVal(document.getElementById('k-veh'),s.vehicle_count??'—');
  document.getElementById('k-lanes').textContent=Object.keys(s.lane_counts||{}).length;
  const incEl=document.getElementById('k-inc');
  incEl.textContent=(s.incidents||[]).length;
  incEl.className='kpi-val '+(s.incidents?.length?'red':'green');
  document.getElementById('k-mode').textContent=(s.mode||'').toUpperCase()||'—';
  document.getElementById('k-fps').textContent=s.fps??'—';
  document.getElementById('k-frame-sub').textContent='frame '+(s.frame_id??'—');
  document.getElementById('k-speeders').textContent=(s.speeders||[]).length;
  if(s.speed_limit)document.getElementById('k-speed-limit').textContent='limit: '+s.speed_limit+' km/h';

  // Emergency
  const emb=document.getElementById('em-banner');
  if(s.emergency_active){
    emb.style.display='block';
    document.getElementById('em-txt').textContent='Lane '+s.emergency_lane+' given priority — timer extended';
  } else emb.style.display='none';

  // Incidents
  const is=document.getElementById('inc-section');
  if(s.incidents?.length){
    is.style.display='block';
    document.getElementById('inc-list').innerHTML=s.incidents.map(inc=>`
      <div class="alert-card" style="background:#1a0808;border-color:#ef444450;color:#fca5a5;margin-bottom:8px">
        <span class="alert-icon">&#x1F6A8;</span>
        <div>
          <div class="alert-title">Incident &mdash; Lane ${inc.lane}</div>
          <div class="alert-sub">Vehicle #${inc.track_id} stopped ${inc.duration}s at (${inc.cx}, ${inc.cy})</div>
        </div>
      </div>`).join('');
  } else is.style.display='none';

  // Lane table
  const tb=document.getElementById('lane-tbody');
  tb.innerHTML='';
  for(const[lid,counts] of Object.entries(s.lane_counts||{})){
    const total=Object.values(counts).reduce((a,b)=>a+b,0);
    const los=(s.lane_los||{})[lid]||'?';
    const flow=(s.lane_flow||{})[lid]??'—';
    const q=(s.lane_queue||{})[lid]??0;
    const arrow=(s.lane_trends||{})[lid]||'\u2192';
    const isSelected=String(selectedLane)===String(lid);
    const lc=LOS_COL[los]||'#64748b';
    const flowDisp=typeof flow==='number'?flow.toFixed(1):flow;
    tb.innerHTML+=`
      <tr onclick="selectLane('${lid}')" ${isSelected?'class="lane-selected"':''} id="lane-row-${lid}">
        <td><b style="font-size:.9rem">Lane ${lid}</b></td>
        <td>${counts.car||0}</td><td>${counts.bus||0}</td>
        <td>${counts.truck||0}</td><td>${counts.motorbike||0}</td>
        <td><b>${total}</b></td>
        <td><span class="los-badge ${losClass(los)}">${los}</span></td>
        <td style="font-variant-numeric:tabular-nums">${flowDisp}</td>
        <td>${q>0?`<span class="queue-val">${q}</span>`:q}</td>
        <td>${statusPill(total)}</td>
        <td>${trendEl(arrow)}</td>
      </tr>`;
  }
  if(selectedLane)document.getElementById('filter-hint').textContent='Lane '+selectedLane+' selected — click again to deselect';
  else document.getElementById('filter-hint').textContent='';

  // Speed
  const ss=document.getElementById('speed-section');
  if(s.speeders?.length){
    ss.style.display='block';
    document.getElementById('speed-tbody').innerHTML=[...s.speeders].reverse().slice(0,12).map(sp=>`
      <tr>
        <td style="color:var(--muted);font-size:.76rem">${sp.timestamp}</td>
        <td><b>#${sp.track_id}</b></td>
        <td>Lane ${sp.lane||'—'}</td>
        <td><span style="color:var(--red);font-weight:700">${sp.speed_kmh} km/h</span></td>
        <td style="color:var(--muted)">${sp.class||'—'}</td>
      </tr>`).join('');
  } else ss.style.display='none';

  // Chart
  updateChart(hist);
}

async function poll(){
  if(paused)return;
  try{
    const[sr,hr]=await Promise.all([fetch('/api/stats'),fetch('/api/history')]);
    if(!sr.ok || !hr.ok) throw new Error('Network response was not ok');
    lastData = {s: await sr.json(), hist: await hr.json()};
    
    // Restore LIVE status if we were offline
    const badge=document.getElementById('live-badge');
    const dot=document.getElementById('live-dot');
    const txt=document.getElementById('live-txt');
    if(badge.classList.contains('badge-offline')){
      badge.classList.remove('badge-offline');
      dot.style.animation='pulse-dot 1.4s ease-in-out infinite';
      dot.style.background='#22c55e';
      txt.style.color='#86efac';
      txt.textContent='LIVE';
    }
    
    renderData(lastData.s, lastData.hist);
  }catch(e){
    console.warn('poll err',e);
    const badge=document.getElementById('live-badge');
    const dot=document.getElementById('live-dot');
    const txt=document.getElementById('live-txt');
    badge.classList.add('badge-offline');
    dot.style.animation='none';
    dot.style.background='#ef4444';
    txt.style.color='#fca5a5';
    txt.textContent='OFFLINE';
  }
}

function selectLane(lid){
  if(String(selectedLane)===String(lid)){selectedLane=null;}
  else{selectedLane=lid;}
  if(lastData.s) renderData(lastData.s, lastData.hist);
}

poll(); setInterval(poll,2000);
</script>
</body></html>
"""

app = Flask(__name__)
app.secret_key = "traffic-intel-secret-2024"

DASH_USER = os.environ.get("DASH_USER", "admin")
DASH_PASS = os.environ.get("DASH_PASS", "changeme")


LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Traffic Intelligence &mdash; AI-Powered Traffic Analysis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"/>
  <style>
    :root{--bg:#08060e;--surface:#110e1a;--surface2:#181326;--border:#2a1f4a;
      --accent:#a855f7;--accent2:#7c3aed;--green:#22c55e;--muted:#8b7fb0;--cyan:#c084fc;--text:#e2e8f0;}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);overflow-x:hidden}
    body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
      background:radial-gradient(ellipse 80% 50% at 50% -20%,#2d1b6955,transparent),
                 radial-gradient(ellipse 60% 40% at 80% 110%,#4c1d9544,transparent),
                 radial-gradient(ellipse 40% 30% at 10% 60%,#7c3aed18,transparent);}
    nav{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;
      padding:16px 48px;background:rgba(8,6,14,.85);backdrop-filter:blur(20px);border-bottom:1px solid var(--border)}
    .nav-brand{display:flex;align-items:center;gap:12px;text-decoration:none}
    .nav-icon{width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#a855f7,#7c3aed);
      display:flex;align-items:center;justify-content:center;font-size:1.2rem;box-shadow:0 0 20px #a855f740}
    .nav-title{font-size:1.1rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em}
    .nav-btn{padding:10px 22px;border-radius:10px;background:linear-gradient(135deg,#a855f7,#7c3aed);
      color:#fff;font-family:'Inter',sans-serif;font-size:.85rem;font-weight:700;border:none;cursor:pointer;
      box-shadow:0 0 20px #a855f740;transition:all .2s}
    .nav-btn:hover{transform:translateY(-2px);box-shadow:0 4px 30px #a855f760}
    .hero{position:relative;z-index:1;text-align:center;padding:120px 24px 80px}
    .hero-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 16px;border-radius:100px;
      background:#2d1b6940;border:1px solid #a855f740;font-size:.75rem;font-weight:600;color:var(--cyan);
      margin-bottom:28px;letter-spacing:.08em;text-transform:uppercase}
    .hero-badge-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pdot 1.4s ease-in-out infinite}
    @keyframes pdot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.7)}}
    .hero h1{font-size:clamp(2.5rem,6vw,4.5rem);font-weight:900;letter-spacing:-.04em;line-height:1.05;margin-bottom:24px}
    .hero h1 span{background:linear-gradient(135deg,#a855f7,#c084fc,#e879f9);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .hero-sub{font-size:1.1rem;color:var(--muted);max-width:580px;margin:0 auto 40px;line-height:1.7}
    .hero-btns{display:flex;gap:14px;justify-content:center;flex-wrap:wrap}
    .btn-primary{padding:14px 32px;border-radius:12px;background:linear-gradient(135deg,#a855f7,#7c3aed);
      color:#fff;font-family:'Inter',sans-serif;font-size:.95rem;font-weight:700;border:none;cursor:pointer;
      box-shadow:0 0 30px #a855f740;transition:all .25s}
    .btn-primary:hover{transform:translateY(-3px);box-shadow:0 8px 40px #a855f760}
    .btn-ghost{padding:14px 32px;border-radius:12px;background:transparent;color:var(--text);
      font-family:'Inter',sans-serif;font-size:.95rem;font-weight:600;border:1px solid var(--border);
      cursor:pointer;transition:all .25s}
    .btn-ghost:hover{border-color:var(--accent);color:var(--accent);background:#1a0f30;transform:translateY(-2px)}
    .stats-bar{position:relative;z-index:1;display:flex;max-width:900px;margin:0 auto 100px;
      border:1px solid var(--border);border-radius:20px;
      background:linear-gradient(145deg,var(--surface),var(--surface2));overflow:hidden;backdrop-filter:blur(8px)}
    .stat-item{flex:1;padding:28px 20px;text-align:center;position:relative}
    .stat-item+.stat-item::before{content:'';position:absolute;left:0;top:20%;height:60%;width:1px;background:var(--border)}
    .stat-num{font-size:1.8rem;font-weight:900;background:linear-gradient(135deg,#a855f7,#c084fc);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .stat-label{font-size:.72rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-top:4px}
    .section{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:0 24px 100px}
    .section-tag{display:inline-block;padding:4px 14px;border-radius:100px;background:#2d1b6940;
      border:1px solid #a855f740;font-size:.72rem;font-weight:700;color:var(--cyan);
      text-transform:uppercase;letter-spacing:.1em;margin-bottom:16px}
    .section-title{font-size:clamp(1.8rem,4vw,2.8rem);font-weight:800;letter-spacing:-.03em;margin-bottom:14px}
    .section-sub{color:var(--muted);font-size:1rem;line-height:1.7;max-width:520px;margin-bottom:56px}
    .feat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}
    .feat-card{background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);
      border-radius:20px;padding:28px;transition:all .3s;position:relative;overflow:hidden}
    .feat-card:hover{border-color:var(--accent);transform:translateY(-5px);box-shadow:0 12px 40px rgba(168,85,247,.15)}
    .feat-card::before{content:'';position:absolute;inset:0;border-radius:20px;opacity:0;
      background:linear-gradient(135deg,rgba(168,85,247,.07),rgba(124,58,237,.04));transition:opacity .3s}
    .feat-card:hover::before{opacity:1}
    .feat-icon{font-size:2rem;margin-bottom:16px}.feat-title{font-size:1.05rem;font-weight:700;margin-bottom:8px}
    .feat-desc{font-size:.85rem;color:var(--muted);line-height:1.65}
    .feat-tag{display:inline-block;margin-top:14px;padding:3px 10px;border-radius:6px;
      font-size:.68rem;font-weight:700;background:#2d1b6940;color:var(--cyan);border:1px solid #a855f730}
    .metrics-preview{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-top:56px}
    .metric-card{background:linear-gradient(145deg,var(--surface),var(--surface2));
      border:1px solid var(--border);border-radius:16px;padding:20px;text-align:center;transition:all .25s}
    .metric-card:hover{border-color:var(--accent);transform:translateY(-3px);box-shadow:0 8px 24px rgba(168,85,247,.12)}
    .metric-card-icon{font-size:1.5rem;margin-bottom:10px}
    .metric-card-val{font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#a855f7,#c084fc);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .metric-card-label{font-size:.72rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-top:4px}
    footer{position:relative;z-index:1;text-align:center;padding:36px 24px;
      border-top:1px solid var(--border);font-size:.78rem;color:var(--muted)}
    .modal-overlay{display:none;position:fixed;inset:0;z-index:1000;background:rgba(8,6,14,.88);
      backdrop-filter:blur(12px);align-items:center;justify-content:center}
    .modal-overlay.open{display:flex}
    .modal{background:linear-gradient(145deg,#110e1a,#181326);border:1px solid var(--border);
      border-radius:24px;padding:40px;width:100%;max-width:420px;
      box-shadow:0 0 80px rgba(168,85,247,.2);position:relative;animation:modal-in .25s ease}
    @keyframes modal-in{from{opacity:0;transform:scale(.95) translateY(16px)}to{opacity:1;transform:scale(1) translateY(0)}}
    .modal-close{position:absolute;top:16px;right:18px;background:none;border:none;
      color:var(--muted);font-size:1.4rem;cursor:pointer;transition:color .15s}
    .modal-close:hover{color:var(--text)}
    .modal-icon{width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,#a855f7,#7c3aed);
      display:flex;align-items:center;justify-content:center;font-size:1.6rem;
      margin:0 auto 20px;box-shadow:0 0 30px #a855f740}
    .modal h2{text-align:center;font-size:1.4rem;font-weight:800;margin-bottom:6px}
    .modal-sub{text-align:center;font-size:.82rem;color:var(--muted);margin-bottom:28px}
    .form-group{margin-bottom:18px}
    .form-label{display:block;font-size:.75rem;font-weight:600;color:var(--muted);
      text-transform:uppercase;letter-spacing:.08em;margin-bottom:7px}
    .form-input{width:100%;padding:12px 16px;border-radius:10px;background:#0d0a14;
      border:1px solid var(--border);color:var(--text);font-family:'Inter',sans-serif;
      font-size:.9rem;outline:none;transition:border-color .2s}
    .form-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(168,85,247,.12)}
    .login-btn{width:100%;padding:13px;border-radius:10px;
      background:linear-gradient(135deg,#a855f7,#7c3aed);color:#fff;
      font-family:'Inter',sans-serif;font-size:.95rem;font-weight:700;
      border:none;cursor:pointer;box-shadow:0 0 24px #a855f730;transition:all .2s;margin-top:4px}
    .login-btn:hover{transform:translateY(-2px);box-shadow:0 6px 32px #a855f750}
    .login-error{display:none;background:#2d0c0c;border:1px solid #ef444450;border-radius:8px;
      padding:10px 14px;font-size:.82rem;color:#fca5a5;margin-bottom:16px}
    .login-error.show{display:block}
    ::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}
    ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
    ::-webkit-scrollbar-thumb:hover{background:#a855f780}
  </style>
</head>
<body>
<nav>
  <a class="nav-brand" href="/"><div class="nav-icon">&#x1F6A6;</div><span class="nav-title">Traffic Intelligence</span></a>
  <button class="nav-btn" onclick="openLogin()">&#x1F511; Login to Dashboard</button>
</nav>
<div class="hero">
  <div class="hero-badge"><span class="hero-badge-dot"></span>AI-Powered &middot; Real-Time &middot; YOLOv8</div>
  <h1>Next-Gen <span>Traffic Intelligence</span><br>at Your Fingertips</h1>
  <p class="hero-sub">Professional-grade traffic analysis powered by YOLOv8 + ByteTrack. Real-time detection, speed analysis, incident alerts and adaptive signals in a live dashboard.</p>
  <div class="hero-btns">
    <button class="btn-primary" onclick="openLogin()">&#x1F680; Open Dashboard</button>
    <button class="btn-ghost" onclick="document.getElementById('features').scrollIntoView({behavior:'smooth'})">&#x1F50E; Explore Features</button>
  </div>
</div>
<div class="stats-bar" style="max-width:900px;margin:0 auto 100px">
  <div class="stat-item"><div class="stat-num">YOLOv8</div><div class="stat-label">Detection Model</div></div>
  <div class="stat-item"><div class="stat-num">ByteTrack</div><div class="stat-label">Object Tracking</div></div>
  <div class="stat-item"><div class="stat-num">4</div><div class="stat-label">Vehicle Classes</div></div>
  <div class="stat-item"><div class="stat-num">2s</div><div class="stat-label">Update Interval</div></div>
</div>
<div class="section" id="features">
  <div class="section-tag">&#x2728; Features</div>
  <div class="section-title">Everything you need to<br>monitor traffic</div>
  <p class="section-sub">Industry-standard algorithms combined in one sleek dashboard.</p>
  <div class="feat-grid">
    <div class="feat-card"><div class="feat-icon">&#x1F3AF;</div><div class="feat-title">Multi-Object Tracking</div><div class="feat-desc">ByteTrack tracks every vehicle with unique IDs across frames, even through occlusions.</div><span class="feat-tag">ByteTrack</span></div>
    <div class="feat-card"><div class="feat-icon">&#x26A1;</div><div class="feat-title">Lane-Level Analytics</div><div class="feat-desc">Per-lane vehicle counts, LOS A&ndash;F grading, flow rate per minute, and queue length at red.</div><span class="feat-tag">HCM Standard</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F4F7;</div><div class="feat-title">Speed Camera</div><div class="feat-desc">Pixel-to-meter calibrated speed per vehicle. Violations logged to CSV in real time.</div><span class="feat-tag">Configurable Limit</span></div>
    <div class="feat-card"><div class="feat-icon">&#x26A0;&#xFE0F;</div><div class="feat-title">Incident Detection</div><div class="feat-desc">Flags vehicles stopped for &gt;5s with instant dashboard alerts showing location and duration.</div><span class="feat-tag">Auto Alert</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F6A8;</div><div class="feat-title">Emergency Priority</div><div class="feat-desc">Fast-moving large vehicles trigger automatic signal timer extension on their lane.</div><span class="feat-tag">Smart Signals</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F4CA;</div><div class="feat-title">Live Chart History</div><div class="feat-desc">2-minute time-series chart of lane traffic with instant bar / line toggle.</div><span class="feat-tag">Chart.js v4</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F525;</div><div class="feat-title">Heatmap Overlay</div><div class="feat-desc">Accumulative density heatmap of vehicle dwell time. PNG exportable for offline analysis.</div><span class="feat-tag">OpenCV</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F4C8;</div><div class="feat-title">Trend Prediction</div><div class="feat-desc">Rolling linear regression predicts traffic direction (rising / falling / stable) per lane.</div><span class="feat-tag">No ML Required</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F4BE;</div><div class="feat-title">CSV Data Logging</div><div class="feat-desc">Every detection and speed event logged to CSV. JSON snapshot export from the dashboard.</div><span class="feat-tag">Analytics-Ready</span></div>
  </div>
</div>
<div class="section">
  <div class="section-tag">&#x1F4DD; Dashboard Metrics</div>
  <div class="section-title">What the dashboard shows you</div>
  <p class="section-sub">All key metrics in real-time, updating every 2 seconds.</p>
  <div class="metrics-preview">
    <div class="metric-card"><div class="metric-card-icon">&#x1F697;</div><div class="metric-card-val">Live</div><div class="metric-card-label">Total Vehicles</div></div>
    <div class="metric-card"><div class="metric-card-icon">&#x1F4CD;</div><div class="metric-card-val">A&ndash;F</div><div class="metric-card-label">Level of Service</div></div>
    <div class="metric-card"><div class="metric-card-icon">&#x1F4F9;</div><div class="metric-card-val">FPS</div><div class="metric-card-label">Processing Speed</div></div>
    <div class="metric-card"><div class="metric-card-icon">&#x1F6A8;</div><div class="metric-card-val">Auto</div><div class="metric-card-label">Speeder Alerts</div></div>
    <div class="metric-card"><div class="metric-card-icon">&#x26A0;&#xFE0F;</div><div class="metric-card-val">Real-Time</div><div class="metric-card-label">Incidents</div></div>
    <div class="metric-card"><div class="metric-card-icon">&#x1F4CA;</div><div class="metric-card-val">2 Min</div><div class="metric-card-label">History Chart</div></div>
  </div>
</div>
<footer>Traffic Intelligence &mdash; AI-Powered Traffic Analysis &middot; Built with YOLOv8 + ByteTrack + Flask
  <span style="color:#a855f7;margin-left:8px">&#x2665; By Varnit</span>
</footer>
<div class="modal-overlay" id="modal-overlay" onclick="if(event.target===this)closeLogin()">
  <div class="modal">
    <button class="modal-close" onclick="closeLogin()">&times;</button>
    <div class="modal-icon">&#x1F6A6;</div>
    <h2>Welcome Back</h2>
    <p class="modal-sub">Enter your credentials to access the live dashboard</p>
    <div class="login-error" id="login-error">&#x26A0; Incorrect username or password</div>
    <form method="POST" action="/login">
      <div class="form-group"><label class="form-label">Username</label>
        <input class="form-input" type="text" name="username" id="inp-user" placeholder="Enter username" required/></div>
      <div class="form-group"><label class="form-label">Password</label>
        <input class="form-input" type="password" name="password" placeholder="Enter password" required/></div>
      <button class="login-btn" type="submit">&#x1F680; Access Dashboard</button>
    </form>
  </div>
</div>
<script>
function openLogin(){document.getElementById('modal-overlay').classList.add('open');setTimeout(()=>document.getElementById('inp-user').focus(),100);}
function closeLogin(){document.getElementById('modal-overlay').classList.remove('open');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLogin();});
if(new URLSearchParams(window.location.search).get('error')==='1'){openLogin();document.getElementById('login-error').classList.add('show');}
</script>
</body></html>"""


@app.route("/")
def home():
    return render_template_string(LANDING_HTML)

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    if username == DASH_USER and password == DASH_PASS:
        session["authenticated"] = True
        return redirect(url_for("dashboard"))
    return redirect(url_for("home") + "?error=1")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if not session.get("authenticated"):
        return redirect(url_for("home") + "?error=1")
    with state_lock:
        s = dict(shared_state)
    return render_template_string(DASHBOARD_HTML, stats=s)

@app.route("/api/stats")
def api_stats():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    with state_lock:
        return jsonify(shared_state)

@app.route("/api/history")
def api_history():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(list(history_buf))

def run_flask():
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)  # silence Flask request logs
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

# ──────────────────────────────────────────────
# ROI SELECTOR
# ──────────────────────────────────────────────
lanes = []
drawing, ix, iy = False, -1, -1
selecting = True

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, lanes, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing, ix, iy = True, x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        lanes.append((min(ix,x), min(iy,y), max(ix,x), max(iy,y)))
        print(f"Lane {len(lanes)} set: {lanes[-1]}")

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video. Make sure Traffic_Video.mp4 exists.")
    cap.release()
    exit()

cv2.namedWindow("ROI Selector")
cv2.setMouseCallback("ROI Selector", draw_rectangle)
print("Draw lane boxes with the mouse. Press ENTER when done.")

while selecting:
    display = frame.copy()
    for i, (x1,y1,x2,y2) in enumerate(lanes, 1):
        cv2.rectangle(display, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(display, f"Lane {i}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow("ROI Selector", display)
    k = cv2.waitKey(1) & 0xFF
    if k == 13: selecting = False
    elif k == ord("q"): cap.release(); exit()

cv2.destroyWindow("ROI Selector")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ──────────────────────────────────────────────
# INIT TRACKING + TOOLS
# ──────────────────────────────────────────────
model = YOLO(MODEL_NAME)
tracker = sv.ByteTrack()

incident_detector = IncidentDetector()
trend_tracker    = LaneTrendTracker()   # predictive optimisation
logger = CSVLogger(LOG_FILE)
speeder_logger = SpeedCameraLogger(SPEEDER_LOG_FILE)

# Speed tracking (ByteTrack gives persistent IDs)
speed_history = defaultdict(lambda: deque(maxlen=8))  # track_id → [(cx,cy,t), ...]
heatmap = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.float32)
flow_tracker  = FlowRateTracker()
lane_y_med: dict = {}          # lane_id → running median y to find dominant direction
wrong_way_counter: dict = {}   # track_id → consecutive frames flagged as wrong-way
vehicle_last_lane: dict = {}   # track_id → last known lane_id (fallback for speed cam)
history_buf: deque = deque(maxlen=40)   # ring buffer of (t, {lane_id: veh_count})


mode = "lanes"
signal_index, signal_timer, signal_start = 0, -1, time.time()  # -1 = uninitialised
last_priority_adjust_time = 0.0   # cooldown tracker for time-nudging
lane_last_green = {}               # {lane_index: timestamp when it last got green}
frame_id = 0
fps_timer = time.time()

# ──────────────────────────────────────────────
# START FLASK IN BACKGROUND
# ──────────────────────────────────────────────
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
print(f"📊 Live dashboard → http://localhost:{FLASK_PORT}")
# Give Flask a moment to start, then open the browser automatically
threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{FLASK_PORT}")).start()

# ──────────────────────────────────────────────
# HELPER: lane for a centroid
# ──────────────────────────────────────────────
def get_lane(cx, cy):
    for i, (lx1,ly1,lx2,ly2) in enumerate(lanes, 1):
        if lx1 <= cx <= lx2 and ly1 <= cy <= ly2:
            return i
    return None

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        # End of video — loop back to the start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break  # truly unreadable, give up
    frame_id += 1

    # ── FPS ──
    now = time.time()
    fps = 1.0 / max(now - fps_timer, 1e-9)
    fps_timer = now

    # ── YOLO DETECT ──
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    # Filter to vehicle classes only
    vehicle_mask = np.array([
        model.names[int(c)] in TRACK_CLASSES
        for c in results.boxes.cls
    ], dtype=bool)

    if vehicle_mask.any():
        filtered = results.boxes[vehicle_mask]
        sv_dets = sv.Detections(
            xyxy=filtered.xyxy.cpu().numpy(),
            confidence=filtered.conf.cpu().numpy(),
            class_id=filtered.cls.cpu().numpy().astype(int),
        )
    else:
        sv_dets = sv.Detections.empty()

    # ── BYTETRACK ──
    tracked = tracker.update_with_detections(sv_dets)

    # ── PER-FRAME ACCUMULATORS ──
    lane_counts = {i+1: {"car":0,"bus":0,"truck":0,"motorbike":0} for i in range(len(lanes))}
    active_ids  = set()
    active_incidents      = []
    frame_speeders        = []
    frame_wrong_way: set  = set()
    frame_tailgating: list = []
    queue_counts: dict    = {}     # lane_id → stopped-vehicle count
    emergency_lane_this_frame = None

    for i in range(len(tracked)):
        x1, y1, x2, y2 = map(int, tracked.xyxy[i])
        track_id = int(tracked.tracker_id[i])
        cls_id   = int(tracked.class_id[i])
        label    = model.names[cls_id]
        cx, cy   = (x1+x2)//2, (y1+y2)//2
        active_ids.add(track_id)
        lane_id = get_lane(cx, cy)
        if lane_id:
            vehicle_last_lane[track_id] = lane_id   # remember last confirmed lane
        else:
            lane_id = vehicle_last_lane.get(track_id)  # fall back to last known

        # Lane count
        if lane_id and label in lane_counts[lane_id]:
            lane_counts[lane_id][label] += 1

        # Speed (pixels/sec → km/h via PIXEL_TO_METER)
        speed_history[track_id].append((cx, cy, time.time()))
        speed_kmh = 0.0
        if len(speed_history[track_id]) >= 2:
            p1 = speed_history[track_id][0]
            p2 = speed_history[track_id][-1]
            dt = p2[2] - p1[2]
            if dt > 0:
                dist_px = hypot(p2[0]-p1[0], p2[1]-p1[1])
                speed_kmh = (dist_px * PIXEL_TO_METER / dt) * 3.6

        # ── SPEED CAMERA ──
        if speed_kmh > SPEED_LIMIT_KMPH:
            event = speeder_logger.log(frame_id, track_id, lane_id, speed_kmh, label)
            if event:
                frame_speeders.append(event)
            # Police-style red alert box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,220), 3)
            cv2.rectangle(frame, (x1, y1-28), (x1+220, y1), (0,0,220), -1)
            cv2.putText(frame, f"SPEEDING  {int(speed_kmh)} km/h", (x1+4, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)

        # ── EMERGENCY VEHICLE DETECTION ──
        if label in EMERGENCY_CLASSES and speed_kmh > EMERGENCY_SPEED_KMH and lane_id:
            emergency_lane_this_frame = lane_id

        # Incident detection
        is_incident = lane_id and incident_detector.update(track_id, cx, cy, lane_id)
        if is_incident:
            still_since = incident_detector.history[track_id]["still_since"]
            duration = round(time.time() - still_since, 1)
            active_incidents.append({
                "track_id": track_id, "lane": lane_id,
                "cx": cx, "cy": cy, "duration": duration
            })
            # Red highlight
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(frame, f"INCIDENT! {duration}s", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        elif speed_kmh <= SPEED_LIMIT_KMPH:  # don't overwrite speeding box
            # Normal annotation with speed
            color = (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            speed_txt = f"{int(speed_kmh)}km/h" if speed_kmh > 2 else label
            cv2.putText(frame, f"ID{track_id} {speed_txt}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        # ── FLOW RATE recording ──
        if lane_id:
            flow_tracker.record(lane_id, track_id)



        # Heatmap update
        cv2.circle(heatmap, (cx, cy), 12, 1, -1)

        # Lane label on vehicle
        if lane_id:
            cv2.putText(frame, f"L{lane_id}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    incident_detector.cleanup(active_ids)
    total_vehicles = sum(sum(c.values()) for c in lane_counts.values())

    # ── SESSION STATS + LOS + FLOW (computed once per frame, after vehicle loop) ──
    session_stats["all_ids"].update(active_ids)
    if total_vehicles > session_stats["peak_count"]:
        session_stats["peak_count"] = total_vehicles
        session_stats["peak_time"]  = datetime.now().strftime("%H:%M:%S")
    if frame_id % 90 == 0:   # history snapshot every ~3 s
        history_buf.append({
            "t": datetime.now().strftime("%H:%M:%S"),
            "lanes": {str(k): sum(v.values()) for k, v in lane_counts.items()}
        })
    lane_los_out  = {str(k): los_grade(sum(v.values()))[0] for k, v in lane_counts.items()}
    lane_flow_out = {str(k): flow_tracker.rate(k)          for k  in lane_counts}

    # ── TREND TRACKER: feed this frame's counts ──
    for lane_id, counts in lane_counts.items():
        trend_tracker.update(lane_id, sum(counts.values()))

    # ── MODES ──
    if mode == "lanes":
        for i, (lx1,ly1,lx2,ly2) in enumerate(lanes, 1):
            total = sum(lane_counts[i].values())
            color = (0,255,0) if total < 5 else (0,255,255) if total < 15 else (0,0,255)
            cv2.rectangle(frame, (lx1,ly1), (lx2,ly2), color, 2)
            cv2.putText(frame, f"Lane {i} ({total})", (lx1+5, ly1+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y = 60
        for i, c in lane_counts.items():
            t = sum(c.values())
            status = "CLEAR" if t < 5 else "MODERATE" if t < 15 else "CONGESTED"
            cv2.putText(frame, f"Lane {i}: {c['car']}C {c['bus']}B {c['truck']}T {c['motorbike']}M | {status}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            y += 22

    elif mode == "heatmap":
        blur = cv2.GaussianBlur(heatmap, (0,0), 25)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.55, colored, 0.45, 0)

    elif mode == "speed":
        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            x1,y1,x2,y2 = map(int, tracked.xyxy[i])
            history = speed_history[track_id]
            if len(history) >= 2:
                p1, p2 = history[0], history[-1]
                dt = p2[2] - p1[2]
                if dt > 0:
                    speed = (hypot(p2[0]-p1[0], p2[1]-p1[1]) * PIXEL_TO_METER / dt) * 3.6
                    color = (0,255,0) if speed < 40 else (0,165,255) if speed < 80 else (0,0,255)
                    cv2.putText(frame, f"{int(speed)} km/h", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    elif mode == "timer":
        # ── Helper: anti-starvation priority score ──
        # Score = vehicle_count + wait_bonus so lanes that haven't had green
        # in a long time naturally rise to the top even with few vehicles.
        # MAX_WAIT guarantees every lane is served at least once per 120s.
        MAX_WAIT       = 120   # seconds — forced green after this wait
        WAIT_SCALE     = 5.0   # 1 extra priority point per WAIT_SCALE seconds waited
        _now = time.time()
        def priority_score(lane_idx):
            vehicles  = sum(lane_counts[lane_idx + 1].values())
            trend_val = trend_tracker.trend(lane_idx + 1)   # +ve = rising demand
            waited    = _now - lane_last_green.get(lane_idx, _now - MAX_WAIT)
            if waited >= MAX_WAIT:          # starvation guard — force to front
                return float("inf")
            # Rising lanes jump the queue sooner; falling lanes wait a little longer
            return vehicles + (trend_val * 2) + (waited / WAIT_SCALE)

        def next_priority_lane():
            waiting = [i for i in range(len(lanes)) if i != signal_index]
            if not waiting:
                return (signal_index + 1) % len(lanes)
            return max(waiting, key=priority_score)

        # ── Initialise timer on first entry into timer mode ──
        if signal_timer < 0:
            lane_last_green[signal_index] = time.time()   # mark lane 0 as starting now
            total      = sum(lane_counts[signal_index + 1].values())
            trend_val  = trend_tracker.trend(signal_index + 1)
            trend_adj  = int(trend_val * 4)              # ~4s per slope unit
            signal_timer = min(90, max(15, total * 3 + trend_adj))
            signal_start = time.time()

        elapsed   = time.time() - signal_start
        remaining = max(0, int(signal_timer - elapsed))
        current_lane_vehicles = sum(lane_counts[signal_index + 1].values())
        now = time.time()
        ADJUST_COOLDOWN = 25   # seconds between adjustments (prevents per-frame trimming)
        MIN_EMERGENCY   = 10   # minimum seconds to leave on green during emergency trim
        MIN_CONGESTION  = 15   # minimum seconds to leave on green during congestion trim
        TRIM_EMERGENCY  = 20   # how many seconds to cut on emergency
        TRIM_CONGESTION = 10   # how many seconds to cut on congestion

        adjust_label = None

        # ── EMERGENCY PRIORITY: trim current green — don't hard-switch ──
        # Gives the current lane time to stop safely, then the priority lane
        # gets its turn sooner because the queue ahead of it is shorter.
        if (emergency_lane_this_frame is not None
                and (emergency_lane_this_frame - 1) != signal_index
                and now - last_priority_adjust_time >= ADJUST_COOLDOWN):
            new_remaining = max(MIN_EMERGENCY, remaining - TRIM_EMERGENCY)
            if new_remaining < remaining:           # only act if it actually shortens
                signal_timer = elapsed + new_remaining
                remaining    = new_remaining
                last_priority_adjust_time = now
                adjust_label = f"EMERGENCY DETECTED  |  Green shortened by {TRIM_EMERGENCY}s"

        # ── CONGESTION ADJUSTMENT: trim green when current lane has cleared ──
        # but a waiting lane is overflowing AND minimum hold-time has passed
        elif (now - last_priority_adjust_time >= ADJUST_COOLDOWN
              and elapsed >= 10          # held green for at least 10s first
              and remaining > MIN_CONGESTION):
            max_waiting = max(
                (sum(lane_counts[i + 1].values()) for i in range(len(lanes)) if i != signal_index),
                default=0
            )
            if current_lane_vehicles <= 2 and max_waiting >= 10:
                new_remaining = max(MIN_CONGESTION, remaining - TRIM_CONGESTION)
                if new_remaining < remaining:
                    signal_timer = elapsed + new_remaining
                    remaining    = new_remaining
                    last_priority_adjust_time = now
                    adjust_label = f"CONGESTION  |  Green shortened by {TRIM_CONGESTION}s"

        # Show adjustment banner if triggered
        if adjust_label:
            cv2.putText(frame, adjust_label, (20, frame.shape[0] - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        for i, (lx1, ly1, lx2, ly2) in enumerate(lanes):
            lane_total  = sum(lane_counts[i + 1].values())
            trend_arrow = trend_tracker.label_ascii(i + 1)
            if i == signal_index:
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 4)
                cv2.putText(frame, f"GO ({remaining}s) [{lane_total}v] {trend_arrow}",
                            (lx1 + 10, ly1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Estimate wait time for this lane
                red_time = remaining
                check = signal_index
                while check != i:
                    check = (check + 1) % len(lanes)
                    _t     = sum(lane_counts[check + 1].values())
                    _tadj  = int(trend_tracker.trend(check + 1) * 4)
                    red_time += min(90, max(15, _t * 3 + _tadj))
                box_color = (0, 160, 255) if lane_total < 5 else (0, 0, 220) if lane_total < 15 else (0, 0, 180)
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), box_color, 2)
                cv2.putText(frame, f"RED (~{red_time}s) [{lane_total}v] {trend_arrow}",
                            (lx1 + 10, ly1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

        # ── Advance to highest-priority waiting lane when timer expires ──
        if elapsed >= signal_timer:
            signal_index = next_priority_lane()
            lane_last_green[signal_index] = time.time()   # mark green start
            total      = sum(lane_counts[signal_index + 1].values())
            trend_val  = trend_tracker.trend(signal_index + 1)
            trend_adj  = int(trend_val * 4)              # pre-adjust for rising/falling demand
            signal_timer = min(90, max(15, total * 3 + trend_adj))
            signal_start = time.time()
            last_priority_adjust_time = 0.0   # reset cooldown for new phase

    # ── INCIDENT OVERLAYS ──
    for inc in active_incidents:
        cv2.putText(frame, f"[!] INCIDENT L{inc['lane']}", (20, frame.shape[0]-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # ── HUD BAR ──
    dash_url = f"http://localhost:{FLASK_PORT}"
    hud_text = f"MODE: {mode.upper()}  |  Vehicles: {total_vehicles}  |  FPS: {fps:.1f}  |  Incidents: {len(active_incidents)}  |  Dashboard: {dash_url}"
    hud_bar = np.zeros((46, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(hud_bar, hud_text, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    frame = np.vstack((hud_bar, frame))

    # ── UPDATE SHARED STATE ──
    with state_lock:
        shared_state["lane_counts"]       = {str(k): dict(v) for k, v in lane_counts.items()}
        shared_state["vehicle_count"]     = total_vehicles
        shared_state["incidents"]         = active_incidents
        shared_state["mode"]              = mode
        shared_state["fps"]               = round(fps, 1)
        shared_state["frame_id"]          = frame_id
        shared_state["emergency_active"]  = emergency_lane_this_frame is not None
        shared_state["emergency_lane"]    = emergency_lane_this_frame
        shared_state["lane_trends"]       = {str(k): trend_tracker.label(k) for k in lane_counts}
        shared_state["lane_los"]          = lane_los_out
        shared_state["lane_flow"]         = lane_flow_out
        shared_state["lane_queue"]        = {str(k): queue_counts.get(k, 0) for k in lane_counts}
        shared_state["wrong_way"]         = list(frame_wrong_way)
        shared_state["tailgating"]        = frame_tailgating[:5]   # cap at 5
        # Keep last 10 speeding events
        if frame_speeders:
            shared_state["speeders"] = (shared_state["speeders"] + frame_speeders)[-10:]

    # ── LOG EVERY 30 FRAMES ──
    if frame_id % 30 == 0:
        logger.log(frame_id, lane_counts, active_incidents)

    # ── DISPLAY ──
    cv2.imshow("Traffic Analysis  [L/H/S/T]  Q=quit", frame)
    key = cv2.waitKey(30) & 0xFF   # 30ms gives reliable key capture
    if key == ord("l"):
        mode = "lanes"
    elif key == ord("h"):
        mode = "heatmap"
    elif key == ord("s"):
        mode = "speed"
    elif key == ord("t"):
        mode = "timer"
    elif key == ord("q") or key == 27:   # Q or Esc
        break
    # Check if window was closed (WND_PROP_AUTOSIZE is reliable cross-platform)
    try:
        if cv2.getWindowProperty("Traffic Analysis  [L/H/S/T]  Q=quit",
                                  cv2.WND_PROP_AUTOSIZE) < 0:
            break
    except cv2.error:
        break

cap.release()
cv2.destroyAllWindows()

# Save heatmap on exit
heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("heatmap_export.png", cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET))

# ──────────────────────────────────────────────
# SESSION SUMMARY
# ──────────────────────────────────────────────
duration_s = (datetime.now() - datetime.fromisoformat(session_stats["session_start"])).seconds
print()
print("═" * 56)
print("  🚦  TRAFFIC SESSION SUMMARY")
print("═" * 56)
print(f"  Started         : {session_stats['session_start']}")
print(f"  Duration        : {duration_s // 60}m {duration_s % 60}s")
print(f"  Total vehicles  : {len(session_stats['all_ids'])} unique IDs")
print(f"  Peak traffic    : {session_stats['peak_count']} vehicles at {session_stats['peak_time']}")
print(f"  Incidents       : {session_stats['total_incidents']}")
print(f"  Wrong-way IDs   : {len(session_stats['wrong_way_ids'])} ({list(session_stats['wrong_way_ids'])[:8]})")
print(f"  Tailgate events : {session_stats['tailgate_events']}")
print(f"  Log             : {LOG_FILE}")
print(f"  Heatmap         : heatmap_export.png")
print("═" * 56)
print()
