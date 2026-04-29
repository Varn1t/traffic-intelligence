# traffic_v2.py — Update Log

**Date:** 2026-02-27  
**File:** `traffic_v2.py`  
**Status:** Syntax verified ✅

---

## What's New in This Session

### 1. 🟢 Level of Service (LOS) Grades (A–F)
- New `los_grade(count)` function rates each lane A (free flow) through F (breakdown) using standard traffic density thresholds
- LOS grade shown in the Lane Summary table on the dashboard
- Color-coded: green (A/B), yellow (C), red (D/E/F)

### 2. 📈 Flow Rate (vehicles/min per lane)
- New `FlowRateTracker` class tracks unique vehicle IDs entering each lane within a 60-second sliding window
- Flow rate displayed as a new **Flow/min** column in the lane table

### 3. ⛔ Wrong-Way Detection
- Compares each vehicle's movement vector against the dominant lane direction
- Wrong-way vehicle IDs flagged and stored in `_ss_wrong_ids`
- **Orange warning banner** appears on the dashboard when a wrong-way vehicle is detected
- Wrong-way IDs listed in the session summary on quit

### 4. ⚠️ Headway / Tailgating Warnings
- Checks following distance between consecutive vehicles in the same lane
- When two vehicles are closer than a safe headway threshold, a tailgating event is logged
- **Orange warning banner** shown on dashboard when tailgating is detected this frame

### 5. 🚦 Queue Length at Red Light
- In timer mode (press **T**), counts stopped vehicles (speed < threshold) per lane while they are waiting at a red light
- **Queue@Red** column in the lane table shows the count (highlighted red if > 0)

### 6. 📊 Live Chart.js Dashboard (replaces meta-refresh)
- Old dashboard used a jarring page reload (`<meta http-equiv="refresh">`) every 2 seconds
- **New dashboard** uses JavaScript AJAX polling (`fetch('/api/stats')` + `fetch('/api/history')`) every 2 seconds — no full page reload
- **Live bar chart** shows lane vehicle counts over the last ~2 minutes, powered by Chart.js (loaded from CDN)
- All KPI cards, lane table, incident list update smoothly in place
- New banner sections for emergency vehicle, wrong-way, and tailgating alerts

### 7. 📋 Session Summary on Quit
- When you press **Q** (or Esc) to quit, a summary is printed to the terminal:
  ```
  ════════════════════════════════════════════════════════
    🚦  TRAFFIC SESSION SUMMARY
  ════════════════════════════════════════════════════════
    Started         : 2026-02-27T09:07:00
    Duration        : 4m 22s
    Total vehicles  : 87 unique IDs
    Peak traffic    : 14 vehicles at 09:09:33
    Incidents       : 3
    Wrong-way IDs   : 1 ([42])
    Tailgate events : 7
    Log             : traffic_log_...csv
    Heatmap         : heatmap_export.png
  ════════════════════════════════════════════════════════
  ```

### 8. 🗂️ History Buffer + `/api/history` Endpoint
- Every ~3 seconds a snapshot of per-lane vehicle counts is stored in a rolling buffer (last 40 snapshots ≈ 2 min)
- Accessible at `http://localhost:5000/api/history` as JSON
- Used internally by the Chart.js chart

---

## Dashboard Changes at a Glance

| Old | New |
|-----|-----|
| Page reloads every 2s (meta-refresh) | Smooth AJAX polling — no reload |
| Lane table: Lane / Cars / Buses / Trucks / Bikes / Total / Status / Trend | + **LOS** / **Flow/min** / **Queue@Red** columns |
| No chart | Live bar chart (last 2 min) |
| No wrong-way / tailgating info | Orange alert banners appear when detected |

---

## How to Run (unchanged)

```bash
python traffic_v2.py
```

1. Draw lane boxes in the **ROI Selector** window → press **Enter**
2. Set speed limit when prompted
3. Dashboard: `http://localhost:5000`
4. Press **T** in the video window to activate Timer mode (traffic signals + queue counting)
5. Press **Q** to quit and see the session summary
