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
from datetime import datetime
from collections import defaultdict, deque
import threading
import webbrowser
from math import hypot

from ultralytics import YOLO
import supervision as sv

# Imports from modularized files
from config import (
    VIDEO_PATH, MODEL_NAME, CONF_THRESHOLD, TRACK_CLASSES,
    PIXEL_TO_METER, SPEED_LIMIT_KMPH, SPEEDER_LOG_FILE,
    EMERGENCY_CLASSES, EMERGENCY_SPEED_KMH, LOG_FILE, FLASK_PORT
)
from state import shared_state, state_lock, history_buf, session_stats
from loggers import CSVLogger, SpeedCameraLogger
from analytics import (
    IncidentDetector, LaneTrendTracker, TrafficPredictor,
    FlowRateTracker, los_grade, _SKLEARN_OK
)
from web_app import run_flask

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

def main():
    global lanes, drawing, ix, iy, selecting, frame
    
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

    # Instantiate ML predictor (one model per lane)
    predictor = TrafficPredictor(lane_ids=list(range(1, len(lanes) + 1)))
    print(f"[ML] TrafficPredictor initialised for {len(lanes)} lane(s).")
    print("     Model warms up after ~60 s of video (20 snapshots).")
    if not _SKLEARN_OK:
        print("[ML] scikit-learn missing — predictions will fall back to last count.")

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
    # Note: history_buf imported from state.py
    
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
        frame_tailgating = []  # list of tailgating event dicts (populated if tailgating detection is added)
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
            # ── Feed ML predictor with this snapshot ──
            for _lid, _cnts in lane_counts.items():
                predictor.update(_lid, sum(_cnts.values()))
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
                predicted = predictor.predict(lane_idx + 1)     # ML future demand
                trend_val = trend_tracker.trend(lane_idx + 1)
                waited    = _now - lane_last_green.get(lane_idx, _now - MAX_WAIT)
                if waited >= MAX_WAIT:          # starvation guard — force to front
                    return float("inf")
                # Blend current (55%) + predicted future (45%) demand
                demand = 0.55 * vehicles + 0.45 * predicted
                return demand + (trend_val * 2) + (waited / WAIT_SCALE)

            def calc_green_time(lane_idx: int) -> int:
                """Green duration blending current + ML-predicted demand."""
                current   = sum(lane_counts[lane_idx + 1].values())
                predicted = predictor.predict(lane_idx + 1)
                demand    = int(0.55 * current + 0.45 * predicted)
                trend_adj = int(trend_tracker.trend(lane_idx + 1) * 4)
                return min(90, max(15, demand * 3 + trend_adj))

            def next_priority_lane():
                waiting = [i for i in range(len(lanes)) if i != signal_index]
                if not waiting:
                    return (signal_index + 1) % len(lanes)
                return max(waiting, key=priority_score)

            # ── Initialise timer on first entry into timer mode ──
            if signal_timer < 0:
                lane_last_green[signal_index] = time.time()
                signal_timer = calc_green_time(signal_index)
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
                    _ml_go = f"ML~{int(predictor.predict(i+1))}v" if predictor.is_ready() else "ML:warmup"
                    cv2.putText(frame, _ml_go, (lx1 + 10, ly1 + 56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 160), 1)
                else:
                    # Estimate wait time for this lane
                    red_time = remaining
                    check = signal_index
                    while check != i:
                        check = (check + 1) % len(lanes)
                        red_time += calc_green_time(check)
                    box_color = (0, 160, 255) if lane_total < 5 else (0, 0, 220) if lane_total < 15 else (0, 0, 180)
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), box_color, 2)
                    cv2.putText(frame, f"RED (~{red_time}s) [{lane_total}v] {trend_arrow}",
                                (lx1 + 10, ly1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

            # ── Advance to highest-priority waiting lane when timer expires ──
            if elapsed >= signal_timer:
                signal_index = next_priority_lane()
                lane_last_green[signal_index] = time.time()
                signal_timer = calc_green_time(signal_index)
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
            shared_state["tailgating"]        = list(frame_tailgating)[:5]  # cap at 5
            shared_state["lane_predictions"]  = {str(k): round(predictor.predict(k), 1) for k in lane_counts}
            shared_state["ml_ready"]          = predictor.is_ready()
            shared_state["speed_limit"]       = SPEED_LIMIT_KMPH
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

if __name__ == "__main__":
    main()
