import csv
from datetime import datetime

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
