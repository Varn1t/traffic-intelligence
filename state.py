import threading
from collections import deque
from datetime import datetime

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
    "lane_predictions": {},  # {lane_id_str: predicted_count_in_15s}
    "ml_ready":      False,  # True once RF model has trained
    "speed_limit":   None,   # expose config for dashboard
}
state_lock = threading.Lock()

history_buf = deque(maxlen=40)

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
