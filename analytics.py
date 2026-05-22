import time
from math import hypot
from collections import defaultdict, deque
from typing import TypedDict
from config import INCIDENT_TIMEOUT, INCIDENT_DIST_PX

try:
    from sklearn.ensemble import RandomForestRegressor as _RF
    _SKLEARN_OK = True
except ImportError:
    _RF = None
    _SKLEARN_OK = False
    print("[WARN] scikit-learn not found — ML prediction disabled. pip install scikit-learn")

# ──────────────────────────────────────────────
# INCIDENT DETECTOR
# ──────────────────────────────────────────────
class _IncidentEntry(TypedDict):
    pos: tuple[int, int]
    still_since: float
    lane: int

class IncidentDetector:
    """Flags vehicles that haven't moved for INCIDENT_TIMEOUT seconds."""
    def __init__(self):
        self.history: dict[int, _IncidentEntry] = {}  # track_id → entry

    def update(self, track_id: int, cx: int, cy: int, lane_id: int, timeout: float = None) -> bool:
        now = time.time()
        effective_timeout = timeout if timeout is not None else INCIDENT_TIMEOUT
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
            return still_for >= effective_timeout

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
# TRAFFIC PREDICTOR  (online Random Forest)
# ──────────────────────────────────────────────
class TrafficPredictor:
    """Per-lane Random-Forest regressor that predicts future vehicle counts.

    Every call to update() ingests the current count for a lane.
    Once MIN_SAMPLES are available a RF model is trained whose features are
    the last WINDOW observations and whose target is the count HORIZON steps
    ahead (≈ HORIZON * 3 seconds with 3-second snapshots).

    predict(lane_id) -> estimated future count (falls back to last known while warming up).
    """
    WINDOW        = 10   # past observations used as features
    HORIZON       = 5    # predict this many steps ahead  (~15 s)
    MIN_SAMPLES   = 20   # need this many before first fit
    RETRAIN_EVERY = 5    # retrain after N new samples

    def __init__(self, lane_ids: list):
        self.lane_ids = list(lane_ids)
        self.history: dict = defaultdict(lambda: deque(maxlen=200))
        self.models:  dict = {}                      # lane_id -> fitted RF
        self._since:  dict = defaultdict(int)        # samples since last retrain
        self._preds:  dict = {lid: 0.0 for lid in lane_ids}
        self._ready:  bool = False

    # ── public ──────────────────────────────────────────────
    def update(self, lane_id: int, count: int) -> None:
        self.history[lane_id].append(float(count))
        self._since[lane_id] += 1
        if (len(self.history[lane_id]) >= self.WINDOW + self.HORIZON
                and self._since[lane_id] >= self.RETRAIN_EVERY):
            self._retrain(lane_id)
            self._since[lane_id] = 0
        self._preds[lane_id] = self._infer(lane_id)

    def predict(self, lane_id: int) -> float:
        return max(0.0, self._preds.get(lane_id, 0.0))

    def is_ready(self) -> bool:
        return self._ready

    # ── private ─────────────────────────────────────────────
    def _retrain(self, lane_id: int) -> None:
        if not _SKLEARN_OK:
            return
        h = list(self.history[lane_id])
        X, y = [], []
        for i in range(len(h) - self.WINDOW - self.HORIZON + 1):
            X.append(h[i: i + self.WINDOW])
            y.append(h[i + self.WINDOW + self.HORIZON - 1])
        if len(X) < 10:
            return
        model = _RF(n_estimators=40, max_depth=5, random_state=42, n_jobs=1)
        model.fit(X, y)
        self.models[lane_id] = model
        self._ready = True

    def _infer(self, lane_id: int) -> float:
        h = list(self.history[lane_id])
        if lane_id not in self.models or len(h) < self.WINDOW:
            return float(h[-1]) if h else 0.0   # warm-up: last known
        return float(self.models[lane_id].predict([h[-self.WINDOW:]])[0])

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

def get_sklearn_ok():
    return _SKLEARN_OK
