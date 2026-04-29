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
