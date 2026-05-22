import os
import secrets
import logging
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, jsonify, render_template_string, session, request, redirect, url_for
from config import FLASK_PORT
from state import shared_state, state_lock, history_buf, session_stats
from datetime import datetime
from templates import DASHBOARD_HTML, LANDING_HTML

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(24))

DASH_USER = os.environ.get("DASH_USER", "admin")
DASH_PASS = os.environ.get("DASH_PASS", "changeme")

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

@app.route("/api/summary")
def api_summary():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    start_str = session_stats.session_start
    try:
        start_dt = datetime.fromisoformat(start_str)
        elapsed = int((datetime.now() - start_dt).total_seconds())
        hours, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        duration = f"{hours:02d}:{mins:02d}:{secs:02d}"
    except Exception:
        duration = "—"
    return jsonify({
        "duration":        duration,
        "total_vehicles":  len(session_stats.all_ids),
        "peak_count":      session_stats.peak_count,
        "peak_time":       session_stats.peak_time,
        "total_incidents": session_stats.total_incidents,
        "speeders":        session_stats.speeders_count,
        "wrong_way":       len(session_stats.wrong_way_ids),
        "tailgate_events": session_stats.tailgate_events,
        "session_start":   start_str,
        "session_ended":   session_stats.session_ended,
    })

@app.route("/api/control/update", methods=["POST"])
def update_controls():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json() or {}
    with state_lock:
        if "mode" in data:
            shared_state["mode"] = str(data["mode"])
        if "speed_limit" in data:
            shared_state["speed_limit"] = int(data["speed_limit"])
        if "incident_timeout" in data:
            shared_state["incident_timeout"] = float(data["incident_timeout"])
        if "conf_threshold" in data:
            shared_state["conf_threshold"] = float(data["conf_threshold"])
    return jsonify({"status": "success", "settings": data})

def run_flask():
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)  # silence Flask request logs
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)
