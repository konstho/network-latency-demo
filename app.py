"""
Flask server for the ball-balance maze.

Key fixes from the old version:
  * picamera2 configured with format="RGB888" so frames arrive in BGR
    (otherwise OpenCV calls produce silent garbage)
  * AI is actually wired to the camera frames — process_frame() runs
    in a dedicated loop and produces the annotated frame that the stream serves
  * frame sharing is properly locked
  * new endpoints:
      POST /set_start_end     set path endpoints in image pixel coords
      POST /plan_path         run path planning on the current frame
      POST /set_gains         tune kp/kd/tilt_gain/lookahead at runtime
      POST /clear_plan        forget planned path
  * backward-compatible with your existing /update, /settings, /ai, /status
  * new page at /ai_test for AI-specific testing (your / page is untouched)
"""

import random
import threading
import time

import cv2
from flask import Flask, render_template, request, jsonify, Response

from picamera2 import Picamera2
from ai import AIController


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Set esp32_port="/dev/ttyUSB0" (or wherever) once you plug the ESP32 in.
ai_controller = AIController(esp32_port=None)

state = {"latency": 0, "jitter": 0, "x": 50, "y": 50, "ai_mode": False}

FRAME_W, FRAME_H = 640, 480

picam = Picamera2()
picam.configure(picam.create_video_configuration(
    main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},   # → BGR numpy
    controls={"FrameRate": 30},
))
picam.start()
time.sleep(0.5)  # let auto-exposure settle

frame_lock = threading.Lock()
raw_frame = None        # latest frame from camera (BGR)
display_frame = None    # latest annotated frame for the stream


def _capture_loop():
    global raw_frame
    while True:
        f = picam.capture_array()
        with frame_lock:
            raw_frame = f
        time.sleep(1 / 60)   # sleep a bit, don't pin a core


def _ai_loop():
    """Pulls the latest raw frame, runs AI, writes the annotated frame."""
    global display_frame
    while True:
        with frame_lock:
            f = None if raw_frame is None else raw_frame.copy()
        if f is None:
            time.sleep(0.02)
            continue
        annotated, _ = ai_controller.process_frame(f)
        with frame_lock:
            display_frame = annotated
        # target ~30 Hz processing
        time.sleep(1 / 30)


threading.Thread(target=_capture_loop, daemon=True).start()
threading.Thread(target=_ai_loop, daemon=True).start()


def _mjpeg_stream():
    while True:
        with frame_lock:
            f = None if display_frame is None else display_frame
            buf = None
            if f is not None:
                ok, encoded = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    buf = encoded.tobytes()
        if buf is None:
            time.sleep(0.03)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
        time.sleep(1 / 30)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    # your existing UI — leave as-is
    return render_template("index.html")


@app.route("/ai_test")
def ai_test():
    return render_template("ai_test.html")


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# --- manual joystick control (original) ---
@app.route("/update", methods=["POST"])
def update():
    if state["ai_mode"]:
        return jsonify({"status": "ai_mode_active"})
    data = request.get_json() or {}
    jit = random.randint(-state["jitter"], state["jitter"]) if state["jitter"] > 0 else 0
    delay = max(0, state["latency"] + jit)
    if delay > 0:
        time.sleep(delay / 1000.0)
    state["x"] = data.get("x", 50)
    state["y"] = data.get("y", 50)
    ai_controller.esp32.send(int(state["x"]), int(state["y"]))
    return jsonify({"status": "ok"})


@app.route("/settings", methods=["POST"])
def settings():
    data = request.get_json() or {}
    state["latency"] = int(data.get("latency", 0))
    state["jitter"]  = int(data.get("jitter", 0))
    ai_controller.set_latency(state["latency"], state["jitter"])
    return jsonify({"status": "ok"})


# --- AI control ---
@app.route("/ai", methods=["POST"])
def toggle_ai():
    data = request.get_json() or {}
    enable = bool(data.get("enable", False))
    if enable and ai_controller.path is None:
        return jsonify({
            "status": "error",
            "message": "Plan a path first (mark start and end, then Plan Path)",
        }), 400
    if enable:
        ai_controller.start()
    else:
        ai_controller.stop()
    state["ai_mode"] = enable
    return jsonify({"status": "ok", "ai_mode": enable})


@app.route("/set_start_end", methods=["POST"])
def set_start_end():
    data = request.get_json() or {}
    start = data.get("start")
    end   = data.get("end")
    if not start or not end:
        return jsonify({"status": "error", "message": "need 'start' and 'end' [x, y]"}), 400
    ai_controller.set_start_end(start, end)
    return jsonify({"status": "ok"})


@app.route("/plan_path", methods=["POST"])
def plan_path_route():
    with frame_lock:
        f = None if raw_frame is None else raw_frame.copy()
    if f is None:
        return jsonify({"status": "error", "message": "no camera frame yet"}), 503
    ok, msg = ai_controller.plan(f)
    return jsonify({"status": "ok" if ok else "error", "message": msg})


@app.route("/clear_plan", methods=["POST"])
def clear_plan():
    ai_controller.clear_plan()
    return jsonify({"status": "ok"})


@app.route("/set_gains", methods=["POST"])
def set_gains():
    data = request.get_json() or {}
    ai_controller.set_gains(
        kp=data.get("kp"),
        kd=data.get("kd"),
        tilt_gain=data.get("tilt_gain"),
        lookahead_px=data.get("lookahead_px"),
    )
    return jsonify({
        "status": "ok",
        "kp": ai_controller.pid.kp,
        "kd": ai_controller.pid.kd,
        "tilt_gain": ai_controller.tilt_gain,
        "lookahead_px": ai_controller.lookahead_px,
    })


@app.route("/set_detector", methods=["POST"])
def set_detector():
    data = request.get_json() or {}
    name = data.get("detector", "hsv")
    ok = ai_controller.set_detector(name)
    return jsonify({
        "status": "ok" if ok else "fallback",
        "detector": ai_controller.detector,
    })


@app.route("/status")
def status():
    return jsonify({
        **state,
        "ball": ai_controller.last_ball,
        "target": ai_controller.last_target,
        "tilt": ai_controller.last_tilt,
        "path_len": len(ai_controller.path) if ai_controller.path else 0,
        "fps": round(ai_controller.last_fps, 1),
        "detector": ai_controller.detector,
        "inference_ms": round(ai_controller.last_inference_ms, 1),
        "gains": {
            "kp": ai_controller.pid.kp,
            "kd": ai_controller.pid.kd,
            "tilt_gain": ai_controller.tilt_gain,
            "lookahead_px": ai_controller.lookahead_px,
        },
    })


if __name__ == "__main__":
    # threaded=True is important so /video_feed and /status can overlap
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
