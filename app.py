from flask import Flask, render_template, request, jsonify
import time
import random
import threading
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import io


app = Flask(__name__)

# Try to import AI controller
try:
    from ai import AIController
    ai_controller = AIController()
    AI_AVAILABLE = True
except Exception as e:
    print(f"AI not available: {e}")
    AI_AVAILABLE = False

current_values = {
    "x": 50,
    "y": 50,
    "latency": 0,
    "jitter": 0,
    "ai_mode": False
}


camera = Picamera2()
camera.configure(camera.create_video_configuration(main={"size": (640, 480)}))

output_frame = None
lock = threading.Lock()

def capture_frames():
    global output_frame
    camera.start()
    while True:
        frame = camera.capture_array()
        with lock:
            output_frame = frame.copy()

threading.Thread(target=capture_frames, daemon=True).start()

def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            import cv2
            _, buffer = cv2.imencode(".jpg", output_frame)
            frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    from flask import Response
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/update", methods=["POST"])
def update():
    if current_values["ai_mode"]:
        return jsonify({"status": "ai_mode_active"})

    data = request.get_json()
    latency = int(current_values["latency"])
    jitter = int(current_values["jitter"])

    jitter_amount = random.randint(-jitter, jitter) if jitter > 0 else 0
    actual_delay = max(0, latency + jitter_amount)
    if actual_delay > 0:
        time.sleep(actual_delay / 1000)

    current_values["x"] = data.get("x", 50)
    current_values["y"] = data.get("y", 50)

    print(f"X: {data.get('x')}, Y: {data.get('y')}, Latency: {latency}ms, Jitter: ±{jitter}ms")
    return jsonify({"status": "ok"})

@app.route("/settings", methods=["POST"])
def settings():
    data = request.get_json()
    current_values["latency"] = data.get("latency", 0)
    current_values["jitter"] = data.get("jitter", 0)

    if AI_AVAILABLE:
        ai_controller.set_latency(
            current_values["latency"],
            current_values["jitter"]
        )

    print(f"Latency: {current_values['latency']}ms, Jitter: ±{current_values['jitter']}ms")
    return jsonify({"status": "ok"})

@app.route("/ai", methods=["POST"])
def toggle_ai():
    if not AI_AVAILABLE:
        return jsonify({"status": "error", "message": "AI not available"})

    data = request.get_json()
    enable = data.get("enable", False)

    if enable:
        ai_controller.start()
        current_values["ai_mode"] = True
        print("AI mode ON")
    else:
        ai_controller.stop()
        current_values["ai_mode"] = False
        print("AI mode OFF")

    return jsonify({"status": "ok", "ai_mode": current_values["ai_mode"]})


@app.route("/status")
def status():
    return jsonify(current_values)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
