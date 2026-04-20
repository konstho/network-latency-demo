"""
AIController — orchestrates vision, control, and ESP32 output.

Detection backend is switchable at runtime:
  - "hsv":   classical color thresholding (vision.detect_red_object)
  - "hailo": YOLO on the AI HAT+ 2 (vision_hailo.HailoDetector)

Every incoming frame passes through process_frame(), which:
  1. Detects the ball (method depends on selected backend)
  2. Smooths position and estimates velocity
  3. If AI is running AND a path exists:
       - Picks a lookahead target on the path
       - Runs PD to compute a correction vector
       - Maps to 0..100 tilt range
       - (Optionally sleeps for injected latency)
       - Sends to ESP32 (or prints if no ESP32 connected)
  4. Returns annotated frame + telemetry dict for the UI
"""

import random
import time

import vision
import controller as ctrl
from esp32 import ESP32Comm

# vision_hailo is optional — catch import errors so the app still runs
# on dev machines without Hailo installed.
try:
    import vision_hailo
    HAILO_OK = True
except Exception as _e:
    vision_hailo = None
    HAILO_OK = False
    _hailo_err = _e


class AIController:
    def __init__(self, esp32_port=None, detector="hsv"):
        self.esp32 = ESP32Comm(port=esp32_port)
        self.esp32.connect()

        self.tracker = ctrl.BallTracker(alpha=0.55)
        self.pid = ctrl.PDController(kp=0.5, kd=0.08)

        self.running = False
        self.latency_ms = 0
        self.jitter_ms = 0

        # Detector state
        self.detector = "hsv"
        self._hailo = None
        self.last_inference_ms = 0.0

        # Planning state
        self.start_xy = None
        self.end_xy = None
        self.path = None
        self.last_skel = None

        # Tuning knobs
        self.tilt_gain = 0.25
        self.lookahead_px = 50

        # Telemetry
        self.last_ball = None
        self.last_target = None
        self.last_tilt = (50.0, 50.0)
        self.last_fps = 0.0
        self._last_frame_t = None

        self.set_detector(detector)

    # ----- detector -----
    def set_detector(self, name):
        name = (name or "hsv").lower()
        if name == "hailo":
            if not HAILO_OK:
                print(f"[AI] Hailo requested but not available: {_hailo_err}")
                print(f"[AI] Falling back to HSV")
                self.detector = "hsv"
                return False
            if self._hailo is None:
                try:
                    self._hailo = vision_hailo.HailoDetector()
                    self._hailo.open()
                except Exception as e:
                    print(f"[AI] Failed to initialize Hailo: {e}")
                    self.detector = "hsv"
                    return False
            self.detector = "hailo"
            print("[AI] Detector: Hailo (YOLO on AI HAT+ 2)")
            return True
        else:
            self.detector = "hsv"
            print("[AI] Detector: HSV (classical color threshold)")
            return True

    def shutdown(self):
        if self._hailo is not None:
            self._hailo.close()
            self._hailo = None

    # ----- configuration -----
    def set_start_end(self, start_xy, end_xy):
        self.start_xy = tuple(map(int, start_xy))
        self.end_xy = tuple(map(int, end_xy))

    def plan(self, bgr_frame):
        if self.start_xy is None or self.end_xy is None:
            return False, "set start and end points first"
        path, skel = vision.plan_path(bgr_frame, self.start_xy, self.end_xy)
        self.last_skel = skel
        if path is None:
            self.path = None
            return False, "no path found — try moving endpoints closer to the channels"
        self.path = path
        return True, f"path planned: {len(path)} points"

    def clear_plan(self):
        self.path = None
        self.start_xy = None
        self.end_xy = None

    def set_latency(self, latency_ms, jitter_ms):
        self.latency_ms = max(0, int(latency_ms))
        self.jitter_ms = max(0, int(jitter_ms))

    def set_gains(self, kp=None, kd=None, tilt_gain=None, lookahead_px=None):
        if kp is not None:           self.pid.kp = float(kp)
        if kd is not None:           self.pid.kd = float(kd)
        if tilt_gain is not None:    self.tilt_gain = float(tilt_gain)
        if lookahead_px is not None: self.lookahead_px = int(lookahead_px)

    # ----- runtime -----
    def start(self):
        self.running = True
        print("[AI] ON")

    def stop(self):
        self.running = False
        print("[AI] OFF")

    def _clamp_tilt(self, ux, uy):
        tx = 50.0 + max(-50.0, min(50.0, ux * self.tilt_gain))
        ty = 50.0 + max(-50.0, min(50.0, uy * self.tilt_gain))
        return tx, ty

    def _apply_latency(self):
        if self.latency_ms == 0 and self.jitter_ms == 0:
            return 0
        j = random.randint(-self.jitter_ms, self.jitter_ms) if self.jitter_ms > 0 else 0
        d = max(0, self.latency_ms + j)
        if d > 0:
            time.sleep(d / 1000.0)
        return d

    def _detect(self, bgr):
        if self.detector == "hailo" and self._hailo is not None:
            try:
                det = self._hailo.detect_ball(bgr)
                self.last_inference_ms = self._hailo.last_inference_ms
                return det
            except Exception as e:
                print(f"[AI] Hailo detect error: {e} — falling back to HSV once")
                self.last_inference_ms = 0.0
                return vision.detect_red_object(bgr)
        self.last_inference_ms = 0.0
        return vision.detect_red_object(bgr)

    def process_frame(self, bgr_frame):
        now = time.time()
        if self._last_frame_t is not None:
            dt = now - self._last_frame_t
            if dt > 0:
                inst = 1.0 / dt
                self.last_fps = 0.8 * self.last_fps + 0.2 * inst if self.last_fps else inst
        self._last_frame_t = now

        det = self._detect(bgr_frame)
        self.last_ball = det
        ball_pos = self.tracker.update(det)

        target = None
        tilt_cmd = None
        if self.running and ball_pos is not None and self.path:
            target = vision.lookahead_target(ball_pos, self.path, self.lookahead_px)
            self.last_target = target
            ux, uy = self.pid.update(ball_pos, self.tracker.vel, target)
            tilt_x, tilt_y = self._clamp_tilt(ux, uy)
            self.last_tilt = (tilt_x, tilt_y)
            self._apply_latency()
            self.esp32.send(int(tilt_x), int(tilt_y))
            tilt_cmd = (tilt_x, tilt_y)

        status = [
            f"AI: {'ON' if self.running else 'OFF'}   "
            f"DET: {self.detector.upper()}   "
            f"FPS: {self.last_fps:4.1f}",
            f"Ball: {'YES @ ('+str(det[0])+','+str(det[1])+')' if det else 'not detected'}",
            f"Path: {str(len(self.path))+' pts' if self.path else 'not planned'}",
        ]
        if self.detector == "hailo" and self.last_inference_ms > 0:
            status.append(f"Hailo inference: {self.last_inference_ms:.1f} ms")
        if tilt_cmd is not None:
            status.append(f"Tilt: x={tilt_cmd[0]:.1f}  y={tilt_cmd[1]:.1f}")
        if self.latency_ms:
            status.append(f"Injected lat {self.latency_ms}ms  jit +/-{self.jitter_ms}ms")

        tilt_vec = None
        if tilt_cmd is not None:
            tilt_vec = (tilt_cmd[0] - 50.0, tilt_cmd[1] - 50.0)

        annotated = vision.annotate(
            bgr_frame,
            ball=det,
            path=self.path,
            start=self.start_xy,
            end=self.end_xy,
            target=target,
            tilt=tilt_vec,
            status_lines=status,
        )
        return annotated, {
            "ball": det,
            "target": target,
            "tilt": tilt_cmd,
            "path_len": len(self.path) if self.path else 0,
            "fps": round(self.last_fps, 1),
            "detector": self.detector,
            "inference_ms": round(self.last_inference_ms, 1),
        }
