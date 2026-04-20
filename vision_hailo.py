"""
Hailo-accelerated ball detector.

Same public API as vision.detect_red_object() so ai.py can swap detectors
without caring which is in use:

    det = detect_ball_hailo(bgr_frame, hailo_ctx)
    # -> (cx, cy, radius, area) or None

The hailo_ctx is a HailoDetector instance created once at startup.
The detector is stateful (holds the loaded model + input buffer), so we
wrap it in a class instead of a module-level singleton — makes it easy
to shut down cleanly when the app stops.

Why this over HSV color thresholding:
- Works with the stock metal ball (no painting/taping needed)
- Robust to lighting changes (the Nokia demo space likely has different
  lighting than our dev setup)
- Rejects red distractor objects that would confuse HSV detection

Runs at ~30 FPS on the Hailo-10H (measured: 34ms per 640x640 inference).
"""

import cv2
import numpy as np

try:
    from picamera2.devices import Hailo
    HAILO_AVAILABLE = True
except Exception as _e:
    HAILO_AVAILABLE = False
    _hailo_import_error = _e


# COCO class indices we care about for "ball-like" objects.
# YOLOv8 is trained on COCO. The metal ball looks most like "sports ball"
# but we accept a few other small-round classes as fallbacks in case
# the model mislabels under this specific lighting/shape.
BALL_CLASS_IDS = {
    32,   # sports ball  <- primary
    29,   # frisbee      (small round-ish object)
    # Anything else we want to accept? Add IDs here. Don't add too many,
    # or we'll detect cups/donuts and steer toward them.
}
BALL_CLASS_NAMES = {
    0: "person", 29: "frisbee", 32: "sports ball", 39: "bottle",
    41: "cup", 44: "spoon", 47: "apple", 49: "orange", 54: "donut",
    74: "clock", 73: "book",
}  # for debug logging only

DEFAULT_HEF = "/usr/share/hailo-models/yolov8m_h10.hef"


class HailoDetector:
    """
    Loads the YOLO HEF once and runs inference on BGR frames.

    Usage:
        det = HailoDetector()
        det.open()
        result = det.detect_ball(bgr_frame)   # -> (cx, cy, r, area) or None
        det.close()
    """

    def __init__(self, hef_path=DEFAULT_HEF, score_threshold=0.25,
                 accepted_classes=None):
        if not HAILO_AVAILABLE:
            raise RuntimeError(
                f"picamera2.devices.Hailo not importable: {_hailo_import_error}"
            )
        self.hef_path = hef_path
        self.score_threshold = score_threshold
        self.accepted_classes = (
            set(accepted_classes) if accepted_classes else set(BALL_CLASS_IDS)
        )
        self._hailo = None
        self._in_h = None
        self._in_w = None

        # Telemetry
        self.last_inference_ms = 0.0
        self.last_raw_detections = []   # list of (class_id, score, box_px)
        self.last_best = None           # picked-as-ball detection

    def open(self):
        if self._hailo is not None:
            return
        # Hailo() is a context manager but we also manually drive it so
        # it can live for the whole app lifetime.
        self._hailo = Hailo(self.hef_path)
        self._hailo.__enter__()
        self._in_h, self._in_w, _ = self._hailo.get_input_shape()
        print(f"[Hailo] loaded {self.hef_path}  input={self._in_w}x{self._in_h}")

    def close(self):
        if self._hailo is not None:
            try:
                self._hailo.__exit__(None, None, None)
            except Exception as e:
                print(f"[Hailo] close error: {e}")
            self._hailo = None

    # ---------- internal: pre/post-process ----------
    def _preprocess(self, bgr):
        """
        The YOLO HEF expects a 640x640 RGB image. Our frames come in at
        whatever size the camera was configured to, as BGR.

        We resize-with-letterbox to preserve aspect ratio. Without the
        letterbox, distortion hurts small-object accuracy.
        """
        h, w = bgr.shape[:2]
        target = self._in_w   # model is square
        scale = min(target / w, target / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # paste into 640x640 canvas, top-left
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return rgb, scale, (new_w, new_h)

    def _postprocess(self, results, scale, padded_size, orig_shape):
        """
        results: list of 80 lists (one per COCO class).
          Each entry is an array of detections with format [y0, x0, y1, x1, score]
          in NORMALIZED coords (0..1) relative to the 640x640 input.

        We convert normalized -> padded pixels -> original frame pixels,
        undoing the letterbox scale.
        """
        orig_h, orig_w = orig_shape[:2]
        pad_w, pad_h = padded_size
        raw = []
        for cls_id, dets in enumerate(results):
            if dets is None:
                continue
            # dets might be a list/array; be defensive
            try:
                n = len(dets)
            except TypeError:
                continue
            for d in dets:
                if len(d) < 5:
                    continue
                y0, x0, y1, x1, score = float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])
                if score < self.score_threshold:
                    continue
                # normalized 0..1 in 640x640 letterboxed canvas
                # → pixel coords in the 640x640 canvas
                X0 = x0 * self._in_w
                X1 = x1 * self._in_w
                Y0 = y0 * self._in_h
                Y1 = y1 * self._in_h
                # undo letterbox: the image was pasted at (0,0) scaled by `scale`
                bx0 = X0 / scale
                by0 = Y0 / scale
                bx1 = X1 / scale
                by1 = Y1 / scale
                # clamp
                bx0 = max(0.0, min(orig_w - 1, bx0))
                bx1 = max(0.0, min(orig_w - 1, bx1))
                by0 = max(0.0, min(orig_h - 1, by0))
                by1 = max(0.0, min(orig_h - 1, by1))
                raw.append({
                    "class_id": cls_id,
                    "score": score,
                    "bbox": (bx0, by0, bx1, by1),
                })
        return raw

    # ---------- public ----------
    def detect_all(self, bgr_frame):
        """Return every detection above threshold (any class)."""
        if self._hailo is None:
            self.open()
        import time
        rgb, scale, padded_size = self._preprocess(bgr_frame)
        t0 = time.time()
        results = self._hailo.run(rgb)
        self.last_inference_ms = (time.time() - t0) * 1000.0
        raw = self._postprocess(results, scale, padded_size, bgr_frame.shape)
        self.last_raw_detections = raw
        return raw

    def detect_ball(self, bgr_frame):
        """
        Public API matching vision.detect_red_object().

        Runs YOLO, filters to ball-like classes, returns the highest-confidence
        match as (cx, cy, radius, area_proxy) — same shape as the HSV detector
        so the rest of the pipeline doesn't need to know which was used.
        """
        all_dets = self.detect_all(bgr_frame)
        candidates = [d for d in all_dets if d["class_id"] in self.accepted_classes]
        if not candidates:
            self.last_best = None
            return None
        best = max(candidates, key=lambda d: d["score"])
        self.last_best = best
        x0, y0, x1, y1 = best["bbox"]
        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)
        # approximate ball radius as half the shortest bbox side
        radius = int(max(4, min(x1 - x0, y1 - y0) / 2))
        area = float((x1 - x0) * (y1 - y0))
        return (cx, cy, radius, area)


# Convenience module-level wrapper so callers can do
#   from vision_hailo import detect_ball_hailo
def detect_ball_hailo(bgr_frame, hailo_ctx):
    return hailo_ctx.detect_ball(bgr_frame)
