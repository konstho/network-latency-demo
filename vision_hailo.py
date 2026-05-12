"""
Hailo-accelerated ball detector using a CUSTOM-trained YOLOv8n model.

Same public API as the previous COCO-based version:
    det.detect_ball(bgr_frame)  # -> (cx, cy, radius, area) or None

Key differences from COCO version:
- Custom model: 1 class (ball), trained on ~85 maze images
- Cut before NMS: model outputs 6 raw conv tensors, we do decoding+NMS in Python
- HEF compiled at /home/konsta/maze_control/ball_model.hef
"""

import cv2
import numpy as np
import time

try:
    from picamera2.devices import Hailo
    HAILO_AVAILABLE = True
except Exception as _e:
    HAILO_AVAILABLE = False
    _hailo_import_error = _e


DEFAULT_HEF = "/home/konsta/maze_control/ball_model.hef"


class HailoDetector:
    """
    Custom YOLOv8n model with 1 class (ball).
    Outputs raw conv tensors; we decode + NMS in Python.
    """

    def __init__(self, hef_path=DEFAULT_HEF, score_threshold=0.25,
                 iou_threshold=0.45):
        if not HAILO_AVAILABLE:
            raise RuntimeError(
                f"picamera2.devices.Hailo not importable: {_hailo_import_error}"
            )
        self.hef_path = hef_path
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self._hailo = None
        self._in_h = None
        self._in_w = None

        self.last_inference_ms = 0.0
        self.last_raw_detections = []
        self.last_best = None

    def open(self):
        if self._hailo is not None:
            return
        self._hailo = Hailo(self.hef_path)
        self._hailo.__enter__()
        self._in_h, self._in_w, _ = self._hailo.get_input_shape()
        print(f"[Hailo] loaded {self.hef_path}  input={self._in_w}x{self._in_h}")

    def close(self):
        if self._hailo is not None:
            hailo = self._hailo
            self._hailo = None
            try:
                hailo.__exit__(None, None, None)
            except Exception as e:
                print(f"[Hailo] close error: {e}")

    def _preprocess(self, bgr):
        h, w = bgr.shape[:2]
        target = self._in_w
        scale = min(target / w, target / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return rgb, scale, (new_w, new_h)

    def _decode_yolov8(self, outputs, num_classes=1):
        """
        Decode raw YOLOv8 head outputs.
        outputs is a dict with 6 tensors of shape (H, W, C):
          conv41 (80,80,64), conv42 (80,80,1)  - stride 8
          conv52 (40,40,64), conv53 (40,40,1)  - stride 16
          conv62 (20,20,64), conv63 (20,20,1)  - stride 32
        """
        if isinstance(outputs, dict):
            outputs = list(outputs.values())

        regs = sorted([o for o in outputs if o.shape[-1] == 64],
                      key=lambda x: -x.shape[0])
        clss = sorted([o for o in outputs if o.shape[-1] == num_classes],
                      key=lambda x: -x.shape[0])

        all_boxes = []
        all_scores = []
        strides = [8, 16, 32]

        for reg, cls, stride in zip(regs, clss, strides):
            H, W = reg.shape[0], reg.shape[1]
            reg = reg.reshape(H * W, 4, 16)
            reg_exp = np.exp(reg - reg.max(axis=-1, keepdims=True))
            reg_softmax = reg_exp / reg_exp.sum(axis=-1, keepdims=True)
            bins = np.arange(16, dtype=np.float32)
            distances = (reg_softmax * bins).sum(axis=-1)

            yv, xv = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            grid = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32) + 0.5
            grid_xy = grid * stride

            x0 = grid_xy[:, 0] - distances[:, 0] * stride
            y0 = grid_xy[:, 1] - distances[:, 1] * stride
            x1 = grid_xy[:, 0] + distances[:, 2] * stride
            y1 = grid_xy[:, 1] + distances[:, 3] * stride
            boxes = np.stack([x0, y0, x1, y1], axis=-1)

            scores = 1.0 / (1.0 + np.exp(-cls.reshape(H * W, num_classes)))

            all_boxes.append(boxes)
            all_scores.append(scores)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        return boxes, scores

    def _nms(self, boxes, scores, iou_threshold):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][ovr <= iou_threshold]
        return keep

    def detect_ball(self, bgr_frame):
        if self._hailo is None:
            self.open()
        rgb, scale, padded_size = self._preprocess(bgr_frame)
        t0 = time.time()
        outputs = self._hailo.run(rgb)
        self.last_inference_ms = (time.time() - t0) * 1000.0

        # Decode YOLO outputs
        boxes, scores = self._decode_yolov8(outputs, num_classes=1)
        scores_flat = scores[:, 0]  # only one class
        mask = scores_flat >= self.score_threshold
        boxes = boxes[mask]
        scores_flat = scores_flat[mask]

        if len(boxes) == 0:
            self.last_best = None
            return None

        # NMS
        keep = self._nms(boxes, scores_flat, self.iou_threshold)
        if not keep:
            self.last_best = None
            return None

        # Best detection (highest score)
        best_idx = keep[0]
        x0, y0, x1, y1 = boxes[best_idx]
        score = float(scores_flat[best_idx])

        # Undo letterbox: scale back to original frame
        x0, y0, x1, y1 = x0 / scale, y0 / scale, x1 / scale, y1 / scale
        orig_h, orig_w = bgr_frame.shape[:2]
        x0 = max(0, min(orig_w - 1, x0))
        x1 = max(0, min(orig_w - 1, x1))
        y0 = max(0, min(orig_h - 1, y0))
        y1 = max(0, min(orig_h - 1, y1))

        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)
        radius = int(max(4, min(x1 - x0, y1 - y0) / 2))
        area = float((x1 - x0) * (y1 - y0))
        self.last_best = {"bbox": (x0, y0, x1, y1), "score": score}
        return (cx, cy, radius, area)


def detect_ball_hailo(bgr_frame, hailo_ctx):
    return hailo_ctx.detect_ball(bgr_frame)