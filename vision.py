"""
Computer vision routines for the ball-balance maze.

All inputs are BGR numpy arrays (standard OpenCV layout). Picamera2 gives
you this directly if you configure main={"format": "RGB888"} — yes, the
name is misleading; "RGB888" in picamera2 lands in numpy as BGR.

Functions here are PURE: no state, no side effects. State lives in ai.py.
"""

import cv2
import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Red object detection
# ---------------------------------------------------------------------------

# Red wraps around hue 0/180 in HSV, so we use two ranges and OR them.
# Tune these if your lighting is warm/cool. Higher S and V = stricter match
# (rejects brownish table, pinkish skin, etc.).
RED_HSV_RANGES = [
    (np.array([0,   130, 70]),  np.array([10,  255, 255])),
    (np.array([170, 130, 70]),  np.array([180, 255, 255])),
]

MIN_RED_AREA_PX = 150   # reject small noise blobs


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in RED_HSV_RANGES:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_red_object(bgr):
    """
    Returns (cx, cy, radius, area) of the largest red blob, or None.
    Works for both a round red ball and the red triangle you're testing with.
    """
    mask = red_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_RED_AREA_PX:
        return None
    (x, y), r = cv2.minEnclosingCircle(c)
    return (int(x), int(y), int(r), float(area))


# ---------------------------------------------------------------------------
# Maze / path extraction
# ---------------------------------------------------------------------------

def extract_traversable_mask(bgr):
    """
    Binary mask where 255 = light channel (ball can be there),
    0 = dark walls / holes / markers. Adaptive threshold handles uneven
    lighting across the board.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=5,
    )
    # remove little speckles inside walls
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def _skeletonize(mask):
    """
    Prefer scikit-image's Zhang-Suen skeletonization (clean, well-connected).
    Fall back to a morphological skeleton if skimage isn't installed.
    """
    try:
        from skimage.morphology import skeletonize as sk_sk
        return (sk_sk(mask > 0).astype(np.uint8)) * 255
    except ImportError:
        # morphological skeleton (lower quality, but no extra dep)
        img = mask.copy()
        skel = np.zeros_like(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(img) > 0:
            eroded = cv2.erode(img, kernel)
            opened = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
            if cv2.countNonZero(img) == 0:
                break
        return skel


def _bfs_path(binary_graph, start, end, max_iter=500_000):
    """BFS on pixels. binary_graph: uint8 with nonzero = passable."""
    H, W = binary_graph.shape
    ys, xs = np.where(binary_graph > 0)
    if len(xs) == 0:
        return None
    pts = np.stack([xs, ys], axis=1)

    def snap(p):
        d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
        i = int(np.argmin(d))
        return int(pts[i, 0]), int(pts[i, 1])

    sx, sy = snap(start)
    ex, ey = snap(end)

    visited = np.zeros_like(binary_graph, dtype=bool)
    parent = {}
    q = deque([(sx, sy)])
    visited[sy, sx] = True

    found = False
    it = 0
    while q and it < max_iter:
        it += 1
        x, y = q.popleft()
        if (x, y) == (ex, ey):
            found = True
            break
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and binary_graph[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    parent[(nx, ny)] = (x, y)
                    q.append((nx, ny))

    if not found:
        return None
    path = [(ex, ey)]
    while path[-1] != (sx, sy):
        path.append(parent[path[-1]])
    path.reverse()
    # downsample (pixels are overkill — one every ~4px is plenty)
    return path[::4] if len(path) > 50 else path


def plan_path(bgr, start_xy, end_xy):
    """
    Full planning pipeline.
      1. Threshold → traversable mask (channels are white)
      2. Skeletonize → centerline of channels
      3. BFS on the skeleton from the click near start to the click near end
      4. If the skeleton has gaps, retry on a slightly dilated skeleton
      5. As a last resort, BFS directly on the traversable mask

    Returns (path_points, debug_mask) where path_points is a list of
    (x, y) tuples or None.
    """
    trav = extract_traversable_mask(bgr)
    skel = _skeletonize(trav)

    path = _bfs_path(skel, start_xy, end_xy)
    if path is None:
        # try bridging small gaps
        dil = cv2.dilate(skel, np.ones((3, 3), np.uint8), iterations=1)
        path = _bfs_path(dil, start_xy, end_xy)
    if path is None:
        # last resort: BFS across the full traversable area
        path = _bfs_path(trav, start_xy, end_xy)

    return path, skel


def lookahead_target(ball_pos, path, lookahead_px=50):
    """
    Given the ball's current (x, y) and the planned path, return a point
    `lookahead_px` pixels ahead of the ball along the path. This is the
    target the controller aims for — pure-pursuit style.
    """
    if not path:
        return None
    arr = np.array(path, dtype=np.float32)
    d = np.hypot(arr[:, 0] - ball_pos[0], arr[:, 1] - ball_pos[1])
    idx = int(np.argmin(d))
    acc = 0.0
    i = idx
    while i < len(path) - 1 and acc < lookahead_px:
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        acc += (dx * dx + dy * dy) ** 0.5
        i += 1
    return int(path[i][0]), int(path[i][1])


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def annotate(frame, ball=None, path=None, start=None, end=None,
             target=None, tilt=None, status_lines=None):
    out = frame.copy()

    if path and len(path) > 1:
        pts = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 255, 255), 2)

    if start is not None:
        cv2.drawMarker(out, tuple(start), (0, 255, 0),
                       markerType=cv2.MARKER_TRIANGLE_UP, markerSize=18, thickness=2)
    if end is not None:
        cv2.drawMarker(out, tuple(end), (0, 0, 255),
                       markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)

    if ball is not None:
        cx, cy, r, _ = ball
        cv2.circle(out, (cx, cy), max(r, 8), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)

    if target is not None:
        cv2.circle(out, tuple(target), 7, (255, 0, 255), 2)
        if ball is not None:
            cv2.line(out, (ball[0], ball[1]), tuple(target), (255, 0, 255), 1)

    if tilt is not None and ball is not None:
        tx, ty = tilt  # deviation from neutral, any units
        scale = 2.0
        end_pt = (int(ball[0] + tx * scale), int(ball[1] + ty * scale))
        cv2.arrowedLine(out, (ball[0], ball[1]), end_pt,
                        (255, 255, 0), 2, tipLength=0.3)

    if status_lines:
        for i, line in enumerate(status_lines):
            y = 25 + 22 * i
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out
