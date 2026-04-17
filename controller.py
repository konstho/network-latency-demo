"""
Ball tracking (smoothing + velocity estimation) and PD controller.

Keeping these separate from vision.py because they hold state between frames.
"""

import time


class BallTracker:
    """
    Exponential moving average over the raw detections, plus a simple
    velocity estimate based on the smoothed position.

    alpha closer to 1 = more responsive, more jittery.
    alpha closer to 0 = smoother, more laggy.
    """

    def __init__(self, alpha=0.55):
        self.alpha = alpha
        self.pos = None           # (x, y) floats
        self.vel = (0.0, 0.0)     # px/s
        self._last_t = None
        self._miss_count = 0

    def update(self, detection):
        now = time.time()
        if detection is None:
            self._miss_count += 1
            # after ~10 consecutive misses, forget the ball
            if self._miss_count > 10:
                self.pos = None
                self.vel = (0.0, 0.0)
            return self.pos

        self._miss_count = 0
        px, py = float(detection[0]), float(detection[1])

        if self.pos is None or self._last_t is None:
            self.pos = (px, py)
            self.vel = (0.0, 0.0)
            self._last_t = now
            return self.pos

        dt = max(1e-3, now - self._last_t)
        nx = self.alpha * px + (1 - self.alpha) * self.pos[0]
        ny = self.alpha * py + (1 - self.alpha) * self.pos[1]
        self.vel = ((nx - self.pos[0]) / dt, (ny - self.pos[1]) / dt)
        self.pos = (nx, ny)
        self._last_t = now
        return self.pos


class PDController:
    """
    Classic PD in pixel units. Output is a 2D vector (dx, dy) with the same
    units as position error — a separate mapping step converts it to the
    0..100 tilt range the web UI / ESP32 expects.

    kp: stiffness. Too high → oscillation.
    kd: damping. Velocity feedback. Helps kill oscillations.

    Start with kp=0.5, kd=0.08 and tune once you have the real ball.
    """

    def __init__(self, kp=0.5, kd=0.08):
        self.kp = kp
        self.kd = kd

    def update(self, ball_pos, ball_vel, target):
        if ball_pos is None or target is None:
            return 0.0, 0.0
        ex = target[0] - ball_pos[0]
        ey = target[1] - ball_pos[1]
        ux = self.kp * ex - self.kd * ball_vel[0]
        uy = self.kp * ey - self.kd * ball_vel[1]
        return ux, uy
