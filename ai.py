import cv2
import numpy as np
import time
import random
from esp32 import ESP32Comm

class AIController:
    def __init__(self):
        self.esp32 = ESP32Comm()
        self.esp32.connect()
        self.latency = 0
        self.jitter = 0
        self.running = False
        self.x = 50
        self.y = 50

    def set_latency(self, latency, jitter):
        self.latency = latency
        self.jitter = jitter

    def find_ball(self, frame):
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color range
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Find ball position
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return cx, cy
        return None

    def find_line(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold to find black line
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Find line contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
        return None

    def decide_tilt(self, ball_pos, line_pos, frame_width, frame_height):
        if ball_pos is None or line_pos is None:
            return self.x, self.y

        ball_x, ball_y = ball_pos
        line_x, line_y = line_pos

        # Calculate error between ball and line
        error_x = ball_x - line_x
        error_y = ball_y - line_y

        # Adjust tilt based on error
        sensitivity = 0.1
        self.x = max(0, min(100, self.x - error_x * sensitivity))
        self.y = max(0, min(100, self.y - error_y * sensitivity))

        return self.x, self.y

    def apply_delay(self):
        jitter_amount = random.randint(-self.jitter, self.jitter) if self.jitter > 0 else 0
        actual_delay = max(0, self.latency + jitter_amount)
        if actual_delay > 0:
            time.sleep(actual_delay / 1000)
        return actual_delay

    def run(self, frame):
        if not self.running:
            return None, None

        ball_pos = self.find_ball(frame)
        line_pos = self.find_line(frame)

        tilt_x, tilt_y = self.decide_tilt(
            ball_pos, line_pos,
            frame.shape[1], frame.shape[0]
        )

        actual_delay = self.apply_delay()
        self.esp32.send(int(tilt_x), int(tilt_y))

        return ball_pos, actual_delay

    def start(self):
        self.running = True
        print("AI controller started")

    def stop(self):
        self.running = False
        print("AI controller stopped")
