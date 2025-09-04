import cv2
import numpy as np
from collections import deque

class ShuttleTracker:
    """
    Motion-based shuttle candidate tracker using MOG2 foreground mask and contour filtering.
    Tracks a single fastest-small object centroid per frame.
    """
    def __init__(self, history=200, var_threshold=16, detect_shadows=False, max_trace=32):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.trace = deque(maxlen=max_trace)
        self.prev = None

    def update(self, frame_gray):
        fg = self.bg.apply(frame_gray)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5 or area > 200:  # shuttle is tiny
                continue
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            centroid = np.array([x, y], dtype=np.float32)

            speed = 0.0
            if self.prev is not None:
                speed = np.linalg.norm(centroid - self.prev)

            score = speed / (area + 1e-3)  # prefer fast & small
            if score > best_score:
                best_score = score
                candidate = centroid

        if candidate is not None:
            self.trace.append(candidate)
            self.prev = candidate
        else:
            self.prev = None

        return candidate, list(self.trace)

    @staticmethod
    def speed(trace, window=5):
        if len(trace) < window + 1:
            return 0.0
        p1 = np.array(trace[-1])
        p0 = np.array(trace[-1 - window])
        return float(np.linalg.norm(p1 - p0) / window)
