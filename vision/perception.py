"""Runs only the detectors the current puzzle needs and draws what the AI sees."""
import cv2

import config
from vision.detectors import ColorDetector, HandTracker, ObjectDetector
from vision.observation import Observation

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                    (17, 18), (18, 19), (19, 20)]

BGR = {"red": (40, 40, 230), "blue": (230, 120, 40), "green": (60, 200, 60), "yellow": (40, 220, 240)}


class Perception:
    def __init__(self, use_yolo: bool = True, model_path: str = config.YOLO_MODEL):
        self.colors = ColorDetector()
        self.hands = HandTracker()
        self.objects = None
        if use_yolo:
            print("Loading YOLOv8 model (first run downloads it)...")
            self.objects = ObjectDetector(model_path)
        self._frame_no = 0
        self._last_objects = []

    def process(self, frame, need_objects=True, need_hands=True, need_colors=True, draw=True):
        self._frame_no += 1
        h, w = frame.shape[:2]
        obs = Observation(frame_size=(w, h))

        if need_objects and self.objects is not None:
            # YOLO is the slowest part, so run it every N frames and reuse results in between
            if self._frame_no % config.YOLO_EVERY_N_FRAMES == 0 or not self._last_objects:
                self._last_objects = self.objects.detect(frame)
            obs.objects = self._last_objects
        else:
            self._last_objects = []
        if need_hands:
            obs.hands = self.hands.detect(frame)
        if need_colors:
            obs.colors = self.colors.detect(frame)

        annotated = self.draw(frame.copy(), obs) if draw else frame
        return obs, annotated

    @staticmethod
    def draw(img, obs: Observation):
        for d in obs.objects:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 120), 2)
            _label(img, f"{d.label} {d.confidence:.2f}", x1, y1, (0, 220, 120))
        for name, blob in obs.colors.items():
            x1, y1, x2, y2 = blob.box
            col = BGR.get(name, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            _label(img, f"{name} object", x1, y2 + 22, col)
        for hand in obs.hands:
            pts = [(int(x), int(y)) for x, y in hand.landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], (255, 255, 255), 2)
            for i, p in enumerate(pts):
                tip_up = i in (4, 8, 12, 16, 20) and hand.fingers[(4, 8, 12, 16, 20).index(i)]
                cv2.circle(img, p, 6 if tip_up else 4, (0, 255, 255) if tip_up else (255, 80, 200), -1)
            wx, wy = pts[0]
            _label(img, f"{hand.gesture.replace('_', ' ')} | {hand.finger_count} up", wx - 40, wy + 30,
                   (255, 80, 200))
        return img

    def close(self):
        self.hands.close()


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
