"""Wrappers around YOLOv8, MediaPipe Hands and HSV colour segmentation.

Heavy libraries are imported lazily so the game logic and tests run without them.
"""
import os
import time
import urllib.request
from typing import Dict, List

import cv2
import numpy as np

import config
from vision import gestures
from vision.observation import ColorBlob, Detection, HandInfo


class ObjectDetector:
    """YOLOv8 (COCO, 80 classes). Only keeps the classes the game cares about."""

    def __init__(self, model_path: str = config.YOLO_MODEL):
        from ultralytics import YOLO  # lazy: pulls in torch
        if model_path == config.YOLO_MODEL:
            _ensure_file(model_path, config.YOLO_MODEL_URL)
        self.model = YOLO(model_path)
        self.names = self.model.names
        wanted = set(config.OBJECT_POOL) | {a for a, _ in config.SPATIAL_PAIRS} | {b for _, b in config.SPATIAL_PAIRS}
        self.class_ids = [i for i, n in self.names.items() if n in wanted]

    def detect(self, frame_bgr) -> List[Detection]:
        result = self.model(frame_bgr, imgsz=config.YOLO_IMG_SIZE, conf=config.YOLO_CONFIDENCE,
                            classes=self.class_ids, verbose=False)[0]
        out = []
        for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                  result.boxes.conf.tolist()):
            out.append(Detection(self.names[int(cls)], float(conf), tuple(int(v) for v in box)))
        return out


class HandTracker:
    """MediaPipe Hand Landmarker (Tasks API) -> landmarks -> finger states -> gesture label.

    Uses the Tasks API because newer MediaPipe releases dropped the old `mp.solutions` module.
    The model file (~7.5 MB) is downloaded into models/ on first run.
    """

    def __init__(self, model_path: str = config.HAND_MODEL_PATH):
        import mediapipe as mp  # lazy
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        _ensure_file(model_path, config.HAND_MODEL_URL)
        self._mp = mp
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.HAND_DETECTION_CONF,
            min_hand_presence_confidence=config.HAND_DETECTION_CONF,
            min_tracking_confidence=config.HAND_TRACKING_CONF,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def detect(self, frame_bgr) -> List[HandInfo]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        res = self.landmarker.detect_for_video(image, ts)
        hands = []
        for i, hand_lm in enumerate(res.hand_landmarks or []):
            lm = [(p.x * w, p.y * h) for p in hand_lm]  # pixels keep the aspect ratio honest
            fingers = gestures.fingers_extended(lm)
            label = res.handedness[i][0].category_name if i < len(res.handedness) else "?"
            hands.append(HandInfo(lm, label, fingers, gestures.classify(lm, fingers)))
        return hands

    def close(self):
        self.landmarker.close()


def _ensure_file(path: str, url: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading {os.path.basename(path)} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        raise RuntimeError(f"Could not download {url}. Download it manually and save it as {path}") from e


class ColorDetector:
    """Finds the largest blob of each configured colour in HSV space."""

    def __init__(self, ranges: Dict = None):
        self.ranges = ranges or config.COLOR_RANGES
        self.kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame_bgr) -> Dict[str, ColorBlob]:
        h, w = frame_bgr.shape[:2]
        scale = 320 / w  # work on a small copy, it's plenty for colour
        small = cv2.resize(frame_bgr, (320, int(h * scale)))
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        total = small.shape[0] * small.shape[1]
        found = {}
        for color, ranges in self.ranges.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            frac = cv2.contourArea(c) / total
            if config.COLOR_MIN_AREA <= frac <= config.COLOR_MAX_AREA:
                x, y, bw, bh = cv2.boundingRect(c)
                box = tuple(int(v / scale) for v in (x, y, x + bw, y + bh))
                found[color] = ColorBlob(color, frac, box)
        return found
