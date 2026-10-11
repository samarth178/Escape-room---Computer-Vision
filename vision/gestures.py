"""Rule-based gesture classifier on top of MediaPipe's 21 hand landmarks.

Landmark ids (MediaPipe convention):
    0 wrist | thumb 1-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-20
For each finger: MCP (knuckle), PIP, DIP, TIP.

All tests are distance ratios relative to palm size, so they work whatever the hand's
distance from the camera and mostly regardless of rotation.
"""
import math
from typing import List, Sequence, Tuple

import config

Point = Tuple[float, float]

WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

GESTURES = ["thumbs_up", "thumbs_down", "peace", "ok", "rock", "point", "open_palm", "fist"]


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_size(lm: List[Point]) -> float:
    return _dist(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6


def fingers_extended(lm: List[Point]) -> List[bool]:
    """Returns [thumb, index, middle, ring, pinky] -> True if extended."""
    palm = palm_size(lm)

    # Thumb: tip far from the index knuckle AND further from the pinky knuckle than its IP joint
    thumb = (_dist(lm[THUMB_TIP], lm[INDEX_MCP]) / palm > config.THUMB_EXTEND_RATIO and
             _dist(lm[THUMB_TIP], lm[PINKY_MCP]) > _dist(lm[THUMB_IP], lm[PINKY_MCP]))

    # Other fingers: an extended fingertip is clearly further from the wrist than its PIP joint;
    # a curled one folds back towards the palm.
    others = [_dist(lm[WRIST], lm[tip]) > config.FINGER_EXTEND_RATIO * _dist(lm[WRIST], lm[pip])
              for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]
    return [thumb] + others


def classify(lm: List[Point], fingers: List[bool] = None) -> str:
    if fingers is None:
        fingers = fingers_extended(lm)
    thumb, index, middle, ring, pinky = fingers
    palm = palm_size(lm)
    four_curled = not (index or middle or ring or pinky)

    if thumb and four_curled:
        # Image y grows downwards: "up" means the thumb tip is the highest point of the hand.
        others_y = [p[1] for i, p in enumerate(lm) if i not in (THUMB_IP, THUMB_TIP)]
        tip_y = lm[THUMB_TIP][1]
        if tip_y < min(others_y) - 0.15 * palm:
            return "thumbs_up"
        if tip_y > max(others_y) + 0.15 * palm:
            return "thumbs_down"

    if _dist(lm[THUMB_TIP], lm[8]) < 0.35 * palm and middle and ring and pinky:
        return "ok"
    if not thumb and four_curled:
        return "fist"
    if all(fingers):
        return "open_palm"
    if index and middle and not ring and not pinky:
        return "peace"
    if index and pinky and not middle and not ring:
        return "rock"
    if index and not middle and not ring and not pinky:
        return "point"
    return "none"


PRETTY = {
    "thumbs_up": "Thumbs up",
    "thumbs_down": "Thumbs down",
    "peace": "Peace sign",
    "ok": "OK sign",
    "rock": "Rock sign",
    "point": "Point with one finger",
    "open_palm": "Open palm",
    "fist": "Closed fist",
}
