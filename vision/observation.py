"""Plain data classes describing what the camera 'sees' in one frame.

Everything downstream (game logic, tests) works only with these, so the game rules
never touch YOLO or MediaPipe directly.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


@dataclass
class Detection:
    label: str
    confidence: float
    box: Box

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


@dataclass
class HandInfo:
    landmarks: List[Tuple[float, float]]   # 21 points, pixel coordinates
    handedness: str                         # "Left" / "Right" as reported by MediaPipe
    fingers: List[bool]                     # [thumb, index, middle, ring, pinky] extended?
    gesture: str                            # e.g. "thumbs_up", "peace", "fist", "none"

    @property
    def finger_count(self) -> int:
        return sum(self.fingers)


@dataclass
class ColorBlob:
    color: str
    area_frac: float
    box: Box


@dataclass
class Observation:
    objects: List[Detection] = field(default_factory=list)
    hands: List[HandInfo] = field(default_factory=list)
    colors: Dict[str, ColorBlob] = field(default_factory=dict)
    frame_size: Tuple[int, int] = (640, 480)

    def best_object(self, label: str) -> Optional[Detection]:
        matches = [d for d in self.objects if d.label == label]
        return max(matches, key=lambda d: d.confidence) if matches else None

    def has_object(self, label: str) -> bool:
        return self.best_object(label) is not None

    @property
    def gestures(self) -> set:
        return {h.gesture for h in self.hands}

    @property
    def total_fingers(self) -> int:
        return sum(h.finger_count for h in self.hands)

    def summary(self) -> List[str]:
        """Short human-readable lines for the 'AI sees' panel."""
        lines = []
        if self.objects:
            labels = sorted({d.label for d in self.objects})
            lines.append("Objects: " + ", ".join(labels))
        if self.colors:
            lines.append("Colours: " + ", ".join(sorted(self.colors)))
        for i, h in enumerate(self.hands, 1):
            g = h.gesture.replace("_", " ")
            lines.append(f"Hand {i}: {h.finger_count} finger(s), {g}")
        return lines or ["Nothing recognised yet"]
