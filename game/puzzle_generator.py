"""Builds a fresh run of puzzles each game, getting harder level by level.

Difficulty tiers
  1  single colour or finger count
  2  single gesture or object
  3  spatial reasoning between two objects
  4  two things at once (object + gesture)
  5  multi-step sequence
"""
import random
from typing import List, Optional

import config
from game.challenges import (AllOfChallenge, Challenge, ColorChallenge, FingerCountChallenge,
                             GestureChallenge, ObjectChallenge, SequenceChallenge, SpatialChallenge)

EASY_GESTURES = ["thumbs_up", "peace", "open_palm", "fist", "point"]
ALL_GESTURES = EASY_GESTURES + ["thumbs_down", "rock", "ok"]


class PuzzleGenerator:
    def __init__(self, seed: Optional[int] = None, use_objects: bool = True):
        self.rng = random.Random(seed)
        self.use_objects = use_objects
        self._recent = []

    # ---- building blocks ----
    def color(self):
        return ColorChallenge(self.rng.choice(list(config.COLOR_RANGES)))

    def fingers(self, max_count=5):
        return FingerCountChallenge(self.rng.randint(1, max_count))

    def gesture(self, pool=EASY_GESTURES):
        return GestureChallenge(self.rng.choice(pool))

    def obj(self):
        return ObjectChallenge(self.rng.choice(config.OBJECT_POOL))

    def spatial(self):
        a, b = self.rng.choice(config.SPATIAL_PAIRS)
        if self.rng.random() < 0.5:
            a, b = b, a
        return SpatialChallenge(a, self.rng.choice(["left of", "right of"]), b)

    def simple(self):
        makers = [self.color, self.fingers, self.gesture] + ([self.obj] if self.use_objects else [])
        return self.rng.choice(makers)()

    # ---- tiers ----
    def for_tier(self, tier: int) -> Challenge:
        for _ in range(20):  # avoid repeating a puzzle the player has just done
            c = self._make(tier)
            if c.description not in self._recent:
                break
        self._recent = (self._recent + [c.description])[-4:]
        return c

    def _make(self, tier: int) -> Challenge:
        o = self.use_objects
        if tier <= 1:
            return self.rng.choice([self.color, self.fingers])()
        if tier == 2:
            return self.rng.choice([self.gesture] + ([self.obj] if o else [self.fingers]))()
        if tier == 3:
            if o:
                return self.spatial()
            return AllOfChallenge([self.color(), self.gesture()])
        if tier == 4:
            first = self.obj() if o else self.color()
            return AllOfChallenge([first, self.gesture(ALL_GESTURES)])
        steps = 2 if tier == 5 else 3
        parts = []
        while len(parts) < steps:
            p = self.simple()
            if all(p.description != q.description for q in parts):
                parts.append(p)
        return SequenceChallenge(parts)

    def generate_run(self, n_levels: int = config.NUM_LEVELS) -> List[Challenge]:
        return [self.for_tier(self.tier_for_level(i, n_levels)) for i in range(n_levels)]

    @staticmethod
    def tier_for_level(level_index: int, n_levels: int) -> int:
        """Spread levels across tiers 1..6 (6 = 3-step sequence, only in long games)."""
        top = 5 if n_levels <= 5 else 6
        return 1 + round(level_index * (top - 1) / max(1, n_levels - 1))


def demo_run() -> List[Challenge]:
    """The exact five levels from the project proposal - handy for a predictable live demo."""
    return [
        ColorChallenge("red"),
        GestureChallenge("thumbs_up"),
        SpatialChallenge("bottle", "left of", "book"),
        FingerCountChallenge(3),
        SequenceChallenge([ColorChallenge("blue"), GestureChallenge("peace")]),
    ]
