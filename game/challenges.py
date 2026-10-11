"""Puzzle types. Each one turns raw detections into a yes/no 'is it solved?' decision.

A puzzle isn't solved by a single lucky frame: the player has to hold the answer for
HOLD_TIME seconds. The hold bar fills while the condition is true and drains when it isn't,
so one dropped frame doesn't throw away progress but random flicker can't win the level.
"""
from typing import List

import config
from vision.gestures import PRETTY
from vision.observation import Observation


class Challenge:
    kind = "base"
    needs_objects = False
    needs_hands = False
    needs_colors = False

    def __init__(self, description: str, hint: str, hold_time: float = config.HOLD_TIME):
        self.description = description
        self.hint = hint
        self.hold_time = hold_time
        self.progress = 0.0

    def is_satisfied(self, obs: Observation) -> bool:
        raise NotImplementedError

    def update(self, obs: Observation, dt: float, now: float = 0.0) -> bool:
        """Advance the hold timer; returns True once the puzzle is solved."""
        if self.is_satisfied(obs):
            self.progress = min(self.hold_time, self.progress + dt)
        else:
            self.progress = max(0.0, self.progress - dt * config.PROGRESS_DECAY)
        return self.progress >= self.hold_time

    @property
    def progress_frac(self) -> float:
        return self.progress / self.hold_time if self.hold_time else 1.0

    def reset(self):
        self.progress = 0.0

    def steps(self) -> List[tuple]:
        """(text, state) rows for the sidebar. state: 'done' | 'active' | 'todo'."""
        return []

    def __repr__(self):
        return f"<{self.kind}: {self.description}>"


class ColorChallenge(Challenge):
    kind = "color"
    needs_colors = True

    def __init__(self, color: str):
        super().__init__(f"Find something {color.upper()} and show it to the camera",
                         f"Any {color} item works: a pen cap, a cloth, a book cover. Fill a good part of the frame.")
        self.color = color

    def is_satisfied(self, obs):
        return self.color in obs.colors


class GestureChallenge(Challenge):
    kind = "gesture"
    needs_hands = True

    def __init__(self, gesture: str):
        super().__init__(f"Make this gesture: {PRETTY[gesture].upper()}",
                         "Keep your whole hand inside the frame, about an arm's length from the camera.")
        self.gesture = gesture

    def is_satisfied(self, obs):
        return self.gesture in obs.gestures


class FingerCountChallenge(Challenge):
    kind = "fingers"
    needs_hands = True

    def __init__(self, count: int):
        word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][count]
        two = " (you'll need both hands)" if count > 5 else ""
        super().__init__(f"Show exactly {word.upper()} finger{'s' if count != 1 else ''}{two}",
                         "Spread your fingers apart and face your palm towards the camera.")
        self.count = count

    def is_satisfied(self, obs):
        return bool(obs.hands) and obs.total_fingers == self.count


class ObjectChallenge(Challenge):
    kind = "object"
    needs_objects = True

    def __init__(self, label: str):
        super().__init__(f"Find a {label.upper()} and bring it to the camera",
                         f"Hold the {label} still and don't cover most of it with your hand.")
        self.label = label

    def is_satisfied(self, obs):
        return obs.has_object(self.label)


class SpatialChallenge(Challenge):
    """'Place A to the left of B' – compares bounding-box centres of the two objects."""
    kind = "spatial"
    needs_objects = True
    RELATIONS = {"left of": "LEFT of", "right of": "RIGHT of", "above": "ABOVE", "below": "BELOW"}

    def __init__(self, a: str, relation: str, b: str):
        super().__init__(f"Place a {a.upper()} {self.RELATIONS[relation]} a {b.upper()}",
                         f"Both the {a} and the {b} must be visible at the same time, clearly apart.")
        self.a, self.relation, self.b = a, relation, b

    def is_satisfied(self, obs):
        da, db = obs.best_object(self.a), obs.best_object(self.b)
        if da is None or db is None:
            return False
        (ax, ay), (bx, by) = da.center, db.center
        # require a gap of a quarter of the smaller object, so "touching" objects don't count
        gap_x = 0.25 * min(da.width, db.width)
        gap_y = 0.25 * min(da.height, db.height)
        return {"left of": ax < bx - gap_x,
                "right of": ax > bx + gap_x,
                "above": ay < by - gap_y,
                "below": ay > by + gap_y}[self.relation]


class AllOfChallenge(Challenge):
    """Several conditions at the same time, e.g. hold a cup while making a thumbs up."""
    kind = "all_of"

    def __init__(self, parts: List[Challenge]):
        desc = " AND ".join(p.description.split(": ")[-1] if p.kind == "gesture" else p.description
                            for p in parts)
        super().__init__("At the same time: " + desc, " / ".join(p.hint for p in parts),
                         hold_time=config.HOLD_TIME * 1.2)
        self.parts = parts
        self.needs_objects = any(p.needs_objects for p in parts)
        self.needs_hands = any(p.needs_hands for p in parts)
        self.needs_colors = any(p.needs_colors for p in parts)
        self._last = None

    def is_satisfied(self, obs):
        self._last = [p.is_satisfied(obs) for p in self.parts]
        return all(self._last)

    def steps(self):
        flags = self._last or [False] * len(self.parts)
        return [(p.description, "done" if ok else "todo") for p, ok in zip(self.parts, flags)]


class SequenceChallenge(Challenge):
    """Do several puzzles in order. Too long between steps and the sequence restarts."""
    kind = "sequence"

    def __init__(self, parts: List[Challenge], window: float = config.SEQUENCE_WINDOW):
        super().__init__(f"Complete {len(parts)} tasks IN ORDER", parts[0].hint,
                         hold_time=1.0)
        self.parts = parts
        self.window = window
        self.index = 0
        self.last_step_time = None
        self.needs_objects = any(p.needs_objects for p in parts)
        self.needs_hands = any(p.needs_hands for p in parts)
        self.needs_colors = any(p.needs_colors for p in parts)
        self.timed_out = False

    @property
    def current(self) -> Challenge:
        return self.parts[min(self.index, len(self.parts) - 1)]

    def is_satisfied(self, obs):
        return self.index >= len(self.parts)

    def update(self, obs, dt, now=0.0):
        self.timed_out = False
        if self.index > 0 and self.last_step_time is not None and now - self.last_step_time > self.window:
            self.reset()
            self.timed_out = True
        if self.current.update(obs, dt, now):
            self.index += 1
            self.last_step_time = now
            if self.index < len(self.parts):
                self.hint = self.current.hint
        self.progress = (self.index + self.current.progress_frac * (self.index < len(self.parts))) \
            / len(self.parts) * self.hold_time
        return self.index >= len(self.parts)

    def reset(self):
        super().reset()
        self.index = 0
        self.last_step_time = None
        for p in self.parts:
            p.reset()
        self.hint = self.parts[0].hint

    def time_left_for_step(self, now) -> float:
        if self.index == 0 or self.last_step_time is None:
            return self.window
        return max(0.0, self.window - (now - self.last_step_time))

    def steps(self):
        rows = []
        for i, p in enumerate(self.parts):
            state = "done" if i < self.index else ("active" if i == self.index else "todo")
            rows.append((f"{i + 1}. {p.description}", state))
        return rows
