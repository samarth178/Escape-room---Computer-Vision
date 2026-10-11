"""The 'AI game master': levels, timer, lives, hints, scoring and progression.

The engine never touches the camera. Each frame it receives an Observation and the current
time, so the whole game can be unit-tested with fake observations.
"""
import json
import os
from enum import Enum
from typing import List, Optional

import config
from game.challenges import Challenge
from game.puzzle_generator import PuzzleGenerator, demo_run


class State(Enum):
    MENU = "menu"
    PLAYING = "playing"
    LEVEL_CLEAR = "level_clear"
    GAME_OVER = "game_over"
    VICTORY = "victory"


class GameEngine:
    def __init__(self, n_levels: int = config.NUM_LEVELS, seed: Optional[int] = None,
                 demo: bool = False, use_objects: bool = True, level_time: float = config.LEVEL_TIME,
                 lives: int = config.LIVES, highscore_file: str = config.HIGHSCORE_FILE):
        self.n_levels = 5 if demo else n_levels
        self.demo = demo
        self.generator = PuzzleGenerator(seed, use_objects)
        self.level_time = level_time
        self.max_lives = lives
        self.highscore_file = highscore_file
        self.highscore = self._load_highscore()
        self.state = State.MENU
        self.levels: List[Challenge] = []
        self.level = 0
        self.score = 0
        self.lives = lives
        self.hints_used = 0
        self.show_hint = False
        self.message = ""
        self.message_until = 0.0
        self.level_started = 0.0
        self.state_since = 0.0
        self.last_update = None
        self.cleared_with = 0.0
        self.level_log = []  # (level, description, seconds, points) for the end screen

    # ---------- lifecycle ----------
    def start(self, now: float):
        self.levels = demo_run() if self.demo else self.generator.generate_run(self.n_levels)
        self.level = 0
        self.score = 0
        self.lives = self.max_lives
        self.level_log = []
        self._begin_level(now)

    def _begin_level(self, now: float):
        self.state = State.PLAYING
        self.level_started = now
        self.state_since = now
        self.last_update = now
        self.hints_used = 0
        self.show_hint = False
        self.challenge.reset()

    @property
    def challenge(self) -> Optional[Challenge]:
        return self.levels[self.level] if self.levels else None

    def requirements(self) -> dict:
        """Which detectors to run this frame. Saves CPU, e.g. no YOLO during a gesture level."""
        c = self.challenge
        if self.state != State.PLAYING or c is None:
            return dict(need_objects=False, need_hands=True, need_colors=False)
        return dict(need_objects=c.needs_objects, need_hands=c.needs_hands, need_colors=c.needs_colors)

    def time_left(self, now: float) -> float:
        if self.state != State.PLAYING:
            return 0.0
        return max(0.0, self.level_time - (now - self.level_started))

    # ---------- per-frame update ----------
    def update(self, obs, now: float):
        dt = 0.0 if self.last_update is None else min(now - self.last_update, 0.25)
        self.last_update = now

        if self.state == State.LEVEL_CLEAR and now - self.state_since >= config.LEVEL_CLEAR_PAUSE:
            self.level += 1
            if self.level >= len(self.levels):
                self._finish(State.VICTORY, now)
            else:
                self._begin_level(now)
            return

        if self.state != State.PLAYING:
            return

        c = self.challenge
        solved = c.update(obs, dt, now)
        if getattr(c, "timed_out", False):
            self.flash("Too slow between steps - sequence restarted!", now)

        if solved:
            secs = now - self.level_started
            self.cleared_with = self.time_left(now)
            points = (config.POINTS_PER_LEVEL + int(self.time_left(now)) * config.POINTS_PER_SECOND_LEFT
                      - self.hints_used * config.HINT_PENALTY)
            points = max(points, 10)
            self.score += points
            self.level_log.append((self.level + 1, c.description, round(secs, 1), points))
            self.state = State.LEVEL_CLEAR
            self.state_since = now
            self.flash(f"Puzzle solved! +{points} points", now, config.LEVEL_CLEAR_PAUSE)
        elif self.time_left(now) <= 0:
            self._lose_life("Time's up!", now)

    def _lose_life(self, reason: str, now: float):
        self.lives -= 1
        if self.lives <= 0:
            self.flash(reason, now)
            self._finish(State.GAME_OVER, now)
            return
        # new puzzle of the same difficulty (demo mode keeps the scripted puzzle)
        if not self.demo:
            tier = PuzzleGenerator.tier_for_level(self.level, len(self.levels))
            self.levels[self.level] = self.generator.for_tier(tier)
        self.flash(f"{reason} You lost a life - here's a new puzzle.", now, 3.0)
        self._begin_level(now)

    def _finish(self, state: State, now: float):
        self.state = state
        self.state_since = now
        if self.score > self.highscore:
            self.highscore = self.score
            self._save_highscore()

    # ---------- player actions ----------
    def use_hint(self, now: float):
        if self.state == State.PLAYING and not self.show_hint:
            self.show_hint = True
            self.hints_used += 1
            self.flash(f"Hint used (-{config.HINT_PENALTY} points)", now)

    def skip(self, now: float):
        if self.state == State.PLAYING:
            self._lose_life("Skipped.", now)

    def flash(self, text: str, now: float, seconds: float = 2.5):
        self.message = text
        self.message_until = now + seconds

    def current_message(self, now: float) -> str:
        return self.message if now < self.message_until else ""

    # ---------- high score ----------
    def _load_highscore(self) -> int:
        try:
            with open(self.highscore_file) as f:
                return int(json.load(f).get("highscore", 0))
        except (OSError, ValueError):
            return 0

    def _save_highscore(self):
        try:
            with open(self.highscore_file, "w") as f:
                json.dump({"highscore": self.highscore}, f)
        except OSError:
            pass
