"""Unit tests for gesture rules, spatial reasoning, colour detection and the game engine.

These use synthetic data, so they run without a webcam:   python -m pytest -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from game.challenges import (AllOfChallenge, ColorChallenge, FingerCountChallenge,  # noqa: E402
                             GestureChallenge, ObjectChallenge, SequenceChallenge, SpatialChallenge)
from game.engine import GameEngine, State  # noqa: E402
from game.puzzle_generator import PuzzleGenerator  # noqa: E402
from vision import gestures  # noqa: E402
from vision.observation import ColorBlob, Detection, HandInfo, Observation  # noqa: E402


# ---------------------------------------------------------------- synthetic hands
MCP = {"index": (0.43, 0.70), "middle": (0.50, 0.68), "ring": (0.56, 0.69), "pinky": (0.62, 0.72)}


def make_hand(index=False, middle=False, ring=False, pinky=False, thumb="tucked"):
    """Upright hand (palm to camera). Extended fingers point up, curled ones fold back."""
    lm = [None] * 21
    lm[0] = (0.50, 0.90)
    lm[1] = (0.42, 0.85)
    if thumb == "out":
        lm[2], lm[3], lm[4] = (0.36, 0.79), (0.31, 0.74), (0.27, 0.69)
    elif thumb == "up":
        lm[2], lm[3], lm[4] = (0.39, 0.70), (0.38, 0.55), (0.38, 0.45)
    elif thumb == "down":  # hand rotated: whole hand above, thumb pointing down
        lm[2], lm[3], lm[4] = (0.39, 0.92), (0.38, 1.02), (0.38, 1.12)
    elif thumb == "touch_index":
        lm[2], lm[3], lm[4] = (0.36, 0.79), (0.33, 0.70), (0.42, 0.52)
    else:  # tucked across the palm
        lm[2], lm[3], lm[4] = (0.40, 0.80), (0.42, 0.77), (0.50, 0.74)
    for (name, ext), base in zip([("index", index), ("middle", middle), ("ring", ring), ("pinky", pinky)],
                                 (5, 9, 13, 17)):
        mx, my = MCP[name]
        lm[base] = (mx, my)
        if ext:
            lm[base + 1], lm[base + 2], lm[base + 3] = (mx, my - 0.08), (mx, my - 0.13), (mx, my - 0.17)
        else:
            lm[base + 1], lm[base + 2], lm[base + 3] = (mx, my - 0.05), (mx, my - 0.02), (mx, my + 0.03)
    if thumb == "touch_index":  # OK sign: index curls to meet the thumb
        lm[6], lm[7], lm[8] = (0.43, 0.62), (0.43, 0.56), (0.43, 0.53)
    return lm


def hand_obs(lm, extra=None):
    fingers = gestures.fingers_extended(lm)
    obs = Observation(hands=[HandInfo(lm, "Right", fingers, gestures.classify(lm, fingers))])
    if extra:
        for k, v in extra.items():
            setattr(obs, k, v)
    return obs


@pytest.mark.parametrize("kwargs, gesture, count", [
    (dict(thumb="tucked"), "fist", 0),
    (dict(index=True, middle=True, ring=True, pinky=True, thumb="out"), "open_palm", 5),
    (dict(index=True, middle=True), "peace", 2),
    (dict(index=True), "point", 1),
    (dict(index=True, pinky=True), "rock", 2),
    (dict(thumb="up"), "thumbs_up", 1),
    (dict(thumb="down"), "thumbs_down", 1),
    (dict(middle=True, ring=True, pinky=True, thumb="touch_index"), "ok", None),
    (dict(index=True, middle=True, ring=True), "none", 3),
    (dict(index=True, middle=True, ring=True, pinky=True), "none", 4),
])
def test_gestures(kwargs, gesture, count):
    lm = make_hand(**kwargs)
    fingers = gestures.fingers_extended(lm)
    assert gestures.classify(lm, fingers) == gesture
    if count is not None:
        assert sum(fingers) == count


def test_gesture_is_scale_invariant():
    lm = [(x * 640 + 100, y * 480 + 20) for x, y in make_hand(index=True, middle=True)]
    assert gestures.classify(lm) == "peace"


# ---------------------------------------------------------------- challenges
def det(label, x, y, w=80, h=120, conf=0.8):
    return Detection(label, conf, (x, y, x + w, y + h))


def test_spatial_left_of():
    c = SpatialChallenge("bottle", "left of", "book")
    assert c.is_satisfied(Observation(objects=[det("bottle", 50, 100), det("book", 400, 100)]))
    assert not c.is_satisfied(Observation(objects=[det("bottle", 400, 100), det("book", 50, 100)]))
    assert not c.is_satisfied(Observation(objects=[det("bottle", 50, 100)]))          # book missing
    assert not c.is_satisfied(Observation(objects=[det("bottle", 100, 100), det("book", 110, 100)]))  # overlapping


def test_hold_time_and_decay():
    c = ObjectChallenge("cup")
    yes, no = Observation(objects=[det("cup", 0, 0)]), Observation()
    assert not c.update(yes, 0.5)
    assert not c.update(no, 0.1)          # one dropped frame only drains a little
    assert c.progress > 0.3
    assert not c.update(yes, 0.5)
    assert c.update(yes, 0.5)             # held long enough


def test_finger_count_over_two_hands():
    c = FingerCountChallenge(7)
    h1 = make_hand(index=True, middle=True, ring=True, pinky=True, thumb="out")
    h2 = make_hand(index=True, middle=True)
    obs = Observation(hands=[HandInfo(h, "R", gestures.fingers_extended(h), "x") for h in (h1, h2)])
    assert c.is_satisfied(obs)


def test_all_of_needs_both():
    c = AllOfChallenge([ObjectChallenge("cup"), GestureChallenge("thumbs_up")])
    thumbs = make_hand(thumb="up")
    assert not c.is_satisfied(hand_obs(thumbs))
    assert c.is_satisfied(hand_obs(thumbs, {"objects": [det("cup", 0, 0)]}))
    assert c.needs_objects and c.needs_hands


def test_sequence_order_and_window():
    seq = SequenceChallenge([ColorChallenge("blue"), GestureChallenge("peace")], window=5)
    blue = Observation(colors={"blue": ColorBlob("blue", 0.1, (0, 0, 10, 10))})
    peace = hand_obs(make_hand(index=True, middle=True))
    t = 0.0
    # doing step 2 first does nothing
    for _ in range(20):
        t += 0.1
        seq.update(peace, 0.1, t)
    assert seq.index == 0
    for _ in range(11):
        t += 0.1
        seq.update(blue, 0.1, t)
    assert seq.index == 1
    # wait too long -> resets
    t += 6
    seq.update(Observation(), 0.1, t)
    assert seq.index == 0 and seq.timed_out
    # do it properly
    for obs in [blue] * 11 + [peace] * 11:
        t += 0.1
        done = seq.update(obs, 0.1, t)
    assert done


# ---------------------------------------------------------------- colour detector (real OpenCV)
def test_color_detector_finds_red_square():
    from vision.detectors import ColorDetector
    img = np.full((480, 640, 3), 128, np.uint8)
    img[150:330, 250:430] = (30, 30, 220)   # BGR red
    found = ColorDetector().detect(img)
    assert "red" in found and "blue" not in found
    x1, y1, x2, y2 = found["red"].box
    assert abs(x1 - 250) < 15 and abs(y2 - 330) < 15


def test_color_detector_ignores_tiny_blobs():
    from vision.detectors import ColorDetector
    img = np.full((480, 640, 3), 128, np.uint8)
    img[10:25, 10:25] = (220, 60, 30)        # small blue speck
    assert "blue" not in ColorDetector().detect(img)


# ---------------------------------------------------------------- generator + engine
def test_generator_is_reproducible_and_progressive():
    a = [c.description for c in PuzzleGenerator(seed=7).generate_run(5)]
    b = [c.description for c in PuzzleGenerator(seed=7).generate_run(5)]
    assert a == b
    run = PuzzleGenerator(seed=3).generate_run(5)
    assert run[0].kind in ("color", "fingers")
    assert run[2].kind == "spatial"
    assert run[4].kind == "sequence"
    assert len({tuple(c.description for c in PuzzleGenerator(seed=s).generate_run(5)) for s in range(10)}) > 5


def test_generator_without_yolo_never_asks_for_objects():
    for s in range(30):
        for c in PuzzleGenerator(seed=s, use_objects=False).generate_run(7):
            assert not c.needs_objects, c


def test_engine_full_demo_run(tmp_path):
    e = GameEngine(demo=True, highscore_file=str(tmp_path / "hs.json"))
    t = 0.0
    e.start(t)
    peace = make_hand(index=True, middle=True)
    answers = [
        Observation(colors={"red": ColorBlob("red", 0.1, (0, 0, 9, 9))}),
        hand_obs(make_hand(thumb="up")),
        Observation(objects=[det("bottle", 40, 100), det("book", 400, 100)]),
        hand_obs(make_hand(index=True, middle=True, ring=True)),
        None,  # sequence handled below
    ]
    for level in range(4):
        assert e.state == State.PLAYING and e.level == level
        while e.state == State.PLAYING:
            t += 1 / 30
            e.update(answers[level], t)
        assert e.state == State.LEVEL_CLEAR
        t += config.LEVEL_CLEAR_PAUSE + 0.01
        e.update(Observation(), t)
    blue = Observation(colors={"blue": ColorBlob("blue", 0.1, (0, 0, 9, 9))})
    for obs in [blue] * 40 + [hand_obs(peace)] * 40:
        t += 1 / 30
        e.update(obs, t)
    t += config.LEVEL_CLEAR_PAUSE + 0.01
    e.update(Observation(), t)
    assert e.state == State.VICTORY
    assert e.score > 5 * config.POINTS_PER_LEVEL
    assert GameEngine(highscore_file=str(tmp_path / "hs.json")).highscore == e.score


def test_engine_timeout_costs_lives_then_game_over(tmp_path):
    e = GameEngine(seed=1, level_time=5, lives=2, highscore_file=str(tmp_path / "hs.json"))
    e.start(0.0)
    t = 0.0
    while e.state == State.PLAYING and t < 60:
        t += 0.2
        e.update(Observation(), t)
    assert e.state == State.GAME_OVER and e.lives == 0


def test_hint_reduces_points(tmp_path):
    def play(hint):
        e = GameEngine(demo=True, highscore_file=str(tmp_path / "h.json"))
        e.start(0.0)
        if hint:
            e.use_hint(0.0)
        red = Observation(colors={"red": ColorBlob("red", 0.1, (0, 0, 9, 9))})
        t = 0.0
        while e.state == State.PLAYING:
            t += 0.05
            e.update(red, t)
        return e.score
    assert play(False) - play(True) == config.HINT_PENALTY


def test_requirements_follow_current_puzzle(tmp_path):
    e = GameEngine(demo=True, highscore_file=str(tmp_path / "h.json"))
    e.start(0.0)
    assert e.requirements() == dict(need_objects=False, need_hands=False, need_colors=True)
