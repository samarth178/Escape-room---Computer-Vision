# AI-Powered Escape Room using Computer Vision

An escape room where the webcam is the game master. Each level is a real-world puzzle (show a red object, make a thumbs up, put a bottle left of a book...) and the next lock only opens when the computer vision pipeline confirms you've actually done it.

## Setup (Windows + VS Code)

Use **Python 3.11 or 3.12** (MediaPipe and PyTorch are most reliable there).

```powershell
cd cv_escape_room
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

In VS Code: open the folder, press `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `.venv`. Then use the Run and Debug panel (F5), which already has configurations for random mode, demo mode, no-YOLO mode and the tests.

Or from the terminal:

```powershell
python main.py            # random puzzles every game
python main.py --demo     # the exact 5 levels from the proposal (best for a live demo)
python main.py --no-yolo  # gestures + colours only
python main.py --camera 1 # external webcam
python -m pytest -q       # 24 unit tests, no webcam needed
```

`models/` already has `yolov8n.pt`. On first run the game downloads MediaPipe's `hand_landmarker.task` (~7.5 MB) next to it. If the download is blocked, get it from the URL in `config.py` and drop it in `models/`.

**Controls:** SPACE start · H hint (−25 pts) · S skip (costs a life) · D toggle detection overlay · ESC quit

## How it works

```
Webcam (OpenCV) ──► Perception ──► Observation ──► Game engine ──► Pygame UI
                    ├─ YOLOv8n        objects        levels, timer,
                    ├─ MediaPipe      hands+gesture  lives, scoring,
                    └─ HSV masks      colour blobs   puzzle checks
```

- **`vision/detectors.py`** wraps the three models. YOLOv8 is filtered to the household objects the game uses; MediaPipe gives 21 hand landmarks; colour uses HSV thresholding + morphology + largest contour.
- **`vision/gestures.py`** turns landmarks into finger states and gestures (thumbs up/down, peace, OK, rock, point, open palm, fist) using distance ratios relative to palm size, so it works at any distance from the camera.
- **`game/challenges.py`** – each puzzle type decides "solved or not" from an Observation: colour, gesture, finger count (works across two hands), object, **spatial relation** (compares bounding-box centres with a minimum gap), **all-of** (two things at once) and **sequence** (steps in order, with a time window between steps).
- **Hold-to-verify:** the answer must be held for 1 s. The bar fills while the condition is true and drains when it isn't, so one missed frame doesn't wipe progress and random flicker can't solve a level.
- **`game/puzzle_generator.py`** builds a new run every game with rising difficulty: colour/fingers → gesture/object → spatial → simultaneous → sequence. `--seed` makes a run reproducible.
- **`game/engine.py`** is the game master: 60 s per level, 3 lives, new puzzle of the same difficulty on timeout, score = 100 + 2 × seconds left − hint penalties, high score saved to `highscore.json`.
- **Efficiency:** the engine tells perception which detectors the current puzzle needs, so YOLO doesn't run during a gesture level, and YOLO runs on every 2nd frame.

## Tuning for your room

Everything is in `config.py`. Before a demo, trim `OBJECT_POOL` and `SPATIAL_PAIRS` to objects you actually have on the table. If colours misfire under your lighting, adjust `COLOR_RANGES`; if finger counting is off, nudge `FINGER_EXTEND_RATIO` / `THUMB_EXTEND_RATIO`. The right-hand "AI SEES" panel and the overlay (D) show exactly what the system detects, which helps both tuning and explaining it to an audience.

## Known limitations

YOLOv8n is trained on COCO, so only those 80 classes can be puzzles. Colour detection reacts to any large coloured area (a red shirt counts as "a red object"). Thumb detection is the least reliable part of the hand rules; facing your palm to the camera helps.
