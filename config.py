"""Central settings for the CV Escape Room. Tweak these to suit your room, lighting and camera."""

# ---------------- Camera ----------------
CAMERA_INDEX = 0            # 0 = default webcam. Try 1 if you have an external camera.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR = True               # selfie view, so "left" on screen is the player's left

# ---------------- Object detection (YOLOv8) ----------------
YOLO_MODEL = "models/yolov8n.pt"  # nano model (~6 MB), downloaded automatically if missing
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
YOLO_CONFIDENCE = 0.40
YOLO_IMG_SIZE = 480
YOLO_EVERY_N_FRAMES = 2     # run YOLO on every 2nd frame to keep the UI smooth on CPU

# COCO objects the game may ask for. Remove anything you don't have at home before the demo.
OBJECT_POOL = ["bottle", "cup", "book", "cell phone", "scissors", "remote", "mouse", "spoon"]

# Pairs that make sense for "place A to the left/right of B" puzzles
SPATIAL_PAIRS = [("bottle", "book"), ("cup", "book"), ("cell phone", "cup"),
                 ("bottle", "cup"), ("remote", "book"), ("mouse", "cell phone")]

# ---------------- Colour detection (HSV, OpenCV ranges: H 0-179) ----------------
COLOR_RANGES = {
    "red":    [((0, 140, 80), (8, 255, 255)), ((170, 140, 80), (179, 255, 255))],
    "blue":   [((100, 150, 60), (128, 255, 255))],
    "green":  [((40, 100, 60), (80, 255, 255))],
    "yellow": [((20, 130, 120), (34, 255, 255))],
}
COLOR_MIN_AREA = 0.02       # blob must cover at least 2% of the frame
COLOR_MAX_AREA = 0.60       # ...and not the whole frame (e.g. a wall)

# ---------------- Hand tracking (MediaPipe) ----------------
HAND_MODEL_PATH = "models/hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
MAX_HANDS = 2
HAND_DETECTION_CONF = 0.6
HAND_TRACKING_CONF = 0.5
FINGER_EXTEND_RATIO = 1.10  # tip must be this much further from wrist than the PIP joint
THUMB_EXTEND_RATIO = 0.60   # thumb tip distance from index knuckle, relative to palm size

# ---------------- Game rules ----------------
NUM_LEVELS = 5
LEVEL_TIME = 60             # seconds per level
LIVES = 3
HOLD_TIME = 1.0             # player must hold the pose/object for this long (stops flicker wins)
PROGRESS_DECAY = 1.5        # how fast the hold bar drains when detection drops out
SEQUENCE_WINDOW = 10.0      # seconds allowed between steps of a sequence puzzle
LEVEL_CLEAR_PAUSE = 1.8     # celebration screen between levels

POINTS_PER_LEVEL = 100
POINTS_PER_SECOND_LEFT = 2
HINT_PENALTY = 25

# ---------------- Window ----------------
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
SIDEBAR_WIDTH = 400
FPS = 30
HIGHSCORE_FILE = "highscore.json"
