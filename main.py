"""AI-Powered Escape Room using Computer Vision.

Run:   python main.py              (random puzzles every game)
       python main.py --demo       (the 5 fixed levels from the proposal)
       python main.py --no-yolo    (gestures + colours only, if torch won't install)
       python main.py --camera 1   (use a different webcam)
"""
import argparse
import os
import sys

# make relative paths (models/, highscore.json) work no matter where VS Code launches us from
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="AI-Powered Escape Room using Computer Vision")
    p.add_argument("--camera", default=str(config.CAMERA_INDEX),
                   help="webcam index (0, 1, ...) or path to a video file")
    p.add_argument("--demo", action="store_true", help="play the 5 fixed levels from the proposal")
    p.add_argument("--levels", type=int, default=config.NUM_LEVELS, help="number of levels (random mode)")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible puzzles")
    p.add_argument("--no-yolo", action="store_true", help="disable object detection puzzles")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.camera) if args.camera.isdigit() else args.camera
    from ui.app import App
    App(source, use_yolo=not args.no_yolo, seed=args.seed, demo=args.demo, levels=args.levels).run()


if __name__ == "__main__":
    main()
