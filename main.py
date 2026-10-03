"""Object Detection and Tracking System: YOLO detection + SORT tracking, fully local.

Examples:
    python main.py                       # interactive menu (webcam or video file)
    python main.py --webcam              # webcam 0
    python main.py --video videos/sample.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np  # noqa: F401  (fail early with a clear message if NumPy is missing)
    import scipy  # noqa: F401
except ImportError as import_error:
    print(f"[ERROR] Missing dependency: {import_error.name}.\n"
          "Activate your virtual environment and run:  pip install -r requirements.txt")
    sys.exit(1)

from detector.yolo_detector import DetectorError, YoloDetector
from tracker.sort_tracker import SortTracker
from utils.drawing import draw_pending_detection, draw_status_overlay, draw_tracked_object
from utils.video import (
    VideoSourceError,
    create_video_writer,
    get_source_fps,
    open_video_file,
    open_webcam,
    resize_to_max_width,
)

WINDOW_NAME = "Object Detection and Tracking (press Q to quit)"
DEFAULT_MODEL_PATH = str(Path("models") / "yolo11n.pt")
QUIT_KEYS = {ord("q"), ord("Q"), 27}  # Q or Esc
MAX_CONSECUTIVE_WEBCAM_FAILURES = 30


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time object detection (YOLO) and tracking (SORT) using a webcam or video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--webcam", action="store_true", help="use the webcam as input")
    source.add_argument("--video", metavar="PATH", help="use a local video file as input")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam index (0 = default camera)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="path to the YOLO .pt weights")
    parser.add_argument("--download-model", action="store_true",
                        help="download the official weights if the model file is missing")
    parser.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold (0-1)")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold used by YOLO")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--classes", type=int, nargs="+", metavar="ID",
                        help="only detect these COCO class IDs (e.g. 0 = person, 2 = car)")
    parser.add_argument("--max-age", type=int, default=30, help="frames to keep a lost track alive")
    parser.add_argument("--min-hits", type=int, default=3, help="consecutive detections before an ID is shown")
    parser.add_argument("--track-iou", type=float, default=0.3, help="minimum IoU to match a detection to a track")
    parser.add_argument("--max-width", type=int, default=1280,
                        help="downscale frames wider than this before processing (0 = never)")
    parser.add_argument("--output", metavar="FILE.mp4", help="also save the annotated video to this file")
    parser.add_argument("--no-display", action="store_true", help="do not open a window (use with --output)")
    return parser.parse_args(argv)


def print_banner() -> None:
    print("=" * 64)
    print(" Object Detection and Tracking System  (YOLO + SORT, runs locally)")
    print("=" * 64)
    print(" Webcam mode     : python main.py --webcam")
    print(" Video-file mode : python main.py --video videos/your_video.mp4")
    print(" Exit            : press Q (or Esc) in the video window, or Ctrl+C in the terminal")
    print("=" * 64)


def choose_source_interactively(args: argparse.Namespace) -> None:
    """Ask the user which input to use when no source flag was given."""
    print("\nSelect input source:\n  1) Webcam\n  2) Video file\n  Q) Quit")
    while True:
        try:
            choice = input("Your choice [1/2/Q]: ").strip().lower()
        except EOFError:
            raise SystemExit("No input source selected.")
        if choice == "1":
            args.webcam = True
            return
        if choice == "2":
            try:
                args.video = input("Path to video file: ").strip()
            except EOFError:
                raise SystemExit("No video path given.")
            return
        if choice in {"q", "quit", "exit"}:
            raise SystemExit(0)
        print("Please enter 1, 2 or Q.")


def validate_args(args: argparse.Namespace) -> None:
    if args.no_display and not args.output:
        raise SystemExit("[ERROR] --no-display only makes sense together with --output FILE.mp4")
    if args.webcam and args.no_display:
        raise SystemExit("[ERROR] Webcam mode needs a window to quit with Q; do not use --no-display.")


def window_closed() -> bool:
    """True if the user closed the window with the X button."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def run(args: argparse.Namespace) -> int:
    # Open the input first: it is quick and fails fast with a clear message.
    is_webcam = bool(args.webcam)
    capture = open_webcam(args.camera_index) if is_webcam else open_video_file(args.video)
    writer = None
    window_opened = False
    try:
        print("\nLoading model (first start can take a few seconds)...")
        detector = YoloDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            nms_iou_threshold=args.nms_iou,
            image_size=args.imgsz,
            device=args.device,
            download_if_missing=args.download_model,
            class_ids=args.classes,
        )
        tracker = SortTracker(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.track_iou)
        print(f"Model ready on device: {detector.device}. Starting... (press Q to quit)\n")

        source_fps = get_source_fps(capture)
        show_window = not args.no_display
        frames_processed = 0
        failed_reads = 0
        smoothed_fps = 0.0
        last_time = time.perf_counter()

        while True:
            success, frame = capture.read()
            if not success:
                failed_reads += 1
                if is_webcam and failed_reads < MAX_CONSECUTIVE_WEBCAM_FAILURES:
                    continue
                if is_webcam:
                    print("[ERROR] The webcam stopped delivering frames.")
                    return 1
                print("End of video reached.")
                break
            failed_reads = 0

            frame = resize_to_max_width(frame, args.max_width)
            detections = detector.detect(frame)
            result = tracker.update(detections)  # always call, even with no detections

            for pending in result.pending:
                draw_pending_detection(frame, pending)
            for tracked in result.tracks:
                draw_tracked_object(frame, tracked)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            draw_status_overlay(frame, [
                f"FPS: {smoothed_fps:.1f}  ({detector.device})",
                f"Detections: {len(detections)}  Tracked: {len(result.tracks)}",
            ])

            if args.output and writer is None:
                height, width = frame.shape[:2]
                writer = create_video_writer(args.output, source_fps, (width, height))
            if writer is not None:
                writer.write(frame)

            if show_window:
                try:
                    cv2.imshow(WINDOW_NAME, frame)
                except cv2.error as exc:
                    print("[ERROR] OpenCV cannot open a window. If you installed "
                          "'opencv-python-headless', run: pip uninstall opencv-python-headless "
                          f"&& pip install opencv-python. Details: {exc}")
                    return 1
                window_opened = True
                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS or window_closed():
                    print("Exit requested by user.")
                    break

            frames_processed += 1

        print(f"Done. Frames processed: {frames_processed}")
        if args.output and writer is not None:
            print(f"Annotated video saved to: {args.output}")
        return 0
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if window_opened:
            cv2.destroyAllWindows()


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    print_banner()
    if not args.webcam and not args.video:
        choose_source_interactively(args)
    validate_args(args)
    try:
        return run(args)
    except (VideoSourceError, DetectorError, ValueError) as error:
        print(f"\n[ERROR] {error}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Resources released.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
