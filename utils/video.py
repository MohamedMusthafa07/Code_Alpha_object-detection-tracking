"""Video input/output helpers built on OpenCV."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


class VideoSourceError(RuntimeError):
    """Raised when the webcam or video file cannot be used (user-facing message)."""


def validate_video_path(path_text: str) -> Path:
    """Check that the path points to an existing, supported video file."""
    if not path_text or not path_text.strip():
        raise VideoSourceError("No video path given. Example: --video videos/sample.mp4")
    path = Path(path_text.strip().strip('"').strip("'")).expanduser()
    if not path.exists():
        raise VideoSourceError(f"Video file not found: '{path}'. Check the path and file name.")
    if not path.is_file():
        raise VideoSourceError(f"'{path}' is a folder, not a video file.")
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise VideoSourceError(
            f"Unsupported video type '{path.suffix}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )
    return path


def open_video_file(path_text: str) -> cv2.VideoCapture:
    """Open a local video file and verify that OpenCV can decode its first frame."""
    path = validate_video_path(path_text)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoSourceError(
            f"OpenCV could not open '{path}'. The file may be corrupted or use an unsupported codec."
        )
    success, _ = capture.read()
    if not success:
        capture.release()
        raise VideoSourceError(f"'{path}' was opened but no frame could be read (empty or corrupted file).")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind so the first frame is processed too
    return capture


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    """Open a webcam by index and verify that it actually delivers frames."""
    if camera_index < 0:
        raise VideoSourceError("Camera index must be 0 or higher.")

    capture: Optional[cv2.VideoCapture] = None
    if platform.system() == "Windows":
        # DirectShow usually opens faster and more reliably than the default backend on Windows.
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = None
    if capture is None:
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        capture.release()
        raise VideoSourceError(
            f"Could not open webcam with index {camera_index}. Make sure a camera is connected, "
            "not used by another app (Zoom, Teams, browser), and allowed in Windows "
            "Settings > Privacy > Camera. Try --camera-index 1 for an external camera."
        )

    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low (ignored by some backends)

    # Some cameras need a few reads before the first valid frame arrives.
    for _ in range(10):
        success, _ = capture.read()
        if success:
            return capture
    capture.release()
    raise VideoSourceError(
        f"Webcam {camera_index} opened but returned no frames. Close other apps using the camera "
        "or try another --camera-index."
    )


def get_source_fps(capture: cv2.VideoCapture, default: float = 30.0) -> float:
    """FPS reported by the source, or a default if the source does not report a sane value."""
    fps = capture.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and 1.0 <= fps <= 240.0 else default


def resize_to_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale a frame that is wider than ``max_width`` (keeps aspect ratio). 0 = never resize."""
    height, width = frame.shape[:2]
    if max_width <= 0 or width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)


def create_video_writer(output_path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create an MP4 writer. ``frame_size`` is (width, height)."""
    path = Path(output_path)
    if path.suffix.lower() != ".mp4":
        raise VideoSourceError("Output file must end with .mp4 (example: --output output/result.mp4)")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        writer.release()
        raise VideoSourceError(f"Could not create the output video '{path}'.")
    return writer
