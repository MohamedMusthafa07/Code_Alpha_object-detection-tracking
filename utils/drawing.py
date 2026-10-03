"""Drawing helpers: boxes, labels and the on-screen status overlay."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from utils.types import Detection, TrackedObject

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PENDING_COLOR = (170, 170, 170)  # grey: detected but no tracking ID yet
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Stable BGR colour per tracking ID."""
    return _PALETTE[track_id % len(_PALETTE)]


def _clip_box(box: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(np.clip(round(float(box[0])), 0, width - 1))
    y1 = int(np.clip(round(float(box[1])), 0, height - 1))
    x2 = int(np.clip(round(float(box[2])), 0, width - 1))
    y2 = int(np.clip(round(float(box[3])), 0, height - 1))
    return x1, y1, x2, y2


def _draw_box_with_label(
    frame: np.ndarray, box: np.ndarray, text: str, color: Tuple[int, int, int], thickness: int
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, width, height)
    if x2 <= x1 or y2 <= y1:
        return  # degenerate box after clipping

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font_scale = max(0.45, min(width, height) / 1400.0 + 0.3)
    text_thickness = 1 if font_scale < 0.8 else 2
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, font_scale, text_thickness)
    label_h = text_h + baseline + 4

    # Put the label above the box; if there is no room, put it inside at the top.
    label_top = y1 - label_h if y1 - label_h >= 0 else y1
    label_w = text_w + 6
    label_left = max(0, min(x1, width - 1 - label_w))  # shift left so the label stays on screen
    label_right = min(label_left + label_w, width - 1)
    cv2.rectangle(frame, (label_left, label_top), (label_right, label_top + label_h), color, cv2.FILLED)
    brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
    cv2.putText(frame, text, (label_left + 3, label_top + text_h + 2), _FONT, font_scale, text_color,
                text_thickness, cv2.LINE_AA)


def draw_tracked_object(frame: np.ndarray, obj: TrackedObject) -> None:
    """Draw ``Label | ID: n | NN%`` for a confirmed track."""
    text = f"{obj.label.capitalize()} | ID: {obj.track_id} | {obj.confidence * 100:.0f}%"
    _draw_box_with_label(frame, obj.box, text, color_for_id(obj.track_id), thickness=2)


def draw_pending_detection(frame: np.ndarray, det: Detection) -> None:
    """Draw a detection that has not been confirmed by the tracker yet (no ID is claimed)."""
    text = f"{det.label.capitalize()} | ID: pending | {det.confidence * 100:.0f}%"
    _draw_box_with_label(frame, det.box, text, _PENDING_COLOR, thickness=1)


def draw_status_overlay(frame: np.ndarray, lines: Iterable[str]) -> None:
    """Draw status text (FPS, counts, hints) in the top-left corner with a dark outline."""
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), _FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
