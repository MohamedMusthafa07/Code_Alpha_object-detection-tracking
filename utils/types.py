"""Shared data containers used by the detector, tracker and drawing code.

Keeping them in one dependency-free module avoids circular imports between
``detector`` and ``tracker``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class Detection:
    """One object found by the detector in a single frame."""

    box: np.ndarray  # shape (4,), float32, pixel coordinates [x1, y1, x2, y2]
    confidence: float  # 0.0 - 1.0
    class_id: int
    label: str


@dataclass(eq=False)
class TrackedObject:
    """An object that the tracker has confirmed and assigned a persistent ID."""

    track_id: int
    box: np.ndarray  # shape (4,), float32, pixel coordinates [x1, y1, x2, y2]
    confidence: float  # confidence of the detection matched in this frame
    class_id: int
    label: str
