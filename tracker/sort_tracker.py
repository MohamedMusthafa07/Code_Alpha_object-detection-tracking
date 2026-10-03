"""SORT (Simple Online and Realtime Tracking) implemented with NumPy + SciPy.

Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.

Each tracked object has a constant-velocity Kalman filter over its bounding box.
Every frame the filters predict a new box, detections are matched to predictions
with the Hungarian algorithm on IoU, and unmatched detections start new tracks.

Only tracks that were matched to a detection in the *current* frame and have
been confirmed (seen in ``min_hits`` consecutive frames) are reported with an ID.
Everything else is returned as a "pending" detection without an ID, so the
display never claims an object is tracked when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.types import Detection, TrackedObject

_EPS = 1e-6


def _box_to_measurement(box: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] -> [cx, cy, area, aspect_ratio] column vector."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    center_x = box[0] + width / 2.0
    center_y = box[1] + height / 2.0
    area = width * height
    aspect_ratio = width / max(height, _EPS)
    return np.array([center_x, center_y, area, aspect_ratio], dtype=np.float64).reshape(4, 1)


def _state_to_box(state: np.ndarray) -> np.ndarray:
    """Kalman state -> [x1, y1, x2, y2] (float32)."""
    center_x, center_y = state[0, 0], state[1, 0]
    area = max(state[2, 0], _EPS)
    aspect_ratio = max(state[3, 0], _EPS)
    width = np.sqrt(area * aspect_ratio)
    height = area / max(width, _EPS)
    return np.array(
        [center_x - width / 2.0, center_y - height / 2.0,
         center_x + width / 2.0, center_y + height / 2.0],
        dtype=np.float32,
    )


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of [x1, y1, x2, y2] boxes -> shape (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    a = boxes_a[:, None, :].astype(np.float64)
    b = boxes_b[None, :, :].astype(np.float64)

    inter_w = np.maximum(0.0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]))
    inter_h = np.maximum(0.0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]))
    intersection = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - intersection
    return intersection / np.maximum(union, _EPS)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter for one bounding box.

    State vector: [cx, cy, area, aspect_ratio, d_cx, d_cy, d_area].
    """

    def __init__(self, track_id: int, detection: Detection) -> None:
        self.track_id = track_id
        self.class_id = detection.class_id
        self.label = detection.label
        self.confidence = detection.confidence

        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.confirmed = False

        self._transition = np.eye(7)
        self._transition[0, 4] = self._transition[1, 5] = self._transition[2, 6] = 1.0
        self._measurement_matrix = np.eye(4, 7)

        self._measurement_noise = np.eye(4)
        self._measurement_noise[2:, 2:] *= 10.0

        self._covariance = np.eye(7)
        self._covariance[4:, 4:] *= 1000.0  # velocities are unobserved at the start
        self._covariance *= 10.0

        self._process_noise = np.eye(7)
        self._process_noise[-1, -1] *= 0.01
        self._process_noise[4:, 4:] *= 0.01

        self._state = np.zeros((7, 1))
        self._state[:4] = _box_to_measurement(detection.box)

    def predict(self) -> np.ndarray:
        """Advance the filter one frame and return the predicted box."""
        if self._state[6, 0] + self._state[2, 0] <= 0:
            self._state[6, 0] = 0.0  # area must not become negative
        self._state = self._transition @ self._state
        self._covariance = self._transition @ self._covariance @ self._transition.T + self._process_noise

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _state_to_box(self._state)

    def update(self, detection: Detection) -> None:
        """Correct the filter with a matched detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.class_id = detection.class_id
        self.label = detection.label
        self.confidence = detection.confidence

        measurement = _box_to_measurement(detection.box)
        innovation = measurement - self._measurement_matrix @ self._state
        innovation_cov = (
            self._measurement_matrix @ self._covariance @ self._measurement_matrix.T
            + self._measurement_noise
        )
        kalman_gain = np.linalg.solve(innovation_cov, self._measurement_matrix @ self._covariance).T
        self._state = self._state + kalman_gain @ innovation
        identity = np.eye(7)
        self._covariance = (identity - kalman_gain @ self._measurement_matrix) @ self._covariance

    @property
    def box(self) -> np.ndarray:
        return _state_to_box(self._state)


@dataclass
class TrackingResult:
    """Output of :meth:`SortTracker.update` for one frame."""

    tracks: List[TrackedObject] = field(default_factory=list)  # confirmed, with ID
    pending: List[Detection] = field(default_factory=list)  # detected, no ID yet


class SortTracker:
    """Multi-object tracker following the SORT algorithm."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        if max_age < 1:
            raise ValueError("max_age must be >= 1")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._trackers: List[KalmanBoxTracker] = []
        self._next_id = 1

    @property
    def active_track_count(self) -> int:
        return len(self._trackers)

    def update(self, detections: Sequence[Detection]) -> TrackingResult:
        """Process the detections of one frame (may be empty) and return the result.

        Must be called once per frame, even when there are no detections, so the
        Kalman filters age correctly.
        """
        # 1. Predict new locations of existing tracks (drop ones that went invalid).
        predicted_boxes: List[np.ndarray] = []
        alive: List[KalmanBoxTracker] = []
        for tracker in self._trackers:
            predicted = tracker.predict()
            if np.all(np.isfinite(predicted)):
                alive.append(tracker)
                predicted_boxes.append(predicted)
        self._trackers = alive

        # 2. Match detections to predictions.
        detection_boxes = (
            np.stack([d.box for d in detections]) if detections else np.zeros((0, 4), dtype=np.float32)
        )
        tracker_boxes = np.stack(predicted_boxes) if predicted_boxes else np.zeros((0, 4), dtype=np.float32)
        matches, unmatched_detections = self._associate(detection_boxes, tracker_boxes)

        # 3. Update matched tracks and create tracks for unmatched detections.
        result = TrackingResult()
        for detection_index, tracker_index in matches:
            tracker = self._trackers[tracker_index]
            detection = detections[detection_index]
            tracker.update(detection)
            self._report(tracker, detection, result)

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            tracker = KalmanBoxTracker(self._next_id, detection)
            self._next_id += 1
            self._trackers.append(tracker)
            self._report(tracker, detection, result)

        # 4. Remove tracks that have been lost for too long.
        self._trackers = [t for t in self._trackers if t.time_since_update <= self.max_age]
        return result

    def _report(self, tracker: KalmanBoxTracker, detection: Detection, result: TrackingResult) -> None:
        """Add a track updated this frame either to confirmed tracks or to pending detections."""
        if not tracker.confirmed and tracker.hit_streak >= self.min_hits:
            tracker.confirmed = True
        if tracker.confirmed:
            result.tracks.append(
                TrackedObject(
                    track_id=tracker.track_id,
                    box=tracker.box,
                    confidence=tracker.confidence,
                    class_id=tracker.class_id,
                    label=tracker.label,
                )
            )
        else:
            result.pending.append(detection)

    def _associate(
        self, detection_boxes: np.ndarray, tracker_boxes: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Hungarian matching on IoU. Returns (matches, unmatched_detection_indices)."""
        if len(tracker_boxes) == 0:
            return [], list(range(len(detection_boxes)))
        if len(detection_boxes) == 0:
            return [], []

        iou = iou_matrix(detection_boxes, tracker_boxes)
        row_indices, col_indices = linear_sum_assignment(-iou)

        matches: List[Tuple[int, int]] = []
        matched_detections = set()
        for row, col in zip(row_indices, col_indices):
            if iou[row, col] >= self.iou_threshold:
                matches.append((int(row), int(col)))
                matched_detections.add(int(row))
        unmatched = [i for i in range(len(detection_boxes)) if i not in matched_detections]
        return matches, unmatched
