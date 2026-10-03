"""Unit tests for the SORT tracker (no model, camera or GUI needed).

Run from the project root:  python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from tracker.sort_tracker import SortTracker, iou_matrix
from utils.types import Detection


def make_detection(x1, y1, x2, y2, label="person", confidence=0.9, class_id=0) -> Detection:
    return Detection(box=np.array([x1, y1, x2, y2], dtype=np.float32),
                     confidence=confidence, class_id=class_id, label=label)


class IouTests(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self):
        box = np.array([[0, 0, 10, 10]], dtype=np.float32)
        self.assertAlmostEqual(float(iou_matrix(box, box)[0, 0]), 1.0, places=5)

    def test_disjoint_boxes_have_iou_zero(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 20, 30, 30]], dtype=np.float32)
        self.assertEqual(float(iou_matrix(a, b)[0, 0]), 0.0)

    def test_empty_inputs(self):
        empty = np.zeros((0, 4), dtype=np.float32)
        self.assertEqual(iou_matrix(empty, empty).shape, (0, 0))


class SortTrackerTests(unittest.TestCase):
    def test_empty_detections_do_not_crash(self):
        tracker = SortTracker()
        for _ in range(5):
            result = tracker.update([])
            self.assertEqual(result.tracks, [])
            self.assertEqual(result.pending, [])

    def test_id_is_only_assigned_after_min_hits(self):
        tracker = SortTracker(min_hits=3)
        first = tracker.update([make_detection(10, 10, 60, 110)])
        second = tracker.update([make_detection(12, 10, 62, 110)])
        third = tracker.update([make_detection(14, 10, 64, 110)])
        self.assertEqual((len(first.tracks), len(first.pending)), (0, 1))
        self.assertEqual((len(second.tracks), len(second.pending)), (0, 1))
        self.assertEqual(len(third.tracks), 1)
        self.assertEqual(third.tracks[0].track_id, 1)
        self.assertEqual(third.tracks[0].label, "person")

    def test_id_persists_for_moving_object(self):
        tracker = SortTracker(min_hits=2)
        ids = set()
        for step in range(20):
            result = tracker.update([make_detection(10 + 4 * step, 20, 60 + 4 * step, 120)])
            ids.update(t.track_id for t in result.tracks)
        self.assertEqual(ids, {1})

    def test_two_objects_get_different_ids(self):
        tracker = SortTracker(min_hits=1)
        result = tracker.update([make_detection(0, 0, 50, 100), make_detection(300, 0, 350, 100, "car", 0.8, 2)])
        self.assertEqual(sorted(t.track_id for t in result.tracks), [1, 2])
        self.assertEqual(sorted(t.label for t in result.tracks), ["car", "person"])

    def test_lost_track_is_not_reported_and_is_removed_after_max_age(self):
        tracker = SortTracker(max_age=3, min_hits=1)
        tracker.update([make_detection(0, 0, 50, 100)])
        for _ in range(2):
            self.assertEqual(tracker.update([]).tracks, [])  # coasting: no ID is claimed
        self.assertEqual(tracker.active_track_count, 1)
        for _ in range(3):
            tracker.update([])
        self.assertEqual(tracker.active_track_count, 0)

    def test_track_survives_short_occlusion_with_same_id(self):
        tracker = SortTracker(max_age=10, min_hits=2)
        for step in range(4):
            tracker.update([make_detection(10 + 3 * step, 10, 60 + 3 * step, 110)])
        for _ in range(3):
            tracker.update([])
        result = tracker.update([make_detection(10 + 3 * 8, 10, 60 + 3 * 8, 110)])
        self.assertEqual([t.track_id for t in result.tracks], [1])

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            SortTracker(max_age=0)
        with self.assertRaises(ValueError):
            SortTracker(min_hits=0)
        with self.assertRaises(ValueError):
            SortTracker(iou_threshold=0)


if __name__ == "__main__":
    unittest.main()
