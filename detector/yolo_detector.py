"""YOLO object detector (Ultralytics) running fully locally."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from utils.types import Detection


class DetectorError(RuntimeError):
    """Raised when the detector cannot be created or run (user-facing message)."""


def _import_ml_stack():
    """Import torch and Ultralytics lazily so a missing install gives a clear message."""
    try:
        import torch
    except ImportError as exc:
        raise DetectorError(
            "PyTorch is not installed. Run:  pip install -r requirements.txt"
        ) from exc
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectorError(
            "Ultralytics is not installed. Run:  pip install -r requirements.txt"
        ) from exc
    return torch, YOLO


def resolve_device(requested: str = "auto") -> str:
    """Return the torch device string to use: 'cuda:0' if available (auto), else 'cpu'."""
    torch, _ = _import_ml_stack()
    requested = requested.lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise DetectorError(
            "CUDA was requested but no CUDA-capable GPU/PyTorch build was found. "
            "Use --device cpu (or --device auto)."
        )
    if requested != "cpu" and not requested.startswith("cuda"):
        raise DetectorError(f"Unsupported device '{requested}'. Use auto, cpu or cuda.")
    return "cuda:0" if requested == "cuda" else requested


class YoloDetector:
    """Thin wrapper around an Ultralytics YOLO model returning :class:`Detection` objects."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        nms_iou_threshold: float = 0.5,
        image_size: int = 640,
        device: str = "auto",
        download_if_missing: bool = False,
        class_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if not 0.0 < confidence_threshold < 1.0:
            raise DetectorError("Confidence threshold must be between 0 and 1 (e.g. 0.35).")
        if not 0.0 < nms_iou_threshold <= 1.0:
            raise DetectorError("NMS IoU threshold must be in (0, 1].")
        if image_size < 32:
            raise DetectorError("Image size must be at least 32 (e.g. 640).")

        _, yolo_class = _import_ml_stack()
        self.device = resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.image_size = image_size
        self.class_ids = list(class_ids) if class_ids else None

        path = self._locate_model(Path(model_path), download_if_missing)
        try:
            self._model = yolo_class(str(path))
        except Exception as exc:  # Ultralytics raises various types for corrupt/invalid files
            raise DetectorError(
                f"Could not load the model file '{path}'. The file may be corrupted or not a "
                f"YOLO .pt model. Delete it and download it again. Details: {exc}"
            ) from exc

        self.class_names = dict(self._model.names)
        self._warm_up()

    @staticmethod
    def _locate_model(path: Path, download_if_missing: bool) -> Path:
        """Return an existing model file, downloading an official one if allowed."""
        if path.is_file():
            return path
        if path.exists():
            raise DetectorError(f"Model path '{path}' is not a file.")

        if not download_if_missing:
            raise DetectorError(
                f"Model file not found: '{path}'.\n"
                "  Option 1: run again with --download-model to fetch the official weights "
                "automatically (needs internet once).\n"
                "  Option 2: download the .pt file manually and place it there "
                "(see models/README.md)."
            )

        try:
            from ultralytics.utils.downloads import attempt_download_asset
        except ImportError as exc:
            raise DetectorError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model '{path.name}' from the official Ultralytics releases...")
        try:
            attempt_download_asset(str(path))
        except Exception as exc:  # network / HTTP errors come in several flavours
            raise DetectorError(
                f"Automatic download of '{path.name}' failed (check your internet connection): {exc}"
            ) from exc

        if not path.is_file():
            raise DetectorError(
                f"'{path.name}' is not an official Ultralytics model name, so it cannot be "
                "downloaded automatically. Place the file manually (see models/README.md)."
            )
        return path

    def _warm_up(self) -> None:
        """Run one dummy inference so the first real frame is not slow."""
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.detect(dummy)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on one BGR frame. Returns an empty list when nothing is found."""
        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            classes=self.class_ids,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        return [
            Detection(
                box=xyxy[i],
                confidence=float(confidences[i]),
                class_id=int(class_ids[i]),
                label=str(self.class_names.get(int(class_ids[i]), int(class_ids[i]))),
            )
            for i in range(len(xyxy))
        ]
