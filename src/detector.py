import os
import logging
from typing import List, Dict, Any, Optional
import torch
from ultralytics import YOLO

logger = logging.getLogger("TrafficMonitor.Detector")

# Default COCO classes for vehicles: 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
DEFAULT_VEHICLE_CLASSES = [1, 2, 3, 5, 7]


class DualDetector:
    """
    Manages dual YOLO models:
    - Custom-trained YOLO11s for road accident detection.
    - Pretrained YOLO11n for vehicle detection.
    """

    def __init__(
        self,
        accident_weights: str = "models/accident_yolo11s.pt",
        vehicle_weights: str = "yolo11n.pt",
        accident_conf: float = 0.40,
        vehicle_conf: float = 0.35,
        accident_iou: float = 0.45,
        vehicle_iou: float = 0.50,
        vehicle_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        self.device = self._resolve_device(device)
        self.accident_conf = accident_conf
        self.vehicle_conf = vehicle_conf
        self.accident_iou = accident_iou
        self.vehicle_iou = vehicle_iou
        self.vehicle_classes = vehicle_classes or DEFAULT_VEHICLE_CLASSES

        # Load Accident Model
        self.accident_weights = self._resolve_weights(
            accident_weights,
            fallback="models/accident_yolo11s.pt",
            description="accident",
        )
        logger.info(f"Loading accident detector from '{self.accident_weights}' on {self.device}...")
        self.accident_model = YOLO(self.accident_weights)

        # Detect whether the loaded model contains explicit accident/crash classes
        self.accident_class_ids = [
            cid for cid, cname in self.accident_model.names.items()
            if any(k in str(cname).lower() for k in ["accident", "crash", "collision", "incident", "damage", "wreck"])
        ]
        self.has_accident_class = len(self.accident_class_ids) > 0
        if not self.has_accident_class:
            logger.warning(
                f"Model '{self.accident_weights}' does not contain dedicated 'accident' or 'crash' classes. "
                "Base vehicle classes will NOT be mislabeled as accidents. "
                "Please use a custom-trained accident weights file (e.g. 'models/accident_yolo11s.pt')."
            )
        else:
            acc_names = [self.accident_model.names[cid] for cid in self.accident_class_ids]
            logger.info(f"Accident detector active with accident class IDs: {self.accident_class_ids} ({acc_names})")

        # Load Vehicle Model (YOLO11n)
        self.vehicle_weights = self._resolve_weights(
            vehicle_weights,
            fallback="yolo11n.pt",
            description="vehicle",
        )
        logger.info(f"Loading vehicle detector from '{self.vehicle_weights}' on {self.device}...")
        self.vehicle_model = YOLO(self.vehicle_weights)

    def _resolve_device(self, requested_device: Optional[str]) -> str:
        if requested_device:
            if requested_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            return requested_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_weights(self, path: str, fallback: str, description: str) -> str:
        """Resolves weights path, checking direct path, models/ subdirectory, or fallback."""
        if os.path.exists(path):
            return path
        # Check inside models/ directory
        candidate = os.path.join("models", os.path.basename(path))
        if os.path.exists(candidate):
            return candidate

        logger.warning(
            f"{description.capitalize()} model weights not found at '{path}'. Falling back to '{fallback}'."
        )
        return fallback

    def detect_accidents(self, frame) -> List[Dict[str, Any]]:
        """
        Run inference using the custom YOLO accident detector.
        Returns a list of accident detection dictionaries with bounding boxes and confidence scores.
        """
        if not self.has_accident_class:
            return []

        results = self.accident_model.predict(
            source=frame,
            conf=self.accident_conf,
            iou=self.accident_iou,
            device=self.device,
            verbose=False,
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.accident_model.names.get(cls_id, "")

            # Filter: only accept genuine accident / crash class predictions
            if cls_id in self.accident_class_ids or any(
                k in str(cls_name).lower() for k in ["accident", "crash", "collision", "incident"]
            ):
                detections.append({
                    "bbox": [float(x) for x in xyxy],
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name if "accident" in cls_name.lower() else "accident",
                })

        return detections
