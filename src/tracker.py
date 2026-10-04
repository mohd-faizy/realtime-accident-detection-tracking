import logging
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("TrafficMonitor.Tracker")


class TrackedVehicle:
    """
    Data structure representing a vehicle tracked across frames.
    """
    def __init__(
        self,
        track_id: int,
        bbox: List[float],
        conf: float,
        class_id: int,
        class_name: str,
        max_history: int = 30,
    ):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.conf = conf
        self.class_id = class_id
        self.class_name = class_name
        self.is_suppressed = False
        self.accident_id: Optional[int] = None
        self.history: deque = deque(maxlen=max_history)
        self.update_centroid(bbox)

    @property
    def centroid(self) -> List[float]:
        x1, y1, x2, y2 = self.bbox
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    def update_centroid(self, bbox: List[float]):
        self.bbox = bbox
        self.history.append(self.centroid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "conf": self.conf,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "is_suppressed": self.is_suppressed,
            "accident_id": self.accident_id,
            "centroid": self.centroid,
        }


class VehicleTracker:
    """
    Tracks vehicles using BoT-SORT algorithm integrated with YOLO11n.
    Maintains persistent IDs and track histories across video frames.
    """

    def __init__(
        self,
        vehicle_model: YOLO,
        tracker_type: str = "botsort.yaml",
        conf: float = 0.35,
        iou: float = 0.50,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
        max_history: int = 30,
    ):
        self.model = vehicle_model
        self.tracker_type = tracker_type
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.device = device
        self.max_history = max_history

        # Active tracks dictionary: track_id -> TrackedVehicle
        self.active_tracks: Dict[int, TrackedVehicle] = {}

    def update(self, frame: np.ndarray) -> List[TrackedVehicle]:
        """
        Runs BoT-SORT tracking on the current frame.
        Returns a list of currently active TrackedVehicle objects.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        current_frame_tracks: List[TrackedVehicle] = []
        if not results or len(results) == 0:
            return current_frame_tracks

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return current_frame_tracks

        # Check if tracking IDs are available
        has_ids = boxes.id is not None
        track_ids = (
            boxes.id.int().cpu().tolist() if has_ids else list(range(len(boxes)))
        )

        for i, box in enumerate(boxes):
            track_id = int(track_ids[i])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.model.names.get(cls_id, f"vehicle_{cls_id}")

            if track_id in self.active_tracks:
                vehicle = self.active_tracks[track_id]
                vehicle.update_centroid(xyxy)
                vehicle.conf = conf
                vehicle.is_suppressed = False  # Reset each frame before spatial logic
                vehicle.accident_id = None
            else:
                vehicle = TrackedVehicle(
                    track_id=track_id,
                    bbox=xyxy,
                    conf=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    max_history=self.max_history,
                )
                self.active_tracks[track_id] = vehicle

            current_frame_tracks.append(vehicle)

        # Cleanup tracks that haven't been seen for a long time (retain active set)
        current_ids = set(track_ids)
        stale_ids = [tid for tid in self.active_tracks if tid not in current_ids]
        # Keep stale tracks around briefly if needed, or prune
        for tid in stale_ids:
            # We can remove stale tracks after not appearing
            del self.active_tracks[tid]

        return current_frame_tracks

    def reset(self):
        """Reset all tracked state."""
        self.active_tracks.clear()
