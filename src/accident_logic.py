import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("TrafficMonitor.Logic")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0.0, x2_inter - x1_inter)
    inter_height = max(0.0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def calculate_containment(inner_box: List[float], outer_box: List[float]) -> float:
    """
    Calculate the containment ratio: Area(inner_box ∩ outer_box) / Area(inner_box).
    Determines how much of the vehicle box is engulfed within the accident region.
    """
    x1_inter = max(inner_box[0], outer_box[0])
    y1_inter = max(inner_box[1], outer_box[1])
    x2_inter = min(inner_box[2], outer_box[2])
    y2_inter = min(inner_box[3], outer_box[3])

    inter_width = max(0.0, x2_inter - x1_inter)
    inter_height = max(0.0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    if inter_area <= 0.0:
        return 0.0

    inner_area = max(0.0, inner_box[2] - inner_box[0]) * max(0.0, inner_box[3] - inner_box[1])
    if inner_area <= 0.0:
        return 0.0

    return inter_area / inner_area


@dataclass
class AccidentZone:
    """
    Represents an ongoing detected or maintained accident region.
    """
    zone_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    conf: float
    lost_frames: int = 0
    consecutive_hits: int = 1
    status: str = "ACTIVE"  # "ACTIVE" or "MAINTAINED"
    involved_vehicle_ids: List[int] = field(default_factory=list)

    def smooth_bbox(self, new_bbox: List[float], alpha: float = 0.6):
        """Exponential moving average smoothing for bounding box coordinates."""
        self.bbox = [
            alpha * n + (1.0 - alpha) * o for n, o in zip(new_bbox, self.bbox)
        ]


class AccidentPersistenceManager:
    """
    Maintains accident bounding boxes across frames during temporary detection loss.
    Provides anti-flicker stability so accident alerts remain steady even if
    the detector drops detection for a few frames due to occlusion or motion blur.
    """

    def __init__(
        self,
        persistence_frames: int = 15,
        iou_match_threshold: float = 0.30,
        smoothing_alpha: float = 0.70,
    ):
        self.persistence_frames = persistence_frames
        self.iou_match_threshold = iou_match_threshold
        self.smoothing_alpha = smoothing_alpha

        self.next_zone_id: int = 1
        self.active_zones: Dict[int, AccidentZone] = {}

    def update(self, raw_detections: List[Dict[str, Any]]) -> List[AccidentZone]:
        """
        Update accident zones with raw detections from the current frame.
        - Matches detections to existing zones via IoU.
        - Maintains zones during temporary loss up to `persistence_frames`.
        - Purges expired zones.
        """
        matched_zone_ids = set()
        matched_det_indices = set()

        # 1. Match existing zones to current detections
        for zid, zone in self.active_zones.items():
            best_iou = 0.0
            best_det_idx = -1

            for idx, det in enumerate(raw_detections):
                if idx in matched_det_indices:
                    continue
                iou = calculate_iou(zone.bbox, det["bbox"])
                if iou > best_iou and iou >= self.iou_match_threshold:
                    best_iou = iou
                    best_det_idx = idx

            if best_det_idx >= 0:
                # Update matched zone
                matched_det = raw_detections[best_det_idx]
                zone.smooth_bbox(matched_det["bbox"], alpha=self.smoothing_alpha)
                zone.conf = matched_det["conf"]
                zone.lost_frames = 0
                zone.consecutive_hits += 1
                zone.status = "ACTIVE"
                zone.involved_vehicle_ids.clear()

                matched_zone_ids.add(zid)
                matched_det_indices.add(best_det_idx)

        # 2. Handle unmatched existing zones (temporary detection loss)
        stale_zone_ids = []
        for zid, zone in self.active_zones.items():
            if zid not in matched_zone_ids:
                zone.lost_frames += 1
                if zone.lost_frames <= self.persistence_frames:
                    zone.status = "MAINTAINED"
                    zone.involved_vehicle_ids.clear()
                else:
                    stale_zone_ids.append(zid)

        # Remove stale/expired zones
        for zid in stale_zone_ids:
            logger.info(f"Accident zone #{zid} cleared after {self.persistence_frames} lost frames.")
            del self.active_zones[zid]

        # 3. Create new zones for unmatched detections
        for idx, det in enumerate(raw_detections):
            if idx not in matched_det_indices:
                new_zone = AccidentZone(
                    zone_id=self.next_zone_id,
                    bbox=det["bbox"],
                    conf=det["conf"],
                    lost_frames=0,
                    consecutive_hits=1,
                    status="ACTIVE",
                )
                self.active_zones[self.next_zone_id] = new_zone
                logger.info(
                    f"New accident zone #{self.next_zone_id} detected (conf={det['conf']:.2f})."
                )
                self.next_zone_id += 1

        return list(self.active_zones.values())

    def reset(self):
        """Clear all active accident zones."""
        self.active_zones.clear()
        self.next_zone_id = 1


def suppress_vehicles_in_accidents(
    vehicles: List[Any],
    accident_zones: List[AccidentZone],
    containment_threshold: float = 0.50,
    iou_threshold: float = 0.30,
) -> Tuple[List[Any], List[AccidentZone]]:
    """
    Suppresses normal vehicle labels inside confirmed accident regions.
    If a vehicle is >= containment_threshold inside an accident bounding box,
    or has an IoU >= iou_threshold, it is marked as suppressed and associated
    with the accident zone.
    """
    for vehicle in vehicles:
        vehicle.is_suppressed = False
        vehicle.accident_id = None

        for zone in accident_zones:
            containment = calculate_containment(vehicle.bbox, zone.bbox)
            iou = calculate_iou(vehicle.bbox, zone.bbox)

            if containment >= containment_threshold or iou >= iou_threshold:
                vehicle.is_suppressed = True
                vehicle.accident_id = zone.zone_id
                if vehicle.track_id not in zone.involved_vehicle_ids:
                    zone.involved_vehicle_ids.append(vehicle.track_id)
                break  # Suppressed by at least one accident zone

    return vehicles, accident_zones


def detect_vehicle_collisions(
    tracked_vehicles: List[Any],
    iou_collision_threshold: float = 0.25,
    containment_collision_threshold: float = 0.40,
) -> List[Dict[str, Any]]:
    """
    Detects potential vehicle-vehicle collision events directly from multi-object tracking.
    When two separate vehicles physically collide or overlap beyond the collision thresholds,
    a collision candidate region is generated enclosing both vehicles.
    """
    collision_detections = []
    num_v = len(tracked_vehicles)
    for i in range(num_v):
        for j in range(i + 1, num_v):
            v1 = tracked_vehicles[i]
            v2 = tracked_vehicles[j]

            # Require both vehicles to have tracked history to avoid new detection jitter
            h1 = getattr(v1, "history", None)
            h2 = getattr(v2, "history", None)
            if h1 is not None and len(h1) < 2:
                continue
            if h2 is not None and len(h2) < 2:
                continue

            iou = calculate_iou(v1.bbox, v2.bbox)
            c1 = calculate_containment(v1.bbox, v2.bbox)
            c2 = calculate_containment(v2.bbox, v1.bbox)
            max_cont = max(c1, c2)

            if iou >= iou_collision_threshold or max_cont >= containment_collision_threshold:
                # Bounding box enveloping both colliding vehicles
                col_box = [
                    min(v1.bbox[0], v2.bbox[0]),
                    min(v1.bbox[1], v2.bbox[1]),
                    max(v1.bbox[2], v2.bbox[2]),
                    max(v1.bbox[3], v2.bbox[3]),
                ]
                conf = min(0.95, max(iou * 1.5, max_cont * 1.2, 0.60))
                collision_detections.append({
                    "bbox": col_box,
                    "conf": float(conf),
                    "class_id": -1,
                    "class_name": "collision",
                    "vehicle_ids": [v1.track_id, v2.track_id],
                })

    return collision_detections
