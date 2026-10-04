"""
Road Accident Detection & Vehicle Tracking System.
Combines custom-trained YOLO11s (accident detection) and pretrained YOLO11n (vehicle detection)
with BoT-SORT multi-object tracking, spatial containment filtering, and accident persistence logic.
"""

__version__ = "2.0.0"

from .detector import DualDetector
from .tracker import VehicleTracker, TrackedVehicle
from .accident_logic import (
    AccidentZone,
    AccidentPersistenceManager,
    calculate_iou,
    calculate_containment,
    suppress_vehicles_in_accidents,
    detect_vehicle_collisions,
)
from .visualizer import Visualizer

__all__ = [
    "DualDetector",
    "VehicleTracker",
    "TrackedVehicle",
    "AccidentZone",
    "AccidentPersistenceManager",
    "calculate_iou",
    "calculate_containment",
    "suppress_vehicles_in_accidents",
    "detect_vehicle_collisions",
    "Visualizer",
]
