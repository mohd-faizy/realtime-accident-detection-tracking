import pytest
from src.accident_logic import (
    calculate_iou,
    calculate_containment,
    AccidentPersistenceManager,
    suppress_vehicles_in_accidents,
    AccidentZone,
)


class DummyVehicle:
    """Mock TrackedVehicle for testing spatial containment and suppression."""
    def __init__(self, track_id: int, bbox, class_name="car"):
        self.track_id = track_id
        self.bbox = bbox
        self.class_name = class_name
        self.is_suppressed = False
        self.accident_id = None


def test_iou_calculation():
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    assert calculate_iou(box1, box2) == 1.0

    box3 = [20, 20, 30, 30]
    assert calculate_iou(box1, box3) == 0.0

    # 50% overlap: intersection is 5x10 = 50, union is 100 + 100 - 50 = 150 -> 1/3
    box4 = [5, 0, 15, 10]
    assert pytest.approx(calculate_iou(box1, box4), 0.01) == 0.333


def test_containment_calculation():
    # Vehicle inside larger accident box
    vehicle_box = [2, 2, 8, 8]  # area = 6x6 = 36
    accident_box = [0, 0, 10, 10]  # area = 100
    # Complete containment of vehicle
    assert calculate_containment(vehicle_box, accident_box) == 1.0

    # 50% of vehicle inside accident box
    # Vehicle area 10x10 = 100, intersection 5x10 = 50
    vehicle_box2 = [5, 0, 15, 10]
    assert pytest.approx(calculate_containment(vehicle_box2, accident_box), 0.01) == 0.50

    # Non-overlapping
    outside_box = [20, 20, 30, 30]
    assert calculate_containment(outside_box, accident_box) == 0.0


def test_accident_persistence_temporary_loss():
    mgr = AccidentPersistenceManager(persistence_frames=3, iou_match_threshold=0.3)

    # Frame 1: Detection present
    dets_f1 = [{"bbox": [100.0, 100.0, 200.0, 200.0], "conf": 0.85, "class_name": "accident"}]
    zones_f1 = mgr.update(dets_f1)
    assert len(zones_f1) == 1
    assert zones_f1[0].status == "ACTIVE"
    assert zones_f1[0].lost_frames == 0

    # Frame 2: Missed detection (temporary loss frame 1)
    zones_f2 = mgr.update([])
    assert len(zones_f2) == 1
    assert zones_f2[0].status == "MAINTAINED"
    assert zones_f2[0].lost_frames == 1

    # Frame 3: Missed detection (temporary loss frame 2)
    zones_f3 = mgr.update([])
    assert len(zones_f3) == 1
    assert zones_f3[0].status == "MAINTAINED"
    assert zones_f3[0].lost_frames == 2

    # Frame 4: Redetected before expiry!
    dets_f4 = [{"bbox": [102.0, 98.0, 201.0, 199.0], "conf": 0.88, "class_name": "accident"}]
    zones_f4 = mgr.update(dets_f4)
    assert len(zones_f4) == 1
    assert zones_f4[0].status == "ACTIVE"
    assert zones_f4[0].lost_frames == 0


def test_accident_expires_after_max_loss_frames():
    mgr = AccidentPersistenceManager(persistence_frames=2, iou_match_threshold=0.3)

    # Frame 1: Detected
    mgr.update([{"bbox": [50.0, 50.0, 150.0, 150.0], "conf": 0.90, "class_name": "accident"}])

    # Lost 1 frame -> maintained
    z1 = mgr.update([])
    assert len(z1) == 1
    assert z1[0].lost_frames == 1

    # Lost 2 frames -> maintained
    z2 = mgr.update([])
    assert len(z2) == 1
    assert z2[0].lost_frames == 2

    # Lost 3 frames (> persistence_frames 2) -> expired and purged!
    z3 = mgr.update([])
    assert len(z3) == 0


def test_suppress_vehicles_in_accidents():
    accident = AccidentZone(
        zone_id=1,
        bbox=[100.0, 100.0, 300.0, 300.0],
        conf=0.92,
        status="ACTIVE",
    )

    # Vehicle 1: Completely inside accident wreckage
    v1 = DummyVehicle(track_id=10, bbox=[120.0, 120.0, 180.0, 180.0])

    # Vehicle 2: Safely passing by in another lane
    v2 = DummyVehicle(track_id=11, bbox=[500.0, 500.0, 600.0, 600.0])

    vehicles = [v1, v2]
    suppressed_vehicles, updated_accidents = suppress_vehicles_in_accidents(
        vehicles=vehicles,
        accident_zones=[accident],
        containment_threshold=0.50,
        iou_threshold=0.30,
    )

    # Vehicle 1 should be suppressed and linked to accident zone #1
    assert v1.is_suppressed is True
    assert v1.accident_id == 1
    assert 10 in accident.involved_vehicle_ids

    # Vehicle 2 should remain active and unsuppressed
    assert v2.is_suppressed is False
    assert v2.accident_id is None
    assert 11 not in accident.involved_vehicle_ids


def test_detect_vehicle_collisions_no_overlap():
    from src.accident_logic import detect_vehicle_collisions
    v1 = DummyVehicle(track_id=1, bbox=[0, 0, 50, 50])
    v1.history = [(25, 25), (25, 26), (25, 27)]
    v2 = DummyVehicle(track_id=2, bbox=[200, 200, 250, 250])
    v2.history = [(225, 225), (225, 226), (225, 227)]

    cols = detect_vehicle_collisions([v1, v2])
    assert len(cols) == 0


def test_detect_vehicle_collisions_overlapping():
    from src.accident_logic import detect_vehicle_collisions
    # Two vehicles colliding with significant bounding box overlap
    v1 = DummyVehicle(track_id=1, bbox=[100, 100, 160, 160])
    v1.history = [(130, 130), (130, 132), (130, 135)]
    v2 = DummyVehicle(track_id=2, bbox=[120, 100, 180, 160])
    v2.history = [(150, 130), (150, 132), (150, 135)]

    cols = detect_vehicle_collisions([v1, v2])
    assert len(cols) == 1
    assert cols[0]["class_name"] == "collision"
    assert 1 in cols[0]["vehicle_ids"]
    assert 2 in cols[0]["vehicle_ids"]


def test_package_exports():
    import src
    expected_exports = [
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
    for item in expected_exports:
        assert hasattr(src, item), f"Missing public export: {item}"

