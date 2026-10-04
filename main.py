#!/usr/bin/env python3
"""
Accident Detection & Vehicle Tracking System.

Main CLI pipeline integrating:
- Custom-trained YOLO11s for accident detection
- Pretrained YOLO11n for vehicle detection
- BoT-SORT multi-object tracking with unique IDs
- Confidence filtering, NMS, IoU, and containment-based spatial matching
- Accident bounding box persistence during temporary detection loss
- Normal vehicle label suppression inside confirmed accident regions
- Real-time emergency alert display and annotated video production
"""

import os
import sys
import time
import argparse
import logging
import cv2
import yaml
from tqdm import tqdm

from src import (
    DualDetector,
    VehicleTracker,
    AccidentPersistenceManager,
    suppress_vehicles_in_accidents,
    detect_vehicle_collisions,
    Visualizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrafficMonitor")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load settings from YAML configuration file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file '{config_path}' not found. Using internal defaults.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Road Accident Detection & Vehicle Tracking System (YOLO11s + YOLO11n + BoT-SORT)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Path to input video file, image, or webcam index (default: '0')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/output.mp4",
        help="Path for saving annotated output video (default: 'outputs/output.mp4')",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: 'config.yaml')",
    )
    parser.add_argument(
        "--accident-model",
        type=str,
        default=None,
        help="Override path to custom YOLO11s accident model weights",
    )
    parser.add_argument(
        "--vehicle-model",
        type=str,
        default=None,
        help="Override path to YOLO11n vehicle model weights",
    )
    parser.add_argument(
        "--conf-accident",
        type=float,
        default=None,
        help="Override accident detection confidence threshold",
    )
    parser.add_argument(
        "--conf-vehicle",
        type=float,
        default=None,
        help="Override vehicle detection confidence threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live annotated video preview window",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving the output video to disk",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve settings (CLI arguments take precedence over config.yaml)
    device_cfg = args.device or cfg.get("device", "auto")
    device = None if device_cfg == "auto" else device_cfg

    m_cfg = cfg.get("models", {})
    acc_cfg = m_cfg.get("accident", {})
    veh_cfg = m_cfg.get("vehicle", {})

    accident_weights = args.accident_model or acc_cfg.get("weights", "models/accident_yolo11s.pt")
    vehicle_weights = args.vehicle_model or veh_cfg.get("weights", "models/yolo11n.pt")
    accident_conf = args.conf_accident if args.conf_accident is not None else acc_cfg.get("conf_threshold", 0.40)
    vehicle_conf = args.conf_vehicle if args.conf_vehicle is not None else veh_cfg.get("conf_threshold", 0.35)
    accident_iou = acc_cfg.get("iou_threshold", 0.45)
    vehicle_iou = veh_cfg.get("iou_threshold", 0.50)
    vehicle_classes = veh_cfg.get("classes", [1, 2, 3, 5, 7])

    t_cfg = cfg.get("tracker", {})
    tracker_type = t_cfg.get("type", "botsort.yaml")
    max_history = t_cfg.get("max_history", 30)

    l_cfg = cfg.get("logic", {})
    containment_threshold = l_cfg.get("containment_threshold", 0.50)
    iou_match_threshold = l_cfg.get("iou_match_threshold", 0.30)
    persistence_frames = l_cfg.get("persistence_frames", 15)
    smoothing_alpha = l_cfg.get("smoothing_alpha", 0.70)
    suppress_labels = l_cfg.get("suppress_vehicle_labels", True)

    v_cfg = cfg.get("visualization", {})
    visualizer = Visualizer(
        show_tracking_ids=v_cfg.get("show_tracking_ids", True),
        show_confidence=v_cfg.get("show_confidence", True),
        show_alert_banner=v_cfg.get("show_alert_banner", True),
        show_hud_stats=v_cfg.get("show_hud_stats", True),
        suppress_vehicle_labels=suppress_labels,
        line_thickness=v_cfg.get("line_thickness", 2),
    )

    logger.info("=" * 60)
    logger.info("Initializing Road Accident Detection & Vehicle Tracking System")
    logger.info(f"Accident Model: {accident_weights}")
    logger.info(f"Vehicle Model:  {vehicle_weights}")
    logger.info(f"Tracker:        BoT-SORT ({tracker_type})")
    logger.info(f"Spatial Logic:  Containment Thresh={containment_threshold}, Persistence={persistence_frames} frames")
    logger.info("=" * 60)

    # Initialize Dual Detector
    detector = DualDetector(
        accident_weights=accident_weights,
        vehicle_weights=vehicle_weights,
        accident_conf=accident_conf,
        vehicle_conf=vehicle_conf,
        accident_iou=accident_iou,
        vehicle_iou=vehicle_iou,
        vehicle_classes=vehicle_classes,
        device=device,
    )

    # Initialize Tracker using the loaded vehicle model
    tracker = VehicleTracker(
        vehicle_model=detector.vehicle_model,
        tracker_type=tracker_type,
        conf=vehicle_conf,
        iou=vehicle_iou,
        classes=vehicle_classes,
        device=detector.device,
        max_history=max_history,
    )

    # Initialize Accident Persistence Manager
    persistence_mgr = AccidentPersistenceManager(
        persistence_frames=persistence_frames,
        iou_match_threshold=iou_match_threshold,
        smoothing_alpha=smoothing_alpha,
    )

    # Open video source (auto-resolving relative paths inside assets/ if needed)
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
        if not os.path.exists(source):
            asset_candidate = os.path.join("assets", os.path.basename(source))
            if os.path.exists(asset_candidate):
                logger.info(f"Resolved video source '{args.source}' to '{asset_candidate}'")
                source = asset_candidate

    cap = cv2.VideoCapture(source)
    if not cap.isOpened() and (args.source == "0" or args.source == 0):
        # Graceful fallback if camera 0 is not available
        sample_path = os.path.join("assets", "sample_traffic.mp4")
        if os.path.exists(sample_path):
            logger.warning(
                f"Webcam index 0 could not be opened. Automatically falling back to sample video: '{sample_path}'"
            )
            source = sample_path
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize VideoWriter if saving output
    writer = None
    if not args.no_save and args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        logger.info(f"Saving output video to: {args.output}")

    logger.info(f"Source: {source} | Resolution: {width}x{height} | Target FPS: {fps:.2f}")

    frame_idx = 0
    start_time = time.time()
    accident_events_total = 0
    unique_tracked_ids = set()

    # Progress bar for video files (or continuous counter for webcam)
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing Frames", unit="frame")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()

            # 1. Track Vehicles with Pretrained YOLO11n + BoT-SORT
            tracked_vehicles = tracker.update(frame)
            for v in tracked_vehicles:
                unique_tracked_ids.add(v.track_id)

            # 2. Detect Accidents with Custom YOLO Detector
            raw_accidents = detector.detect_accidents(frame)

            # 3. Detect Vehicle-Vehicle Collisions from tracking
            collisions = detect_vehicle_collisions(tracked_vehicles)
            all_accident_candidates = raw_accidents + collisions

            # 4. Maintain Accident Zones across temporary loss
            active_accidents = persistence_mgr.update(all_accident_candidates)

            # 5. Containment Matching & Suppression of normal labels inside accident zones
            if active_accidents:
                accident_events_total += len(active_accidents)
                tracked_vehicles, active_accidents = suppress_vehicles_in_accidents(
                    vehicles=tracked_vehicles,
                    accident_zones=active_accidents,
                    containment_threshold=containment_threshold,
                    iou_threshold=iou_match_threshold,
                )

            # Measure processing FPS
            dt = time.time() - t0
            proc_fps = 1.0 / dt if dt > 0 else fps

            # 5. Render Annotations, Alerts, and HUD
            annotated_frame = visualizer.render(
                frame=frame,
                vehicles=tracked_vehicles,
                accident_zones=active_accidents,
                fps=proc_fps,
                frame_idx=frame_idx,
            )

            # Write frame to video
            if writer:
                writer.write(annotated_frame)

            # Live preview if requested
            if args.show:
                cv2.imshow("Accident Detection & Vehicle Tracking System", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested stop via 'q'. Exiting...")
                    break

            frame_idx += 1
            pbar.update(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Cleaning up...")
    finally:
        pbar.close()
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0

    logger.info("=" * 60)
    logger.info("Processing Summary:")
    logger.info(f"Total Frames Processed: {frame_idx}")
    logger.info(f"Total Elapsed Time:      {elapsed:.2f} s")
    logger.info(f"Average Inference Speed: {avg_fps:.2f} FPS")
    logger.info(f"Unique Vehicles Tracked: {len(unique_tracked_ids)}")
    logger.info(f"Active Accident Frames:  {accident_events_total}")
    if writer:
        logger.info(f"Annotated Video Output:  {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
