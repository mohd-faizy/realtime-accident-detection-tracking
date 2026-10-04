import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from src.accident_logic import AccidentZone
from src.tracker import TrackedVehicle


class Visualizer:
    """
    Renders high-quality computer vision overlays:
    - Vehicle bounding boxes with BoT-SORT tracking IDs
    - Highlighted accident regions with containment logic
    - Real-time emergency alert banner
    - System HUD statistics (FPS, vehicle counts, accident status)
    """

    def __init__(
        self,
        show_tracking_ids: bool = True,
        show_confidence: bool = True,
        show_alert_banner: bool = True,
        show_hud_stats: bool = True,
        suppress_vehicle_labels: bool = True,
        line_thickness: int = 2,
    ):
        self.show_tracking_ids = show_tracking_ids
        self.show_confidence = show_confidence
        self.show_alert_banner = show_alert_banner
        self.show_hud_stats = show_hud_stats
        self.suppress_vehicle_labels = suppress_vehicle_labels
        self.line_thickness = line_thickness

        # Colors in BGR format
        self.color_vehicle = (230, 160, 50)      # Sleek cyan/blue-orange
        self.color_accident = (30, 30, 235)      # Intense warning red
        self.color_maintained = (0, 140, 255)    # Amber orange for maintained/held
        self.color_banner = (25, 25, 220)        # Bold alert banner red
        self.color_hud_bg = (20, 20, 20)         # Dark glass HUD background
        self.color_white = (255, 255, 255)
        self.color_green = (50, 220, 50)

    def draw_rounded_rect(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        radius: int = 8,
    ):
        """Draws a neat bounding box with rounded corners."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

        # Draw box corners
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    def draw_label(
        self,
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
        scale: float = 0.55,
        thickness: int = 1,
    ):
        """Draws a label pill with solid background behind text for readability."""
        x, y = pos
        (w, h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        pad = 4
        y_top = max(0, y - h - 2 * pad)
        y_bot = max(0, y)

        cv2.rectangle(img, (x, y_top), (x + w + 2 * pad, y_bot), bg_color, -1)
        cv2.putText(
            img,
            text,
            (x + pad, y_bot - pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_vehicles(self, img: np.ndarray, vehicles: List[TrackedVehicle]):
        """Draws tracked vehicle bounding boxes and labels."""
        for v in vehicles:
            # If vehicle is inside an accident zone and suppression is enabled, skip normal label
            if v.is_suppressed and self.suppress_vehicle_labels:
                continue

            x1, y1, x2, y2 = map(int, v.bbox)
            self.draw_rounded_rect(
                img, (x1, y1), (x2, y2), self.color_vehicle, self.line_thickness
            )

            # Build label
            label_parts = []
            if self.show_tracking_ids:
                label_parts.append(f"#{v.track_id}")
            label_parts.append(v.class_name)
            if self.show_confidence:
                label_parts.append(f"{v.conf:.2f}")

            label_text = " ".join(label_parts)
            self.draw_label(img, label_text, (x1, y1), self.color_vehicle)

    def draw_accidents(self, img: np.ndarray, accident_zones: List[AccidentZone]):
        """Draws confirmed and maintained accident bounding boxes."""
        for zone in accident_zones:
            x1, y1, x2, y2 = map(int, zone.bbox)
            color = self.color_accident if zone.status == "ACTIVE" else self.color_maintained
            thickness = self.line_thickness + 2

            # Outer bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw corner accent marks for high visibility
            corner_len = min(25, (x2 - x1) // 3, (y2 - y1) // 3)
            acc_color = (0, 0, 255)
            # Top-left
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), acc_color, thickness + 2)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), acc_color, thickness + 2)
            # Top-right
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), acc_color, thickness + 2)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), acc_color, thickness + 2)
            # Bottom-left
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), acc_color, thickness + 2)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), acc_color, thickness + 2)
            # Bottom-right
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), acc_color, thickness + 2)
            cv2.line(img, (x2, y2), (x2, y2 - corner_len), acc_color, thickness + 2)

            # Label text
            status_tag = "ACTIVE" if zone.status == "ACTIVE" else f"HELD (lost {zone.lost_frames}f)"
            label = f"[!] ACCIDENT #{zone.zone_id} ({status_tag}) {zone.conf:.2f}"
            if zone.involved_vehicle_ids:
                label += f" | IDs: {zone.involved_vehicle_ids}"

            self.draw_label(
                img,
                label,
                (x1, y1),
                bg_color=color,
                text_color=(255, 255, 255),
                scale=0.6,
                thickness=2,
            )

    def draw_alert_banner(self, img: np.ndarray, accident_count: int, frame_idx: int):
        """Draws top alert banner when accidents are active."""
        if not self.show_alert_banner or accident_count == 0:
            return

        h, w = img.shape[:2]
        banner_height = 45

        # Create overlay for smooth blending
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), self.color_banner, -1)
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Pulsing text indicators
        pulse = ">>>" if (frame_idx // 15) % 2 == 0 else "!!!"
        alert_text = (
            f"{pulse} CRITICAL ALERT: ROAD ACCIDENT DETECTED! "
            f"Active Zones: {accident_count} {pulse}"
        )

        (text_w, text_h), _ = cv2.getTextSize(
            alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2
        )
        text_x = max(20, (w - text_w) // 2)
        text_y = (banner_height + text_h) // 2

        cv2.putText(
            img,
            alert_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            self.color_white,
            2,
            cv2.LINE_AA,
        )

    def draw_hud(
        self,
        img: np.ndarray,
        fps: float,
        vehicle_count: int,
        accident_count: int,
    ):
        """Draws HUD status panel in top-left."""
        if not self.show_hud_stats:
            return

        # Position below banner if banner is active
        start_y = 55 if accident_count > 0 and self.show_alert_banner else 15
        panel_w = 260
        panel_h = 100
        x1, y1 = 15, start_y
        x2, y2 = x1 + panel_w, y1 + panel_h

        # Semi-transparent dark background
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_hud_bg, -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (70, 70, 70), 1)

        # Title
        cv2.putText(
            img,
            "TRAFFIC MONITORING SYSTEM",
            (x1 + 10, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # FPS
        cv2.putText(
            img,
            f"FPS: {fps:.1f}",
            (x1 + 10, y1 + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.color_white,
            1,
            cv2.LINE_AA,
        )

        # Vehicles Count
        cv2.putText(
            img,
            f"Tracked Vehicles: {vehicle_count}",
            (x1 + 10, y1 + 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.color_white,
            1,
            cv2.LINE_AA,
        )

        # Status
        status_text = "ACCIDENT DETECTED" if accident_count > 0 else "NORMAL"
        status_color = self.color_accident if accident_count > 0 else self.color_green
        cv2.putText(
            img,
            f"Status: {status_text}",
            (x1 + 10, y1 + 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            2 if accident_count > 0 else 1,
            cv2.LINE_AA,
        )

    def render(
        self,
        frame: np.ndarray,
        vehicles: List[TrackedVehicle],
        accident_zones: List[AccidentZone],
        fps: float,
        frame_idx: int,
    ) -> np.ndarray:
        """
        Renders complete frame annotation with vehicles, accidents, alert banner, and HUD.
        """
        annotated = frame.copy()

        # 1. Render tracked vehicles (suppressed ones inside accidents are filtered out)
        self.draw_vehicles(annotated, vehicles)

        # 2. Render accident zones
        self.draw_accidents(annotated, accident_zones)

        # 3. Render top emergency alert banner
        self.draw_alert_banner(annotated, len(accident_zones), frame_idx)

        # 4. Render HUD dashboard
        self.draw_hud(annotated, fps, len(vehicles), len(accident_zones))

        return annotated
