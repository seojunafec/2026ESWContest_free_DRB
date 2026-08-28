"""Vision utilities for the MEDIFLY-MAP prototype."""

from .detector import CasualtyDetector, DetectionResult
from .geometry import CameraModel, DropPointEstimate
from .vertiport_aruco import (
    MarkerObservation,
    VertiportArucoConfig,
    VertiportGuidance,
    approximate_camera_matrix,
    compute_vertiport_guidance,
    detect_aruco_markers,
    draw_vertiport_overlay,
)
from .usb_camera import (
    describe_capture,
    open_usb_camera,
    set_auto_exposure,
    set_manual_exposure,
)


__all__ = [
    "CameraModel",
    "CasualtyDetector",
    "DetectionResult",
    "DropPointEstimate",
    "MarkerObservation",
    "VertiportArucoConfig",
    "VertiportGuidance",
    "approximate_camera_matrix",
    "compute_vertiport_guidance",
    "detect_aruco_markers",
    "draw_vertiport_overlay",
    "describe_capture",
    "open_usb_camera",
    "set_auto_exposure",
    "set_manual_exposure",
]
