from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import CameraModel, DropPointEstimate, estimate_drop_point


@dataclass(frozen=True)
class DetectionResult:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_px: tuple[float, float]
    ground_m: tuple[float, float]
    drop: DropPointEstimate
    inference_source: str = "full"


class CasualtyDetector:
    """YOLO wrapper focused on rescue-person detection."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        target_classes: Iterable[str] = (
            "person",
            "casualty",
            "person_down",
            "fallen_person",
            "lying_person",
            "rescued_person",
        ),
        image_size: int = 640,
        device: str | None = None,
        adaptive_tiling: bool = False,
        tile_overlap: float = 0.15,
        tile_activation_confidence: float | None = None,
        roi_margin: float = 0.0,
        tile_target_distance_ratio: float = 0.15,
        global_search_frames: int = 6,
        tile_refresh_frames: int = 15,
        tile_lost_frames: int = 3,
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in the range (0, 1]")
        if image_size <= 0:
            raise ValueError("image_size must be greater than 0")
        if not 0.0 <= tile_overlap < 0.5:
            raise ValueError("tile_overlap must be in the range [0, 0.5)")
        if tile_activation_confidence is not None and not (
            confidence <= tile_activation_confidence <= 1.0
        ):
            raise ValueError(
                "tile_activation_confidence must be between confidence and 1"
            )
        if not 0.0 <= roi_margin < 0.5:
            raise ValueError("roi_margin must be in the range [0, 0.5)")
        if not 0.0 < tile_target_distance_ratio < 1.0:
            raise ValueError(
                "tile_target_distance_ratio must be in the range (0, 1)"
            )
        if global_search_frames <= 0:
            raise ValueError("global_search_frames must be greater than 0")
        if tile_refresh_frames <= 0:
            raise ValueError("tile_refresh_frames must be greater than 0")
        if tile_lost_frames <= 0:
            raise ValueError("tile_lost_frames must be greater than 0")

        self.model_path = model_path
        self.confidence = confidence
        self.target_classes = {name.lower() for name in target_classes}
        self.image_size = image_size
        self.device = device
        self.adaptive_tiling = adaptive_tiling
        self.tile_overlap = tile_overlap
        self.tile_activation_confidence = (
            confidence
            if tile_activation_confidence is None
            else tile_activation_confidence
        )
        self.roi_margin = roi_margin
        self.tile_target_distance_ratio = tile_target_distance_ratio
        self.global_search_frames = global_search_frames
        self.tile_refresh_frames = tile_refresh_frames
        self.tile_lost_frames = tile_lost_frames
        self.model = self._load_model(model_path)

        self.last_inference_mode = "full"
        self.last_inference_ms = 0.0
        self._frame_shape: tuple[int, int] | None = None
        self._search_mode = "global"
        self._global_misses = 0
        self._tile_scan_index = 0
        self._active_tile: tuple[int, int, int, int] | None = None
        self._active_target_center: tuple[float, float] | None = None
        self._tile_misses = 0
        self._tile_follow_frames = 0

    def detect(
        self,
        frame_bgr: np.ndarray,
        altitude_m: float,
        camera: CameraModel | None = None,
        drop_distance_m: float = 1.5,
        drop_direction: str = "image_down",
    ) -> list[DetectionResult]:
        if camera is None:
            height, width = frame_bgr.shape[:2]
            camera = CameraModel(width_px=width, height_px=height)

        return self._detect_region(
            frame_bgr=frame_bgr,
            bounds=(0, 0, frame_bgr.shape[1], frame_bgr.shape[0]),
            altitude_m=altitude_m,
            camera=camera,
            drop_distance_m=drop_distance_m,
            drop_direction=drop_direction,
            inference_source="full",
        )

    def detect_adaptive(
        self,
        frame_bgr: np.ndarray,
        altitude_m: float,
        camera: CameraModel | None = None,
        drop_distance_m: float = 1.5,
        drop_direction: str = "image_down",
    ) -> list[DetectionResult]:
        """Run one YOLO inference per frame while periodically scanning 2x2 tiles."""
        height, width = frame_bgr.shape[:2]
        if camera is None:
            camera = CameraModel(width_px=width, height_px=height)

        if not self.adaptive_tiling:
            return self.detect(
                frame_bgr=frame_bgr,
                altitude_m=altitude_m,
                camera=camera,
                drop_distance_m=drop_distance_m,
                drop_direction=drop_direction,
            )

        frame_shape = (height, width)
        if self._frame_shape != frame_shape:
            self._reset_adaptive_state(frame_shape)

        if self._search_mode == "tile_scan":
            return self._scan_next_tile(
                frame_bgr,
                altitude_m,
                camera,
                drop_distance_m,
                drop_direction,
            )

        if self._search_mode == "tile_follow":
            return self._follow_active_tile(
                frame_bgr,
                altitude_m,
                camera,
                drop_distance_m,
                drop_direction,
            )

        detections = self.detect(
            frame_bgr=frame_bgr,
            altitude_m=altitude_m,
            camera=camera,
            drop_distance_m=drop_distance_m,
            drop_direction=drop_direction,
        )
        self.last_inference_mode = "global"
        if self._has_activation_detection(detections):
            self._global_misses = 0
        else:
            self._global_misses += 1
            if self._global_misses >= self.global_search_frames:
                self._search_mode = "tile_scan"
                self._tile_scan_index = 0
        return detections

    def detect_tiled(
        self,
        frame_bgr: np.ndarray,
        altitude_m: float,
        camera: CameraModel | None = None,
        drop_distance_m: float = 1.5,
        drop_direction: str = "image_down",
        include_full_frame: bool = True,
        iou_threshold: float = 0.45,
    ) -> list[DetectionResult]:
        """Run a complete 2x2 tiled scan, intended for still-image validation."""
        if not 0.0 < iou_threshold < 1.0:
            raise ValueError("iou_threshold must be in the range (0, 1)")

        height, width = frame_bgr.shape[:2]
        if camera is None:
            camera = CameraModel(width_px=width, height_px=height)

        detections: list[DetectionResult] = []
        elapsed_ms = 0.0
        if include_full_frame:
            detections.extend(
                self.detect(
                    frame_bgr=frame_bgr,
                    altitude_m=altitude_m,
                    camera=camera,
                    drop_distance_m=drop_distance_m,
                    drop_direction=drop_direction,
                )
            )
            elapsed_ms += self.last_inference_ms

        for index, bounds in enumerate(self._tile_bounds(width, height)):
            detections.extend(
                self._detect_region(
                    frame_bgr=frame_bgr,
                    bounds=bounds,
                    altitude_m=altitude_m,
                    camera=camera,
                    drop_distance_m=drop_distance_m,
                    drop_direction=drop_direction,
                    inference_source=f"tile_{index + 1}",
                )
            )
            elapsed_ms += self.last_inference_ms

        self.last_inference_mode = "full_2x2_tiles"
        self.last_inference_ms = elapsed_ms
        return self._non_max_suppression(detections, iou_threshold)

    def _scan_next_tile(
        self,
        frame_bgr: np.ndarray,
        altitude_m: float,
        camera: CameraModel,
        drop_distance_m: float,
        drop_direction: str,
    ) -> list[DetectionResult]:
        height, width = frame_bgr.shape[:2]
        bounds_list = self._tile_bounds(width, height)
        bounds = bounds_list[self._tile_scan_index]
        tile_number = self._tile_scan_index + 1
        detections = self._detect_region(
            frame_bgr=frame_bgr,
            bounds=bounds,
            altitude_m=altitude_m,
            camera=camera,
            drop_distance_m=drop_distance_m,
            drop_direction=drop_direction,
            inference_source=f"tile_scan_{tile_number}",
        )
        self.last_inference_mode = f"tile_scan_{tile_number}/4"

        activation = next(
            (
                detection
                for detection in detections
                if detection.confidence >= self.tile_activation_confidence
            ),
            None,
        )
        if activation is not None:
            tile_width = bounds[2] - bounds[0]
            tile_height = bounds[3] - bounds[1]
            self._active_tile = self._centered_bounds(
                activation.center_px,
                tile_width,
                tile_height,
                width,
                height,
            )
            self._active_target_center = activation.center_px
            self._search_mode = "tile_follow"
            self._tile_misses = 0
            self._tile_follow_frames = 0
        else:
            self._tile_scan_index += 1
            if self._tile_scan_index >= len(bounds_list):
                self._search_mode = "global"
                self._global_misses = 0
                self._tile_scan_index = 0
        return detections

    def _follow_active_tile(
        self,
        frame_bgr: np.ndarray,
        altitude_m: float,
        camera: CameraModel,
        drop_distance_m: float,
        drop_direction: str,
    ) -> list[DetectionResult]:
        height, width = frame_bgr.shape[:2]
        if self._active_tile is None:
            self._search_mode = "global"
            self._active_target_center = None
            return []

        if self._tile_follow_frames >= self.tile_refresh_frames:
            detections = self.detect(
                frame_bgr=frame_bgr,
                altitude_m=altitude_m,
                camera=camera,
                drop_distance_m=drop_distance_m,
                drop_direction=drop_direction,
            )
            self.last_inference_mode = "global_refresh"
            self._tile_follow_frames = 0
            activation = self._select_active_detection(
                detections,
                width,
                height,
                minimum_confidence=self.tile_activation_confidence,
            )
            if activation is not None:
                self._search_mode = "global"
                self._global_misses = 0
                self._active_tile = None
                self._active_target_center = None
            return detections

        bounds = self._active_tile
        detections = self._detect_region(
            frame_bgr=frame_bgr,
            bounds=bounds,
            altitude_m=altitude_m,
            camera=camera,
            drop_distance_m=drop_distance_m,
            drop_direction=drop_direction,
            inference_source="tile_follow",
        )
        self.last_inference_mode = "tile_follow"
        self._tile_follow_frames += 1

        active_detection = self._select_active_detection(
            detections,
            width,
            height,
            minimum_confidence=self.confidence,
        )
        if active_detection is not None:
            tile_width = bounds[2] - bounds[0]
            tile_height = bounds[3] - bounds[1]
            self._active_tile = self._centered_bounds(
                active_detection.center_px,
                tile_width,
                tile_height,
                width,
                height,
            )
            self._active_target_center = active_detection.center_px
            self._tile_misses = 0
        else:
            self._tile_misses += 1
            if self._tile_misses >= self.tile_lost_frames:
                self._search_mode = "tile_scan"
                self._tile_scan_index = 0
                self._active_tile = None
                self._active_target_center = None
                self._tile_misses = 0
        return detections

    def _detect_region(
        self,
        frame_bgr: np.ndarray,
        bounds: tuple[int, int, int, int],
        altitude_m: float,
        camera: CameraModel,
        drop_distance_m: float,
        drop_direction: str,
        inference_source: str,
    ) -> list[DetectionResult]:
        x_offset, y_offset, x2_bound, y2_bound = bounds
        is_full_frame = (
            x_offset == 0
            and y_offset == 0
            and x2_bound == frame_bgr.shape[1]
            and y2_bound == frame_bgr.shape[0]
        )
        region = frame_bgr if is_full_frame else frame_bgr[
            y_offset:y2_bound, x_offset:x2_bound
        ]
        if not region.flags.c_contiguous:
            region = np.ascontiguousarray(region)
        if region.size == 0:
            return []

        started = time.perf_counter()
        predictions = self.model.predict(
            region,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        detections: list[DetectionResult] = []
        for result in predictions:
            names = result.names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item())
                if isinstance(names, dict):
                    class_name = str(names.get(class_id, class_id)).lower()
                else:
                    class_name = str(names[class_id]).lower()
                if self.target_classes and class_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                drop = estimate_drop_point(
                    person_center_px=center,
                    camera=camera,
                    altitude_m=altitude_m,
                    distance_m=drop_distance_m,
                    direction=drop_direction,  # type: ignore[arg-type]
                )
                detections.append(
                    DetectionResult(
                        class_name=class_name,
                        confidence=float(box.conf[0].item()),
                        bbox_xyxy=(x1, y1, x2, y2),
                        center_px=center,
                        ground_m=drop.person_ground_m,
                        drop=drop,
                        inference_source=inference_source,
                    )
                )

        detections = self._filter_global_roi(
            detections,
            frame_bgr.shape[1],
            frame_bgr.shape[0],
        )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def _has_activation_detection(
        self, detections: list[DetectionResult]
    ) -> bool:
        return any(
            detection.confidence >= self.tile_activation_confidence
            for detection in detections
        )

    def _select_active_detection(
        self,
        detections: list[DetectionResult],
        frame_width: int,
        frame_height: int,
        minimum_confidence: float,
    ) -> DetectionResult | None:
        eligible = [
            detection
            for detection in detections
            if detection.confidence >= minimum_confidence
        ]
        if not eligible:
            return None
        if self._active_target_center is None:
            return eligible[0]

        diagonal = max(float(np.hypot(frame_width, frame_height)), 1.0)
        nearest = min(
            eligible,
            key=lambda detection: np.hypot(
                detection.center_px[0] - self._active_target_center[0],
                detection.center_px[1] - self._active_target_center[1],
            ),
        )
        distance_ratio = float(
            np.hypot(
                nearest.center_px[0] - self._active_target_center[0],
                nearest.center_px[1] - self._active_target_center[1],
            )
            / diagonal
        )
        return (
            nearest
            if distance_ratio <= self.tile_target_distance_ratio
            else None
        )

    def _filter_global_roi(
        self,
        detections: list[DetectionResult],
        frame_width: int,
        frame_height: int,
    ) -> list[DetectionResult]:
        if self.roi_margin <= 0.0:
            return detections
        left = frame_width * self.roi_margin
        right = frame_width * (1.0 - self.roi_margin)
        top = frame_height * self.roi_margin
        bottom = frame_height * (1.0 - self.roi_margin)
        return [
            detection
            for detection in detections
            if left <= detection.center_px[0] <= right
            and top <= detection.center_px[1] <= bottom
        ]

    def _reset_adaptive_state(self, frame_shape: tuple[int, int]) -> None:
        self._frame_shape = frame_shape
        self._search_mode = "global"
        self._global_misses = 0
        self._tile_scan_index = 0
        self._active_tile = None
        self._active_target_center = None
        self._tile_misses = 0
        self._tile_follow_frames = 0

    def _tile_bounds(
        self, width: int, height: int
    ) -> list[tuple[int, int, int, int]]:
        tile_width = min(
            width,
            int(np.ceil(width / (2.0 - self.tile_overlap))),
        )
        tile_height = min(
            height,
            int(np.ceil(height / (2.0 - self.tile_overlap))),
        )
        x_positions = (0, width - tile_width)
        y_positions = (0, height - tile_height)
        return [
            (x, y, x + tile_width, y + tile_height)
            for y in y_positions
            for x in x_positions
        ]

    @staticmethod
    def _centered_bounds(
        center: tuple[float, float],
        tile_width: int,
        tile_height: int,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        x1 = int(round(center[0] - tile_width / 2.0))
        y1 = int(round(center[1] - tile_height / 2.0))
        x1 = min(max(x1, 0), max(frame_width - tile_width, 0))
        y1 = min(max(y1, 0), max(frame_height - tile_height, 0))
        return x1, y1, x1 + tile_width, y1 + tile_height

    @classmethod
    def _non_max_suppression(
        cls,
        detections: list[DetectionResult],
        iou_threshold: float,
    ) -> list[DetectionResult]:
        kept: list[DetectionResult] = []
        for candidate in sorted(
            detections, key=lambda item: item.confidence, reverse=True
        ):
            duplicate = any(
                candidate.class_name == existing.class_name
                and (
                    cls._box_iou(candidate.bbox_xyxy, existing.bbox_xyxy)
                    >= iou_threshold
                    or cls._intersection_over_smaller(
                        candidate.bbox_xyxy, existing.bbox_xyxy
                    )
                    >= 0.75
                    and cls._box_center_distance_ratio(
                        candidate.bbox_xyxy, existing.bbox_xyxy
                    )
                    <= 0.40
                )
                for existing in kept
            )
            if not duplicate:
                kept.append(candidate)
        return kept

    @staticmethod
    def _box_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        if intersection <= 0.0:
            return 0.0
        first_area = max(0.0, first[2] - first[0]) * max(
            0.0, first[3] - first[1]
        )
        second_area = max(0.0, second[2] - second[0]) * max(
            0.0, second[3] - second[1]
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    @staticmethod
    def _intersection_over_smaller(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(
            0.0, first[3] - first[1]
        )
        second_area = max(0.0, second[2] - second[0]) * max(
            0.0, second[3] - second[1]
        )
        smaller_area = min(first_area, second_area)
        return intersection / smaller_area if smaller_area > 0.0 else 0.0

    @staticmethod
    def _box_center_distance_ratio(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
        second_center = (
            (second[0] + second[2]) / 2.0,
            (second[1] + second[3]) / 2.0,
        )
        distance = float(
            np.hypot(
                first_center[0] - second_center[0],
                first_center[1] - second_center[1],
            )
        )
        first_diagonal = float(np.hypot(first[2] - first[0], first[3] - first[1]))
        second_diagonal = float(
            np.hypot(second[2] - second[0], second[3] - second[1])
        )
        return distance / max(first_diagonal, second_diagonal, 1.0)

    @staticmethod
    def _load_model(model_path: str):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install requirements.txt first."
            ) from exc

        return YOLO(model_path)
