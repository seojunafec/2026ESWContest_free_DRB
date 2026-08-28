from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan
from typing import Literal

DropDirection = Literal["image_up", "image_down", "image_left", "image_right"]


@dataclass(frozen=True)
class CameraModel:
    width_px: int
    height_px: int
    horizontal_fov_deg: float = 70.0
    vertical_fov_deg: float = 43.0

    @property
    def center_x(self) -> float:
        return (self.width_px - 1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.height_px - 1) / 2.0

    @property
    def half_width_ground_ratio(self) -> float:
        return tan(radians(self.horizontal_fov_deg) / 2.0)

    @property
    def half_height_ground_ratio(self) -> float:
        return tan(radians(self.vertical_fov_deg) / 2.0)


@dataclass(frozen=True)
class DropPointEstimate:
    person_center_px: tuple[float, float]
    person_ground_m: tuple[float, float]
    drop_point_px: tuple[float, float]
    drop_ground_m: tuple[float, float]
    distance_from_person_m: float


def pixel_to_ground_m(
    point_px: tuple[float, float],
    camera: CameraModel,
    altitude_m: float,
) -> tuple[float, float]:
    """Estimate ground-plane offset from image center using altitude and FOV."""
    _require_positive_altitude(altitude_m)

    px, py = point_px
    x_norm = (px - camera.center_x) / max(camera.center_x, 1.0)
    y_norm = (py - camera.center_y) / max(camera.center_y, 1.0)

    x_m = x_norm * altitude_m * camera.half_width_ground_ratio
    y_m = y_norm * altitude_m * camera.half_height_ground_ratio
    return x_m, y_m


def ground_delta_to_pixel_delta(
    delta_m: tuple[float, float],
    camera: CameraModel,
    altitude_m: float,
) -> tuple[float, float]:
    _require_positive_altitude(altitude_m)

    dx_m, dy_m = delta_m
    dx_px = dx_m / (altitude_m * camera.half_width_ground_ratio) * max(
        camera.center_x, 1.0
    )
    dy_px = dy_m / (altitude_m * camera.half_height_ground_ratio) * max(
        camera.center_y, 1.0
    )
    return dx_px, dy_px


def estimate_drop_point(
    person_center_px: tuple[float, float],
    camera: CameraModel,
    altitude_m: float,
    distance_m: float = 1.5,
    direction: DropDirection = "image_down",
) -> DropPointEstimate:
    """Place the supply drop point a fixed ground distance from the person."""
    _require_positive_altitude(altitude_m)
    if distance_m <= 0:
        raise ValueError("distance_m must be greater than 0")

    delta_ground_m = _direction_to_ground_delta(distance_m, direction)
    delta_px = ground_delta_to_pixel_delta(delta_ground_m, camera, altitude_m)

    drop_px = (
        person_center_px[0] + delta_px[0],
        person_center_px[1] + delta_px[1],
    )
    return DropPointEstimate(
        person_center_px=person_center_px,
        person_ground_m=pixel_to_ground_m(person_center_px, camera, altitude_m),
        drop_point_px=drop_px,
        drop_ground_m=pixel_to_ground_m(drop_px, camera, altitude_m),
        distance_from_person_m=distance_m,
    )


def _direction_to_ground_delta(
    distance_m: float,
    direction: DropDirection,
) -> tuple[float, float]:
    if direction == "image_up":
        return 0.0, -distance_m
    if direction == "image_down":
        return 0.0, distance_m
    if direction == "image_left":
        return -distance_m, 0.0
    if direction == "image_right":
        return distance_m, 0.0
    raise ValueError(f"unsupported drop direction: {direction}")


def _require_positive_altitude(altitude_m: float) -> None:
    if altitude_m <= 0:
        raise ValueError("altitude_m must be greater than 0")
