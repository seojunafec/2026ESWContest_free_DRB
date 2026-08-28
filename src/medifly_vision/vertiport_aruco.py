from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
Point2D = tuple[float, float]


@dataclass(frozen=True)
class MarkerObservation:
    """한 프레임에서 감지된 ArUco 마커 1개의 정보를 담는 구조체."""

    # ArUco 마커 ID. 예를 들어 중앙 마커는 0, 앞방향 마커는 1로 사용
    marker_id: int

    # 마커 네 꼭짓점의 픽셀 좌표
    # OpenCV가 마커를 감지하면 사각형 네 귀퉁이를 탐지
    corners_px: np.ndarray

    # 네 꼭짓점의 평균으로 구한 마커 중심 픽셀 좌표
    center_px: Point2D

    # 마커의 회전 자세를 나타내는 벡터
    rvec: Optional[np.ndarray] = None
    tvec_m: Optional[np.ndarray] = None


@dataclass(frozen=True)
class VertiportArucoConfig:
    """버티포트 착륙 정렬에 필요한 설정값 모음."""
    center_marker_id: int = 0
    front_marker_id: int = 1
    marker_size_m: float = 0.15
    
    # 사용할 ArUco DICT_4X4_50.
    aruco_dict_name: str = "DICT_4X4_50"
    desired_front: str = "image_up"

    # 중앙 마커가 화면 중앙에서 얼마나 벗어나도 착륙 가능으로 볼지 정하는 허용치. 정규화 값이라 0.08은 화면 기준 약 8% 오차 허용
    center_tolerance_norm: float = 0.08
    yaw_tolerance_deg: float = 8.0
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None


@dataclass(frozen=True)
class VertiportGuidance:
    """한 프레임에서 계산된 착륙 정렬 상태."""
    visible_ids: tuple[int, ...]
    center_marker_id: int
    front_marker_id: int
    has_center: bool
    has_front: bool
    center_px: Optional[Point2D]
    front_px: Optional[Point2D]
    center_error_px: Optional[Point2D]
    center_error_norm: Optional[Point2D]
    front_vector_px: Optional[Point2D]
    yaw_error_deg: Optional[float]
    center_distance_m: Optional[float]
    ready_to_land: bool
    status: str


def approximate_camera_matrix(
    width_px: int,
    height_px: int,
    horizontal_fov_deg: float = 70.0,
    vertical_fov_deg: Optional[float] = None,
) -> np.ndarray:
    """정확한 카메라 캘리브레이션이 없을 때 대략적인 카메라 행렬 생성"""
    fx = (width_px / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)

    # 수직 화각을 모르면 fy도 fx와 같다고 가정
    if vertical_fov_deg is None:
        fy = fx
    else:
        fy = (height_px / 2.0) / math.tan(math.radians(vertical_fov_deg) / 2.0)
        
    return np.array(
        [
            [fx, 0.0, width_px / 2.0],
            [0.0, fy, height_px / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

_DETECTOR_CACHE: dict = {}


def _get_cached_detector(cv2, dictionary_name: str):
    cached = _DETECTOR_CACHE.get(dictionary_name)
    if cached is not None:
        return cached
    dictionary = _get_aruco_dictionary(cv2, dictionary_name)
    parameters = _make_detector_parameters(cv2)
    detector = None
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    cached = (dictionary, parameters, detector)
    _DETECTOR_CACHE[dictionary_name] = cached
    return cached



def detect_aruco_markers(
    frame_bgr: np.ndarray,
    *,
    aruco_dict_name: str = "DICT_4X4_50",
    marker_size_m: Optional[float] = None,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> list[MarkerObservation]:
    """카메라 프레임 1장에서 ArUco 마커들을 찾아 MarkerObservation 목록으로 반환"""
    cv2 = _import_cv2()
    aruco = cv2.aruco



    dictionary, parameters, detector = _get_cached_detector(cv2, aruco_dict_name)
    gray = cv2.cvtColor(np.ascontiguousarray(frame_bgr), cv2.COLOR_BGR2GRAY)
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)



    if ids is None:
        return []

    # pose_by_index는 "몇 번째로 검출된 마커" -> (회전벡터, 위치벡터)를 저장한다.
    pose_by_index: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    if marker_size_m and camera_matrix is not None:
        dist = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1), dtype=np.float64)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners,
            float(marker_size_m),
            camera_matrix,
            dist,
        )
        
        for index in range(len(corners)):
            pose_by_index[index] = (rvecs[index][0], tvecs[index][0])

    observations: list[MarkerObservation] = []


    for index, marker_id in enumerate(ids.flatten()):
        corner = np.asarray(corners[index][0], dtype=np.float64)
        center = corner.mean(axis=0)
        rvec = tvec = None
        if index in pose_by_index:
            rvec, tvec = pose_by_index[index]

        observations.append(
            MarkerObservation(
                marker_id=int(marker_id),
                corners_px=corner,
                center_px=(float(center[0]), float(center[1])),
                rvec=rvec,
                tvec_m=tvec,
            )
        )
    return observations


def compute_vertiport_guidance(
    *,
    frame_width_px: int,
    frame_height_px: int,
    markers: Iterable[MarkerObservation],
    config: VertiportArucoConfig,
) -> VertiportGuidance:
    """감지된 마커 목록으로 현재 착륙 정렬 상태를 계산"""

    # ID로 바로 찾을 수 있게 마커 목록을 딕셔너리로 변경
    markers_by_id = {marker.marker_id: marker for marker in markers}
    center = markers_by_id.get(config.center_marker_id)
    front = markers_by_id.get(config.front_marker_id)

    center_error_px = None
    center_error_norm = None
    center_distance_m = None
    front_vector_px = None
    yaw_error_deg = None
    ready = False

    # 중앙 마커가 보이면 화면 중앙과의 위치 오차 계산
    if center is not None:
        image_center = (frame_width_px / 2.0, frame_height_px / 2.0)

        dx = center.center_px[0] - image_center[0]
        dy = center.center_px[1] - image_center[1]
        center_error_px = (dx, dy)

        center_error_norm = (dx / image_center[0], dy / image_center[1])
        if center.tvec_m is not None:
            center_distance_m = float(np.linalg.norm(center.tvec_m))

    # 중앙 마커와 앞방향 마커가 둘 다 보이면 방향 정렬 오차를 계산
    if center is not None and front is not None:
        front_vector_px = (
            front.center_px[0] - center.center_px[0],
            front.center_px[1] - center.center_px[1],
        )

        yaw_error_deg = signed_angle_deg(
            front_vector_px,
            desired_front_vector(config.desired_front),
        )

    if center is None:
        status = "CENTER_MARKER_MISSING"
    elif front is None:
        status = "FRONT_MARKER_MISSING"
    elif center_error_norm is None or yaw_error_deg is None:
        status = "WAITING"
    else:
        center_ok = (
            abs(center_error_norm[0]) <= config.center_tolerance_norm
            and abs(center_error_norm[1]) <= config.center_tolerance_norm
        )
        yaw_ok = abs(yaw_error_deg) <= config.yaw_tolerance_deg
        # 둘 다 만족해야 착륙 가능 상태로 반환
        ready = center_ok and yaw_ok
        status = "READY_TO_LAND" if ready else "ALIGNING"
        
    
    # 계산한 모든 값을 VertiportGuidance로 묶어서 반환
    return VertiportGuidance(
        visible_ids=tuple(sorted(markers_by_id.keys())),
        center_marker_id=config.center_marker_id,
        front_marker_id=config.front_marker_id,
        has_center=center is not None,
        has_front=front is not None,
        center_px=center.center_px if center is not None else None,
        front_px=front.center_px if front is not None else None,
        center_error_px=center_error_px,
        center_error_norm=center_error_norm,
        front_vector_px=front_vector_px,
        yaw_error_deg=yaw_error_deg,
        center_distance_m=center_distance_m,
        ready_to_land=ready,
        status=status,
    )


def desired_front_vector(direction: str) -> Point2D:
    """문자열로 받은 목표 앞방향을 화면 좌표계의 단위 벡터로 변경"""
    mapping = {
        "image_up": (0.0, -1.0),
        "image_down": (0.0, 1.0),
        "image_left": (-1.0, 0.0),
        "image_right": (1.0, 0.0),
    }
    try:
        return mapping[direction]
    except KeyError as exc:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown desired front direction: {direction}. Use one of: {valid}") from exc


def signed_angle_deg(from_vector: Point2D, to_vector: Point2D) -> float:
    """from_vector를 to_vector에 맞추려면 몇 도 회전해야 하는지 계산"""

    from_x, from_y = _normalize_vector(from_vector)
    to_x, to_y = _normalize_vector(to_vector)
    cross = from_x * to_y - from_y * to_x
    dot = from_x * to_x + from_y * to_y
    return math.degrees(math.atan2(cross, dot))



def draw_vertiport_overlay(
    frame_bgr: np.ndarray,
    markers: Iterable[MarkerObservation],
    guidance: VertiportGuidance,
    *,
    desired_front: str = "image_up",
) -> None:
    """웹캠 화면 위에 마커, 중심점, 방향선, 상태 텍스트를 그린다"""

    cv2 = _import_cv2()
    markers = list(markers)
    height, width = frame_bgr.shape[:2]
    image_center = (int(width / 2), int(height / 2))
    
    # 화면 중앙에 흰색 십자 표시
    cv2.drawMarker(
        frame_bgr,
        image_center,
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )

    # 감지된 모든 마커의 사각형, 중심점, ID를 화면에 표시
    for marker in markers:
        corners = marker.corners_px.astype(np.int32)
        color = (0, 220, 80)
        if marker.marker_id == guidance.center_marker_id:
            color = (255, 180, 0)
        elif marker.marker_id == guidance.front_marker_id:
            color = (0, 120, 255)
        cv2.polylines(frame_bgr, [corners], isClosed=True, color=color, thickness=2)
        center = _as_int_point(marker.center_px)
        cv2.circle(frame_bgr, center, 5, color, -1)
        cv2.putText(
            frame_bgr,
            f"ID {marker.marker_id}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # 중앙 마커와 앞방향 마커가 둘 다 있으면 두 마커를 선으로 연결 - 이 선이 드론의 앞 방향
    if guidance.center_px and guidance.front_px:
        cv2.line(
            frame_bgr,
            _as_int_point(guidance.center_px),
            _as_int_point(guidance.front_px),
            (0, 120, 255),
            3,
        )
    desired = desired_front_vector(desired_front)
    desired_end = (
        int(image_center[0] + desired[0] * 70),
        int(image_center[1] + desired[1] * 70),
    )
    cv2.arrowedLine(frame_bgr, image_center, desired_end, (220, 220, 220), 2, tipLength=0.25)

    # 착륙 가능하면 초록색, 정렬 중이면 노란색, 마커가 부족하면 빨간색 계열로 상태 표시
    status_color = (0, 220, 80) if guidance.ready_to_land else (0, 180, 255)
    if not guidance.has_center or not guidance.has_front:
        status_color = (0, 80, 255)
    lines = [
        f"status: {guidance.status}",
        f"visible ids: {list(guidance.visible_ids)}",
    ]

    # 중앙 오차가 계산됐으면 x/y 오차를 표시
    if guidance.center_error_norm is not None:
        lines.append(
            "center err: "
            f"x={guidance.center_error_norm[0]:+.3f}, "
            f"y={guidance.center_error_norm[1]:+.3f}"
        )

    # 회전 오차가 계산됐으면 yaw 오차를 표시
    if guidance.yaw_error_deg is not None:
        lines.append(f"yaw err: {guidance.yaw_error_deg:+.1f} deg")

    # 거리 추정값이 있으면 중앙 마커까지의 거리를 표시
    if guidance.center_distance_m is not None:
        lines.append(f"center distance: {guidance.center_distance_m:.2f} m")
        
        
    for index, line in enumerate(lines):
        y = 28 + index * 28
        cv2.putText(
            frame_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color if index == 0 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _get_aruco_dictionary(cv2, dictionary_name: str):
    """문자열 이름으로 OpenCV ArUco 딕셔너리를 호출"""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is missing. Install opencv-contrib-python or a Jetson OpenCV build with ArUco.")

    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def _make_detector_parameters(cv2):
    """ArUco 검출 파라미터를 OpenCV 버전에 맞게 생성"""
    
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()

    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1

    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    return parameters


def _normalize_vector(vector: Point2D) -> Point2D:
    """2D 벡터를 길이 1짜리 단위 벡터로 변경"""

    length = math.hypot(vector[0], vector[1])
    if length <= 1e-9:
        raise ValueError("Cannot normalize a zero-length vector.")
    return (vector[0] / length, vector[1] / length)


def _as_int_point(point: Point2D) -> tuple[int, int]:
    """OpenCV 그리기 함수에 넣을 수 있도록 float 좌표를 int 좌표로 변경"""

    return (int(round(point[0])), int(round(point[1])))


def _import_cv2():

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for ArUco detection. Install opencv-contrib-python on PC, "
            "or python3-opencv with ArUco support on Jetson."
        ) from exc
    return cv2
