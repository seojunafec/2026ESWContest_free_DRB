#!/usr/bin/env python3
"""
P5 비주얼 서보잉 검증기

키
    q : 종료
    l : CSV 로깅 on/off
    r : 정렬 타이머 강제 리셋
    x : 좌우 부호 반전
    y : 전후 부호 반전
    t : 요 부호 반전
"""
from __future__ import annotations

import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from medifly_vision import (
    VertiportArucoConfig,
    approximate_camera_matrix,
    compute_vertiport_guidance,
    describe_capture,
    detect_aruco_markers,
    open_usb_camera,
)


# ======================================================================
# 카메라 / 마커 설정
# ======================================================================

CAMERA_DEVICE = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1200

HFOV_DEG = 70.0            
VFOV_DEG = 43.0            
MARKER_SIZE_M = 0.22       

# ======================================================================
# 부호 규약. 실측으로 확정
# ======================================================================

SIGN_LATERAL = +1.0        # 좌우
SIGN_FORWARD = -1.0        # 전후
SIGN_YAW = +1.0            # 요


# ======================================================================
# 제어 파라미터
# ======================================================================

KP_LATERAL = 0.4           # 정규화 오차 -> m/s
KP_YAW = 0.15              # deg -> deg/s
MAX_SPEED = 0.3            # m/s 상한
MAX_YAW_RATE = 15.0        # deg/s 상한
DEADBAND = 0.05            # 중심 오차 데드밴드 (정규화)
YAW_DEADBAND = 4.0         # 요 오차 데드밴드 (deg)

FOV_RATIO = math.tan(math.radians(VFOV_DEG) / 2.0) / math.tan(
    math.radians(HFOV_DEG) / 2.0
)


# ======================================================================
# 착륙 승인 조건
# ======================================================================

LAND_CENTER_TOL = 0.08     # 중심 오차 허용 (정규화)
LAND_YAW_TOL = 8.0         # 요 오차 허용 (deg)
LAND_HOLD_SEC = 3.0        # 이 시간만큼 연속 유지해야 승인


def compute_body_velocity(center_error_norm, yaw_error_deg,
                          sign_lat, sign_fwd, sign_yaw):
    """
    화면 오차 -> 기체 body frame 속도 명령.

    MAVSDK VelocityBodyYawspeed 규약:
        +x = 전방, +y = 우측, +z = 아래, yawspeed = 시계방향(+)
    """
    if center_error_norm is None:
        return 0.0, 0.0, 0.0

    e_x, e_y = center_error_norm


    if abs(e_x) < DEADBAND:
        e_x = 0.0
    if abs(e_y) < DEADBAND:
        e_y = 0.0

    v_y = sign_lat * KP_LATERAL * e_x
    v_x = sign_fwd * KP_LATERAL * e_y * FOV_RATIO

    yaw_rate = 0.0
    if yaw_error_deg is not None:
        yaw_err = yaw_error_deg
        if abs(yaw_err) < YAW_DEADBAND:
            yaw_err = 0.0
        yaw_rate = sign_yaw * KP_YAW * yaw_err

    # 속도 상한
    v_x = max(-MAX_SPEED, min(MAX_SPEED, v_x))
    v_y = max(-MAX_SPEED, min(MAX_SPEED, v_y))
    yaw_rate = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, yaw_rate))
    return v_x, v_y, yaw_rate


def describe(v_x, v_y, yaw_rate) -> str:
    parts = []
    if abs(v_x) < 0.01 and abs(v_y) < 0.01:
        parts.append("HOLD")
    else:
        if v_x > 0.01:
            parts.append("FORWARD")
        elif v_x < -0.01:
            parts.append("BACKWARD")
        if v_y > 0.01:
            parts.append("RIGHT")
        elif v_y < -0.01:
            parts.append("LEFT")
    if yaw_rate > 0.5:
        parts.append("CW")
    elif yaw_rate < -0.5:
        parts.append("CCW")
    return " + ".join(parts)


def is_aligned(guidance) -> bool:
    if not guidance.has_center or not guidance.has_front:
        return False
    err = guidance.center_error_norm
    yaw = guidance.yaw_error_deg
    if err is None or yaw is None:
        return False
    return (
        abs(err[0]) <= LAND_CENTER_TOL
        and abs(err[1]) <= LAND_CENTER_TOL
        and abs(yaw) <= LAND_YAW_TOL
    )


def marker_pixel_size(marker_size_m: float, altitude_m: float,
                      width_px: int = FRAME_WIDTH) -> float:
    ground_width_m = 2.0 * altitude_m * math.tan(math.radians(HFOV_DEG) / 2.0)
    if ground_width_m <= 0:
        return 0.0
    return marker_size_m * (width_px / ground_width_m)


def print_sizing_table(marker_size_m: float) -> None:
    print("=" * 64)
    print(f"마커 {marker_size_m*100:.0f}cm 고도별 화면 크기 "
          f"(HFOV {HFOV_DEG:.0f}deg 가정 - 미측정값이라 참고용)")
    print("-" * 64)
    for alt in (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
        px = marker_pixel_size(marker_size_m, alt)
        if px > FRAME_HEIGHT:
            verdict = "화면 초과"
        elif px >= 60:
            verdict = "양호"
        elif px >= 30:
            verdict = "한계"
        else:
            verdict = "검출 불가"
        print(f"  고도 {alt:5.1f} m -> {px:7.1f} px   {verdict}")
    print("=" * 64)


def draw_arrow(frame, v_x, v_y, origin, scale=400.0):
    """기체 이동 방향을 탑뷰 화살표로 작성"""
    if abs(v_x) < 0.01 and abs(v_y) < 0.01:
        cv2.circle(frame, origin, 30, (0, 220, 80), 4)
        return
    end = (int(origin[0] + v_y * scale), int(origin[1] - v_x * scale))
    cv2.arrowedLine(frame, origin, end, (0, 80, 255), 8, tipLength=0.3)


def open_log():
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"servo_{datetime.now():%Y%m%d_%H%M%S}.csv"
    handle = path.open("w", newline="")
    writer = csv.writer(handle)
    writer.writerow([
        "t", "ids", "status", "e_x", "e_y", "yaw_err_deg",
        "dist_m", "marker_px", "v_x", "v_y", "yaw_rate", "hold_sec",
    ])
    print(f"\n[LOG] 기록 시작: {path}", flush=True)
    return handle, writer, path


def main() -> int:
    print_sizing_table(MARKER_SIZE_M)

    cap = open_usb_camera(device=CAMERA_DEVICE,
                          width=FRAME_WIDTH, height=FRAME_HEIGHT)
    if cap is None:
        print("USB 카메라 열기 실패", file=sys.stderr)
        return 1
    print(f"[카메라] {describe_capture(cap)}\n")

    config = VertiportArucoConfig(
        center_marker_id=0,
        front_marker_id=1,
        marker_size_m=MARKER_SIZE_M,
        desired_front="image_down",   # !! 재검증 필요 !!
        center_tolerance_norm=LAND_CENTER_TOL,
        yaw_tolerance_deg=LAND_YAW_TOL,
    )

    sign_lat, sign_fwd, sign_yaw = SIGN_LATERAL, SIGN_FORWARD, SIGN_YAW
    camera_matrix = None
    aligned_since = None
    hold_sec = 0.0
    land_ready = False
    frame_no = 0
    log_handle = log_writer = log_path = None
    t_start = time.monotonic()
    seen_frames = 0
    total_frames = 0

    cv2.namedWindow("servo direction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("servo direction", 1100, 690)

    print("검증 시작.  q=종료  l=로깅  r=리셋  "
          "x=좌우반전  y=전후반전  t=요반전\n", flush=True)

    try:
        while True:
            ok, raw = cap.read()
            if not ok or raw is None:
                continue
            frame = np.ascontiguousarray(raw.copy(), dtype=np.uint8)
            height, width = frame.shape[:2]

            if camera_matrix is None:
                camera_matrix = approximate_camera_matrix(
                    width, height, HFOV_DEG, VFOV_DEG)

            markers = detect_aruco_markers(
                frame,
                aruco_dict_name=config.aruco_dict_name,
                marker_size_m=config.marker_size_m,
                camera_matrix=camera_matrix,
            )
            guidance = compute_vertiport_guidance(
                frame_width_px=width,
                frame_height_px=height,
                markers=markers,
                config=config,
            )

            v_x, v_y, yaw_rate = compute_body_velocity(
                guidance.center_error_norm, guidance.yaw_error_deg,
                sign_lat, sign_fwd, sign_yaw)
                
            now = time.monotonic()
            if is_aligned(guidance):
                if aligned_since is None:
                    aligned_since = now
                hold_sec = now - aligned_since
            else:
                aligned_since = None  
                hold_sec = 0.0
            land_ready = hold_sec >= LAND_HOLD_SEC

            total_frames += 1
            if guidance.has_center:
                seen_frames += 1
            hit_rate = seen_frames / max(total_frames, 1)

            marker_px = 0.0
            center_marker = next(
                (m for m in markers if m.marker_id == config.center_marker_id),
                None)
            if center_marker is not None:
                c = center_marker.corners_px
                marker_px = float(np.linalg.norm(c[0] - c[1]))


            # ---------------- 화면 표시 ----------------
            for m in markers:
                corner = m.corners_px.astype(np.int32)
                color = ((255, 180, 0) if m.marker_id == config.center_marker_id
                         else (0, 120, 255))
                cv2.polylines(frame, [corner], True, color, 3)
                cx, cy = int(m.center_px[0]), int(m.center_px[1])
                cv2.circle(frame, (cx, cy), 10, color, -1)
                cv2.putText(frame, f"ID{m.marker_id}", (cx + 14, cy - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

            if guidance.center_px and guidance.front_px:
                cv2.line(frame,
                         (int(guidance.center_px[0]), int(guidance.center_px[1])),
                         (int(guidance.front_px[0]), int(guidance.front_px[1])),
                         (0, 120, 255), 3)

            cx0, cy0 = width // 2, height // 2
            cv2.drawMarker(frame, (cx0, cy0), (255, 255, 255),
                           cv2.MARKER_CROSS, 60, 3)
            tol_w = int(LAND_CENTER_TOL * width / 2)
            tol_h = int(LAND_CENTER_TOL * height / 2)
            box_color = (0, 255, 0) if land_ready else (140, 140, 140)
            cv2.rectangle(frame, (cx0 - tol_w, cy0 - tol_h),
                          (cx0 + tol_w, cy0 + tol_h), box_color, 2)


            cv2.putText(frame, "screen TOP = FRONT", (cx0 - 200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
            cv2.putText(frame, "screen BOTTOM = BACK", (cx0 - 220, height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
            cv2.putText(frame, "LEFT", (25, cy0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            cv2.putText(frame, "RIGHT", (width - 130, cy0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

            panel_origin = (width - 220, 220)
            cv2.rectangle(frame, (width - 420, 20), (width - 20, 420),
                          (40, 40, 40), -1)
            cv2.putText(frame, "BODY FRAME (top view)", (width - 410, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "^ nose", (width - 250, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            draw_arrow(frame, v_x, v_y, panel_origin)

            err = guidance.center_error_norm
            err_txt = f"e=({err[0]:+.3f},{err[1]:+.3f})" if err else "e=(--,--)"
            yaw_txt = (f"{guidance.yaw_error_deg:+.1f}deg"
                       if guidance.yaw_error_deg is not None else "--")
            dist_txt = (f"{guidance.center_distance_m:.2f}m"
                        if guidance.center_distance_m is not None else "--")

            if land_ready:
                align_txt, align_color = ">>> LAND APPROVED <<<", (0, 255, 0)
            elif aligned_since is not None:
                align_txt = f"aligned {hold_sec:.1f}/{LAND_HOLD_SEC:.0f}s"
                align_color = (0, 255, 255)
            else:
                align_txt, align_color = "not aligned", (0, 80, 255)

            sign_txt = (f"sign lat={sign_lat:+.0f} fwd={sign_fwd:+.0f} "
                        f"yaw={sign_yaw:+.0f}   [x/y/t]")

            lines = [
                (sign_txt, (0, 220, 80)),
                (f"status: {guidance.status}  ids={[m.marker_id for m in markers]}",
                 (255, 255, 255)),
                (f"{err_txt}  yaw={yaw_txt}  dist={dist_txt}", (255, 255, 255)),
                (f"marker={marker_px:.0f}px  hit={hit_rate*100:.0f}%  "
                 f"LOG={'ON' if log_writer else 'off'}", (255, 255, 255)),
                (f"CMD vx={v_x:+.2f} vy={v_y:+.2f} yaw={yaw_rate:+.1f}",
                 (0, 255, 255)),
                (f">>> {describe(v_x, v_y, yaw_rate)}", (0, 255, 255)),
                (align_txt, align_color),
                ("DRY RUN - no command sent", (0, 80, 255)),
            ]
            for i, (text, color) in enumerate(lines):
                cv2.putText(frame, text, (20, 110 + i * 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 3,
                            cv2.LINE_AA)

            cv2.imshow("servo direction", frame)

            # ---------------- 로깅 / 콘솔 ----------------
            if log_writer is not None:
                log_writer.writerow([
                    f"{now - t_start:.3f}",
                    "|".join(str(m.marker_id) for m in markers),
                    guidance.status,
                    f"{err[0]:.4f}" if err else "",
                    f"{err[1]:.4f}" if err else "",
                    (f"{guidance.yaw_error_deg:.2f}"
                     if guidance.yaw_error_deg is not None else ""),
                    (f"{guidance.center_distance_m:.3f}"
                     if guidance.center_distance_m is not None else ""),
                    f"{marker_px:.1f}",
                    f"{v_x:.3f}", f"{v_y:.3f}", f"{yaw_rate:.2f}",
                    f"{hold_sec:.2f}",
                ])

            frame_no += 1
            if frame_no % 20 == 0 and guidance.has_center:
                print(f"{err_txt} yaw={yaw_txt} {marker_px:.0f}px -> "
                      f"vx={v_x:+.2f} vy={v_y:+.2f} yaw={yaw_rate:+.1f} "
                      f"| {describe(v_x, v_y, yaw_rate)} | {align_txt}",
                      flush=True)

            # ---------------- 키 입력 ----------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("x"):
                sign_lat = -sign_lat
                print(f"\n=== 좌우 부호 {sign_lat:+.0f} ===", flush=True)
            elif key == ord("y"):
                sign_fwd = -sign_fwd
                print(f"\n=== 전후 부호 {sign_fwd:+.0f} ===", flush=True)
            elif key == ord("t"):
                sign_yaw = -sign_yaw
                print(f"\n=== 요 부호 {sign_yaw:+.0f} ===", flush=True)
            elif key == ord("r"):
                aligned_since = None
                print("\n=== 타이머 리셋 ===", flush=True)
            elif key == ord("l"):
                if log_writer is None:
                    log_handle, log_writer, log_path = open_log()
                else:
                    log_handle.close()
                    print(f"[LOG] 종료: {log_path}", flush=True)
                    log_handle = log_writer = None
            elif key == ord("q"):
                break
    finally:
        if log_handle is not None:
            log_handle.close()
            print(f"\n[LOG] 저장됨: {log_path}", flush=True)
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n검출률 {seen_frames}/{total_frames} "
              f"({seen_frames/max(total_frames,1)*100:.1f}%)")
        print(f"확정된 부호: SIGN_LATERAL={sign_lat:+.0f}  "
              f"SIGN_FORWARD={sign_fwd:+.0f}  SIGN_YAW={sign_yaw:+.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
