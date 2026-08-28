from __future__ import annotations

import argparse
from collections import deque
import sys
import time
import math
import numpy as np
from pathlib import Path

# ==============================================================
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
# ==============================================================

from pymavlink import mavutil
import cv2

import threading
import asyncio
from mavsdk import System

from flight_fsm import FlightFSM, FlightPhase



from medifly_vision import (
    CameraModel, CasualtyDetector,
    VertiportArucoConfig, approximate_camera_matrix,
    compute_vertiport_guidance, describe_capture, detect_aruco_markers,
    draw_vertiport_overlay, open_usb_camera,
)



def run_fsm_background(fsm):
    async def start_fsm():
        print("[SYSTEM] FSM용 픽스호크(14551 포트) 연결 대기 중...")
        await fsm.drone.connect(system_address="udpin://127.0.0.1:14551")
        async for state in fsm.drone.core.connection_state():
            if state.is_connected:
                print("[SYSTEM] FSM 픽스호크 연결 완료!")
                break
        
        print("[INFO] 이륙 전 그리퍼 잠금(180도) 초기화 중...")
        await asyncio.to_thread(fsm._set_servo_angle_sync, 180)
        asyncio.create_task(fsm.subscribe_altitude())
        await fsm.run_fsm()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_fsm())


SIGN_GIMBAL_PITCH = -1.0


class GimbalTracker:
    def __init__(self, frame_height=1080):
        self.center_y = frame_height / 2.0
        self.target_pitch = 0.0    
        self.target_roll = 0.0     
        self.Kp_y = 0.005  
        self.Kd_y = 0.002  
        self.prev_err_y = 0
        self.prev_time = time.monotonic()
        self.is_first_lock = True  

    def update_tracking_pitch(self, target_cy):
        current_time = time.monotonic()
        dt = current_time - self.prev_time
        if dt <= 0: dt = 0.033
        err_y = target_cy - self.center_y
        if self.is_first_lock:
            self.prev_err_y = err_y
            self.is_first_lock = False
        d_err_y = (err_y - self.prev_err_y) / dt
        delta_pitch = SIGN_GIMBAL_PITCH * ((err_y * self.Kp_y) + (d_err_y * self.Kd_y))
        delta_pitch = max(-0.5, min(0.5, delta_pitch))
        self.target_pitch += delta_pitch
        self.target_pitch = max(-90.0, min(90.0, self.target_pitch))
        self.prev_err_y = err_y
        self.prev_time = current_time

    def reset_lock(self):
        self.is_first_lock = True

    def get_servo_commands(self, drone_roll, drone_pitch, is_tracking=False):
        if not is_tracking:
            self.target_pitch = self.target_pitch * 0.90
            if abs(self.target_pitch) < 0.5: self.target_pitch = 0.0
        return self.target_roll - drone_roll, self.target_pitch - drone_pitch

class StableTargetSelector:
    def __init__(self, acquire_confidence=0.60, follow_confidence=0.35, max_center_distance_ratio=0.15, max_missed_frames=3):
        self.acquire_confidence = acquire_confidence
        self.follow_confidence = follow_confidence
        self.max_center_distance_ratio = max_center_distance_ratio
        self.max_missed_frames = max_missed_frames
        self.active_bbox = None
        self.missed_frames = 0

    def select(self, detections, frame_width, frame_height, allow_reacquire=True):
        candidates = [d for d in detections if d.confidence >= self.follow_confidence]
        if self.active_bbox is None: return self._acquire(candidates)
        matching = [d for d in candidates if self._is_consistent(d.bbox_xyxy, self.active_bbox, frame_width, frame_height)]
        if matching:
            selected = max(matching, key=lambda d: (self._box_iou(d.bbox_xyxy, self.active_bbox), d.confidence))
            self.active_bbox = selected.bbox_xyxy
            self.missed_frames = 0
            return selected
        self.missed_frames += 1
        if self.missed_frames >= self.max_missed_frames:
            if allow_reacquire:
                self.reset()
                return self._acquire(candidates)
        return None

    def reset(self):
        self.active_bbox = None
        self.missed_frames = 0

    def _acquire(self, detections):
        eligible = [d for d in detections if d.confidence >= self.acquire_confidence]
        if not eligible: return None
        selected = max(eligible, key=lambda d: d.confidence)
        self.active_bbox = selected.bbox_xyxy
        self.missed_frames = 0
        return selected

    def _is_consistent(self, candidate_bbox, active_bbox, frame_width, frame_height):
        if self._box_iou(candidate_bbox, active_bbox) >= 0.05: return True
        candidate_center = self._box_center(candidate_bbox)
        active_center = self._box_center(active_bbox)
        distance = math.hypot(candidate_center[0] - active_center[0], candidate_center[1] - active_center[1])
        diagonal = max(math.hypot(frame_width, frame_height), 1.0)
        return distance / diagonal <= self.max_center_distance_ratio

    @staticmethod
    def _box_center(bbox): return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    @staticmethod
    def _box_iou(first, second):
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        if intersection <= 0.0: return 0.0
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

def set_gimbal_angle(master, pitch_deg, roll_deg):
    pitch_deg = max(-90.0, min(90.0, pitch_deg))
    roll_deg = max(-90.0, min(90.0, roll_deg))
    master.mav.command_long_send(1, 1, 205, 0, pitch_deg, roll_deg, 0.0, 0.0, 0.0, 0.0, 2.0)

def main() -> int:
    try: sys.stdout = open('/dev/tty', 'w')
    except Exception: pass
        
    args = parse_args()
    model_path = resolve_model_path(args.model, prefer_engine=args.prefer_engine)
    print(f"[INFO] YOLO model: {model_path}")

    sys.stdout.write("[INFO] 픽스호크(/dev/ttyTHS1) 연결 중...\n")
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    heartbeat = master.wait_heartbeat(timeout=args.mavlink_timeout)
    if heartbeat is None:
        print(f"[ERROR] MAVLink heartbeat timeout", file=sys.stderr)
        return 2
    sys.stdout.write("[INFO] 픽스호크 연결 완료!\n")

    detector = CasualtyDetector(
        model_path=str(model_path), confidence=args.confidence, image_size=args.image_size, device=args.device,
        adaptive_tiling=args.adaptive_tiling, tile_overlap=args.tile_overlap, tile_activation_confidence=args.tracking_confidence,
        roi_margin=args.roi_margin, tile_target_distance_ratio=args.max_center_distance_ratio,
        global_search_frames=args.global_search_frames, tile_refresh_frames=args.tile_refresh_frames, tile_lost_frames=args.tile_lost_frames,
    )

    tracker = GimbalTracker(frame_height=1200)
    target_selector = StableTargetSelector(
        acquire_confidence=args.tracking_confidence, follow_confidence=args.confidence,
        max_center_distance_ratio=args.max_center_distance_ratio, max_missed_frames=args.target_lost_frames,
    )



    sys.stdout.write("[INFO] USB 카메라 시작 중...\n")
    capture = open_usb_camera(device=args.camera_id, width=1920, height=1200)
    if capture is None:
        print(f"\n[ERROR] USB 카메라 device={args.camera_id} 를 열 수 없습니다!",
              file=sys.stderr)
        return 1
    sys.stdout.write(f"[INFO] 카메라: {describe_capture(capture)}\n")




    fsm_drone = System()
    fsm = FlightFSM(fsm_drone)
    if args.bench_no_rtl:
        fsm.rtl_commanded = True
    fsm_thread = threading.Thread(target=run_fsm_background, args=(fsm,), daemon=True)
    fsm_thread.start()
    
    drone_roll, drone_pitch = 0.0, 0.0
    window_name = "MEDIFLY-MAP casualty detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    detection_history = deque()
    TARGET_LOCKED = False
    last_seen_time = 0.0  
    last_phase = None     

    aruco_config = VertiportArucoConfig(
        center_marker_id=0,          # 버티포트 중앙
        front_marker_id=1,           # 드론 정방향
        marker_size_m=args.marker_size_m,
        desired_front="image_up",
    )
    aruco_camera_matrix = None
    aruco_hit_frames = 0


    try:
        while True:
            while True:  
                msg = master.recv_match(type='ATTITUDE', blocking=False)
                if not msg: break
                if msg.get_srcComponent() == 1:
                    drone_roll = math.degrees(msg.roll)
                    drone_pitch = math.degrees(msg.pitch)

            ok, frame = capture.read()
            if not ok or frame is None: continue

            safe_frame = np.ascontiguousarray(frame.copy(), dtype=np.uint8)
            height, width = safe_frame.shape[:2]
            tracker.center_y = height / 2.0
            camera = CameraModel(width_px=width, height_px=height, horizontal_fov_deg=args.hfov_deg, vertical_fov_deg=args.vfov_deg)
            if fsm.current_phase.value >= 4:
                # ---- [P4/P5] 아루코 단계: 짐벌 정하단 + 마커 탐색 ----
                servo_roll = drone_roll * -1.0
                servo_pitch = (drone_pitch + 20) * -1.0   # +20 오프셋 = 정하단
                set_gimbal_angle(master, servo_pitch, servo_roll)
                fsm.gimbal_pitch = servo_pitch

                if aruco_camera_matrix is None:
                    aruco_camera_matrix = approximate_camera_matrix(
                        width, height, args.hfov_deg, args.vfov_deg)

                markers = detect_aruco_markers(
                    safe_frame,
                    aruco_dict_name=aruco_config.aruco_dict_name,
                    marker_size_m=aruco_config.marker_size_m,
                    camera_matrix=aruco_camera_matrix,
                )
                guidance = compute_vertiport_guidance(
                    frame_width_px=width, frame_height_px=height,
                    markers=markers, config=aruco_config,
                )

                # 한 프레임 오탐으로 RTL을 끊지 않도록 연속 검출 요구
                aruco_hit_frames = aruco_hit_frames + 1 if guidance.has_center else 0
                fsm.aruco_detected = aruco_hit_frames >= args.aruco_lock_frames
                fsm.aruco_status = guidance.status
                fsm.aruco_center_error = guidance.center_error_norm
                fsm.aruco_yaw_error = guidance.yaw_error_deg
                fsm.aruco_aligned = (
                    guidance.has_center and guidance.has_front
                    and guidance.center_error_norm is not None
                    and guidance.yaw_error_deg is not None
                    and abs(guidance.center_error_norm[0]) <= 0.08
                    and abs(guidance.center_error_norm[1]) <= 0.08
                    and abs(guidance.yaw_error_deg) <= 8.0
                )

                draw_vertiport_overlay(safe_frame, markers, guidance,
                                       desired_front=aruco_config.desired_front)

                err = guidance.center_error_norm
                err_txt = f"e=({err[0]:+.2f},{err[1]:+.2f})" if err else "e=--"
                ids_txt = ",".join(str(m.marker_id) for m in markers) or "-"
                v_stat = f"🛬[ARUCO id={ids_txt} {err_txt} n={aruco_hit_frames}]"
                detections = []
                stable_target = None
            else:
                # [P0~P3] 원본 YOLO 추적
                detections = detector.detect_adaptive(
                    frame_bgr=safe_frame, altitude_m=args.height_m, camera=camera, 
                    drop_distance_m=args.drop_distance_m, drop_direction=args.drop_direction
                )
                
                margin = args.roi_margin
                roi_left, roi_right = width * margin, width * (1.0 - margin)
                roi_top, roi_bottom = height * margin, height * (1.0 - margin)
                cv2.rectangle(safe_frame, (int(roi_left), int(roi_top)), (int(roi_right), int(roi_bottom)), (255, 0, 0), 2)

                filtered_detections = [d for d in detections if roi_left <= d.center_px[0] <= roi_right and roi_top <= d.center_px[1] <= roi_bottom]
                detections = filtered_detections

                current_time = time.monotonic()
                stable_target = target_selector.select(detections, width, height, allow_reacquire=not TARGET_LOCKED)

                detection_history.append((current_time, 1 if stable_target is not None else 0))
                cutoff_time = current_time - args.history_seconds
                while detection_history and detection_history[0][0] < cutoff_time: detection_history.popleft()
                if stable_target is not None: last_seen_time = current_time  
                hit_rate = sum(value for _, value in detection_history) / max(len(detection_history), 1)
                history_ready = len(detection_history) >= args.min_history_samples

                if history_ready and hit_rate >= args.lock_hit_rate: TARGET_LOCKED = True
                if TARGET_LOCKED and history_ready and hit_rate < args.unlock_hit_rate: TARGET_LOCKED = False; detection_history.clear(); target_selector.reset()
                if TARGET_LOCKED and (current_time - last_seen_time > 2.0): TARGET_LOCKED = False; detection_history.clear(); target_selector.reset()

                if TARGET_LOCKED:
                    if stable_target is not None:
                        tracker.update_tracking_pitch(stable_target.center_px[1])
                        fsm.target_cx = stable_target.center_px[0]
                        fsm.camera_center_x = width / 2.0
                    servo_roll, servo_pitch = tracker.get_servo_commands(drone_roll, drone_pitch, is_tracking=True)
                else:
                    tracker.reset_lock()  
                    servo_roll, servo_pitch = tracker.get_servo_commands(drone_roll, drone_pitch, is_tracking=False)
                
                set_gimbal_angle(master, servo_pitch, servo_roll)
                fsm.target_locked = TARGET_LOCKED
                fsm.gimbal_pitch = servo_pitch
                v_stat = "🎯[YOLO]" if TARGET_LOCKED else "🔍[YOLO]"

            if last_phase != fsm.current_phase:
                if last_phase is not None: print()
                last_phase = fsm.current_phase
                
            alt_txt = f"{fsm.altitude_m:.1f}m" if fsm.altitude_m is not None else "--"
            subtitles = f"🤖 [P{fsm.current_phase.value}] {fsm.status_text} | 📷 {v_stat} ALT:{alt_txt} P:{servo_pitch:.0f} | ⏱️ {detector.last_inference_ms:.0f}ms"
            if len(subtitles) > 130: subtitles = subtitles[:127] + "..."
            
            print(f"\r\033[K{subtitles}", end="", flush=True)

            if fsm.current_phase.value < 4:
                draw_overlay(safe_frame, detections, stable_target)
            cv2.imshow(window_name, safe_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("4"):   # 지상 테스트: P4 강제 전환
                fsm.current_phase = FlightPhase.PHASE_4_RTL
                fsm.rtl_commanded = True     # 실제 RTL 명령 차단
            if key == ord("q"):
                print("\n[INFO] 종료 중... 짐벌을 중앙으로 정렬합니다.")
                break
            if key == ord("8"):   # 지상 테스트: P8 버티포트 시퀀스 강제 실행
                fsm.vertiport_done = False   # 재실행 허용
                fsm.current_phase = FlightPhase.PHASE_8_VERTIPORT
                print("\n[TEST] 8키 - 5초 후 버티포트 시퀀스 시작")
                
    finally:
        try: set_gimbal_angle(master, 0.0, 0.0)
        except Exception: pass
        capture.release()
        cv2.destroyAllWindows()

def draw_overlay(frame, detections, stable_target=None) -> None:
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        center_x, center_y = [int(round(value)) for value in detection.center_px]
        drop_x, drop_y = [int(round(value)) for value in detection.drop.drop_point_px]
        is_stable = detection == stable_target
        color = (0, 220, 80) if is_stable else (0, 180, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        cv2.circle(frame, (drop_x, drop_y), 7, (0, 80, 255), -1)
        cv2.line(frame, (center_x, center_y), (drop_x, drop_y), (0, 80, 255), 2)
        label = (f"{detection.class_name} {detection.confidence:.2f} [{detection.inference_source}] | "
                 f"p=({detection.ground_m[0]:+.2f},{detection.ground_m[1]:+.2f}) "
                 f"d=({detection.drop.drop_ground_m[0]:+.2f},{detection.drop.drop_ground_m[1]:+.2f})")
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/medifly_person_down_colab/yolov8n_v2_best.pt")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--height-m", type=float, default=0.8)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--tracking-confidence", type=float, default=0.60)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mavlink-timeout", type=float, default=15.0)
    parser.add_argument("--hfov-deg", type=float, default=70.0)
    parser.add_argument("--vfov-deg", type=float, default=43.0)
    parser.add_argument("--drop-distance-m", type=float, default=1.5)
    parser.add_argument("--roi-margin", type=float, default=0.10)
    parser.add_argument("--history-seconds", type=float, default=0.5)
    parser.add_argument("--min-history-samples", type=int, default=3)
    parser.add_argument("--lock-hit-rate", type=float, default=0.70)
    parser.add_argument("--unlock-hit-rate", type=float, default=0.30)
    parser.add_argument("--max-center-distance-ratio", type=float, default=0.15)
    parser.add_argument("--target-lost-frames", type=int, default=3)
    parser.add_argument("--tile-overlap", type=float, default=0.15)
    parser.add_argument("--global-search-frames", type=int, default=6)
    parser.add_argument("--tile-refresh-frames", type=int, default=15)
    parser.add_argument("--tile-lost-frames", type=int, default=3)
    parser.add_argument("--marker-size-m", type=float, default=0.22)
    parser.add_argument("--aruco-lock-frames", type=int, default=5)
    parser.add_argument("--bench-no-rtl", action="store_true",
                        help="지상 테스트: 실제 RTL 명령을 보내지 않는다")
    tiling_group = parser.add_mutually_exclusive_group()
    tiling_group.add_argument("--adaptive-tiling", dest="adaptive_tiling", action="store_true")
    tiling_group.add_argument("--no-adaptive-tiling", dest="adaptive_tiling", action="store_false")
    engine_group = parser.add_mutually_exclusive_group()
    engine_group.add_argument("--prefer-engine", dest="prefer_engine", action="store_true")
    engine_group.add_argument("--no-prefer-engine", dest="prefer_engine", action="store_false")
    parser.set_defaults(adaptive_tiling=True, prefer_engine=True)
    parser.add_argument("--drop-direction", choices=["image_up", "image_down", "image_left", "image_right"], default="image_down")
    return parser.parse_args()

def resolve_model_path(value: str, prefer_engine: bool) -> Path:
    requested = Path(value).expanduser()
    candidates = [requested] if requested.is_absolute() else [Path.cwd() / requested, ROOT / requested]
    model_path = next((path.resolve() for path in candidates if path.is_file()), None)
    if model_path is None: raise FileNotFoundError(f"YOLO model not found: {value}")
    if prefer_engine and model_path.suffix.lower() == ".pt":
        engine_path = model_path.with_suffix(".engine")
        if engine_path.is_file(): return engine_path
    return model_path

if __name__ == "__main__":
    raise SystemExit(main())
