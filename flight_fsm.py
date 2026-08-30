import asyncio
import sys
import time
import Jetson.GPIO as GPIO
from enum import Enum
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError

try:
    import requests
except ImportError:
    requests = None


# ======================================================================
# 비주얼 서보잉 파라미터
#
# 부호 5개는 모두 실측으로 확정했다.
# 카메라를 다시 장착하거나 짐벌 마운트를 바꾸면 전부 재검증해야 한다.
#
# PX4 파라미터 전제
#   RTL_RETURN_ALT 5m / RTL_DESCEND_ALT 5m / RTL_MIN_DIST 3m
#   MPC_XY_CRUISE 4.0
# ======================================================================
SIGN_LATERAL = +1.0        # 좌우 부호
SIGN_FORWARD = -1.0        # 전후 부호
SIGN_YAW = -1.0            # 요 부호 (34도 오차 실측 확정)
SIGN_APPROACH_YAW = +1.0   # P2 접근 요 부호 (녹화 확인 확정)

KP_LATERAL = 1.2           # 정규화 오차 -> m/s
KP_YAW = 0.45              # deg -> deg/s
MAX_SPEED = 0.9            # 수평 속도 상한 (m/s)
MAX_YAW_RATE = 45.0        # deg/s 상한
DEADBAND = 0.05            # 중심 오차 데드밴드
YAW_DEADBAND = 4.0         # 요 오차 데드밴드 (deg)
FOV_RATIO = 0.56           # 전후/좌우 응답 밸런스

# --- P5 수평 정렬 ---
ALIGN_HOLD_SEC = 3.0       # 정렬 조건을 이만큼 연속 유지해야 하강 시작
LOST_HOLD_SEC = 3.0        # 마커 로스트 후 제자리 유지 시간
LOST_CLIMB_SEC = 10.0      # 이 시간 넘으면 RTL 폴백
CLIMB_SPEED = 0.3          # 재탐색 상승 속도 (m/s)
MAX_SERVO_ALT = 12.0       # 재탐색 상승 고도 상한 (m)
SERVO_TIMEOUT_SEC = 90.0   # P5 전체 제한 시간
POS_TOL = 0.08             # 위치 정렬 허용 오차 (정규화)
POS_TOL_EXIT = 0.14        # 이보다 벌어지면 위치 정렬로 복귀 (히스테리시스)
YAW_TOL = 8.0              # 방향 정렬 허용 오차 (deg)

# --- P6 하강 ---
DESCENT_SPEED = 0.5        # 하강 속도 (m/s). +z 가 아래다
DESCENT_YAW_SCALE = 0.5    # 하강 중에는 요 제어를 절반으로
MARKER_GRACE_SEC = 1.0     # 검출이 이만큼 끊기면 진짜 유실로 본다
DESCENT_TIMEOUT_SEC = 60.0 # 이 시간 넘으면 land() 위임

# --- 그리퍼 ---
GRIPPER_OPEN = 0           # 열림. 투하 후 착륙까지 이 상태를 유지한다
GRIPPER_CLOSE = 180        # 닫힘. 상자 파지
GRIPPER_SETTLE_SEC = 1.5   # 체결 후 안정화 대기

# --- 버티포트 ESP32 ---
ESP32_URL = "http://vertiport.local"
ESP32_TRIES = 3
ESP32_TIMEOUT = 15.0
LAND_SETTLE_SEC = 2.0      # 접지 후 기체가 안정될 때까지 대기
P8_DELAY_SEC = 5.0         # 버티포트 시퀀스 시작 전 대기

def esp(path: str) -> bool:
    """
    버티포트 ESP32 에 HTTP 요청을 보낸다. 동기 함수다.
    반드시 asyncio.to_thread 로 감싸 호출해야 한다.
    직접 await 없이 부르면 offboard setpoint 스트림이 끊긴다.
    """
    if requests is None:
        print("[ESP32] requests 미설치", file=sys.stderr)
        return False
    for attempt in range(ESP32_TRIES):
        try:
            r = requests.get(ESP32_URL + path, timeout=ESP32_TIMEOUT)
            if r.status_code == 200:
                print(f"\n[ESP32] {path} OK")
                return True
            print(f"\n[ESP32] {path} -> {r.status_code}")
        except Exception as exc:
            print(f"\n[ESP32] {path} 실패 {attempt+1}/{ESP32_TRIES}: {exc}")
    return False


class FlightPhase(Enum):
    PHASE_0_INIT = 0
    PHASE_1_SEARCHING = 1
    PHASE_2_APPROACHING = 2
    PHASE_3_DROPPING = 3
    PHASE_4_RTL = 4
    PHASE_5_VERTIPORT_HOLD = 5   # 아루코 기반 수평 정렬
    PHASE_6_DESCENDING = 6       # 서보잉 하강
    PHASE_7_LANDED = 7           # land() 위임, 접지 대기
    PHASE_8_VERTIPORT = 8        # 리프트로 새 상자 체결


class FlightFSM:
    def __init__(self, drone: System):
        self.drone = drone
        self.current_phase = FlightPhase.PHASE_0_INIT
        self.target_locked = False
        self.target_cx = 0.0
        self.camera_center_x = 960.0     # 1920 기준. 비전 루프가 덮어쓴다
        self.Kp_yaw = 0.05
        self.gimbal_pitch = 0.0
        self.is_dropping = False
        self.rtl_commanded = False
        self.hold_engaged = False
        self.align_stage = "POS"         # POS -> YAW -> HOLD

        self.status_text = "[P0] INIT 대기"

        # 비전 루프가 매 프레임 갱신하는 서보잉 입력
        self.aruco_detected = False
        self.aruco_status = "NO_MARKER"
        self.aruco_center_error = None   # (e_x, e_y) 정규화 오차
        self.aruco_yaw_error = None      # deg
        self.aruco_aligned = False

        self.altitude_m = None           # 지면 기준 고도 (없을 수 있다)
        self.in_air = True               # 접지 판정용
        self.marker_lost_since = None
        self.aligned_since = None
        self.servo_started_at = None
        self.descent_started_at = None
        self.landed_at = None            # 접지 확인 시각
        self.servo_abandoned = False
        self.land_commanded = False
        self.vertiport_done = False      # P8 시퀀스 1회 실행 가드

        self.SERVO_PIN = 32
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.SERVO_PIN, GPIO.OUT, initial=GPIO.LOW)

    async def run_fsm(self):
        print("\n[SYSTEM] 🚀 FSM 비행 제어 루프 시작")
        while True:
            if self.current_phase == FlightPhase.PHASE_0_INIT:
                await self.handle_phase_0_init()
            elif self.current_phase == FlightPhase.PHASE_1_SEARCHING:
                await self.handle_phase_1_searching()
            elif self.current_phase == FlightPhase.PHASE_2_APPROACHING:
                await self.handle_phase_2_approaching()
            elif self.current_phase == FlightPhase.PHASE_3_DROPPING:
                await self.handle_phase_3_dropping()
            elif self.current_phase == FlightPhase.PHASE_4_RTL:
                await self.handle_phase_4_rtl()
            elif self.current_phase == FlightPhase.PHASE_5_VERTIPORT_HOLD:
                await self.handle_phase_5_vertiport_hold()
            elif self.current_phase == FlightPhase.PHASE_6_DESCENDING:
                await self.handle_phase_6_descending()
            elif self.current_phase == FlightPhase.PHASE_7_LANDED:
                await self.handle_phase_7_landed()
            elif self.current_phase == FlightPhase.PHASE_8_VERTIPORT:
                await self.handle_phase_8_vertiport()

            await asyncio.sleep(0.05)

    # ==================================================================
    # P0 ~ P2 : YOLO 기반 탐색과 접근
    # ==================================================================
    async def handle_phase_0_init(self):
        if self.target_locked:
            self.status_text = "[P0->2] 🎯 락온! 오프보드 진입"
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            try:
                await self.drone.offboard.start()
                self.current_phase = FlightPhase.PHASE_2_APPROACHING
            except OffboardError:
                self.status_text = "[P0] ❌ 오프보드 실패 (재시도)"
                await asyncio.sleep(1)
        else:
            self.status_text = "[P0] ✈️ 미션 비행 (락온 대기)"

    async def handle_phase_1_searching(self):
        if self.target_locked:
            self.current_phase = FlightPhase.PHASE_2_APPROACHING
            return
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        self.status_text = "[P1] 🔍 타겟 로스트 (호버링)"

    async def handle_phase_2_approaching(self):
        if not self.target_locked:
            self.current_phase = FlightPhase.PHASE_1_SEARCHING
            return
        if self.gimbal_pitch <= -14.0:
            self.current_phase = FlightPhase.PHASE_3_DROPPING
            return
        error_x = self.target_cx - self.camera_center_x
        yaw_speed_cmd = max(-20.0, min(20.0,
                            SIGN_APPROACH_YAW * error_x * self.Kp_yaw))
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(1.0, 0.0, 0.0, yaw_speed_cmd))
        self.status_text = f"[P2] 🚀 접근 | X오차:{error_x:.0f} Yaw:{yaw_speed_cmd:.1f}"

    # ==================================================================
    # P3 : 투하 + 버티포트 사출 요청
    # ==================================================================
    async def handle_phase_3_dropping(self):
        """
        투하 후 그리퍼를 닫지 않는다.
        착륙 시 그리퍼가 열려 있어야 리프트가 올린 상자를 물 수 있다.
        """
        if self.is_dropping:
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            return

        self.is_dropping = True

        self.status_text = "[P3] ⏱️ 타점 도달 (5초 대기)"
        wait_end = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < wait_end:
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)

        self.status_text = "[P3] 📦 페이로드 투하 중..."
        gripper_task = asyncio.create_task(
            asyncio.to_thread(self._set_servo_angle_sync, GRIPPER_OPEN, True))

        drop_end = asyncio.get_event_loop().time() + 1.5
        while asyncio.get_event_loop().time() < drop_end:
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)
        await gripper_task

        # 버티포트에 새 상자 사출을 요청한다.
        # to_thread 로 감싸야 이 몇 초 동안에도 setpoint 스트림이 유지된다.
        # 직접 호출하면 offboard 가 failsafe 로 풀린다.
        self.status_text = "[P3] 📡 버티포트 사출 요청 중..."
        eject_task = asyncio.create_task(asyncio.to_thread(esp, "/eject"))
        while not eject_task.done():
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)
        if not eject_task.result():
            print("\n[P3] ⚠️ 사출 요청 실패. 착륙 후 수동 처리 필요")

        self.status_text = "[P3] ✅ 투하 완료 (그리퍼 개방 유지)"
        self.current_phase = FlightPhase.PHASE_4_RTL

    # ==================================================================
    # P4 : RTL 복귀 + 마커 탐색
    # ==================================================================
    async def handle_phase_4_rtl(self):
        if not self.rtl_commanded:
            self.rtl_commanded = True
            self.status_text = "[P4] 🏠 RTL (복귀) 전환 중..."
            try:
                await self.drone.action.return_to_launch()
            except Exception:
                self.status_text = "[P4] ❌ RTL 실패! 수동 개입"

        if self.aruco_detected and not self.servo_abandoned:
            self.status_text = "[P4->5] 🛬 마커 발견! 정렬 전환"
            self.current_phase = FlightPhase.PHASE_5_VERTIPORT_HOLD
        elif self.servo_abandoned:
            self.status_text = "[P4] 🏠 RTL 착륙 (서보잉 포기됨)"
            if not self.in_air:
                self.current_phase = FlightPhase.PHASE_7_LANDED
        else:
            self.status_text = f"[P4] 🏠 RTL 복귀 중 ({self.aruco_status})"

    # ==================================================================
    # 서보잉 공통
    # ==================================================================
    def compute_servo_velocity(self, yaw_scale=1.0):
        """화면 오차 -> body frame 수평 속도. 부호는 실측 확정값이다."""
        error = self.aruco_center_error
        if error is None:
            return 0.0, 0.0, 0.0

        e_x, e_y = error
        if abs(e_x) < DEADBAND:
            e_x = 0.0
        if abs(e_y) < DEADBAND:
            e_y = 0.0

        v_y = SIGN_LATERAL * KP_LATERAL * e_x
        v_x = SIGN_FORWARD * KP_LATERAL * e_y * FOV_RATIO

        yaw_rate = 0.0
        if self.aruco_yaw_error is not None:
            yaw_err = self.aruco_yaw_error
            if abs(yaw_err) < YAW_DEADBAND:
                yaw_err = 0.0
            yaw_rate = SIGN_YAW * KP_YAW * yaw_err * yaw_scale

        v_x = max(-MAX_SPEED, min(MAX_SPEED, v_x))
        v_y = max(-MAX_SPEED, min(MAX_SPEED, v_y))
        yaw_rate = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, yaw_rate))
        return v_x, v_y, yaw_rate

    # ==================================================================
    # P5 : 2단계 수평 정렬 (고도 유지)
    # ==================================================================
    async def handle_phase_5_vertiport_hold(self):
        loop_now = asyncio.get_event_loop().time()

        if not self.hold_engaged:
            self.status_text = "[P5] ⏸️ RTL 취소, 오프보드 진입"
            for _ in range(10):
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                await asyncio.sleep(0.05)
            try:
                await self.drone.offboard.start()
                self.hold_engaged = True
                self.servo_started_at = loop_now
            except OffboardError as e:
                self.status_text = f"[P5] ❌ 오프보드 거부 ({e}) 재시도"
                await asyncio.sleep(1)
                return

        if (self.servo_started_at is not None
                and loop_now - self.servo_started_at > SERVO_TIMEOUT_SEC):
            elapsed = loop_now - self.servo_started_at
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            self.status_text = f"[P5] ⏰ 정렬 지연 {elapsed:.0f}s (수동 개입 권장)"
            return

        # --- 마커 로스트 처리 ---
        if self.aruco_center_error is None:
            self.aligned_since = None
            self.align_stage = "POS"     # 재획득 시 위치 정렬부터 다시
            if self.marker_lost_since is None:
                self.marker_lost_since = loop_now
            lost_sec = loop_now - self.marker_lost_since

            if lost_sec > LOST_CLIMB_SEC:
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                self.status_text = (f"[P5] ❌ 마커 로스트 {lost_sec:.0f}s "
                                    f"(수동 개입 권장)")
                return

            if lost_sec > LOST_HOLD_SEC:
                too_high = (self.altitude_m is not None
                            and self.altitude_m > MAX_SERVO_ALT)
                climb = 0.0 if too_high else -CLIMB_SPEED
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, climb, 0.0))
                self.status_text = (
                    f"[P5] ⬆️ 재탐색 상승 {lost_sec:.1f}s" if climb
                    else f"[P5] ⏸️ 고도상한 유지 {lost_sec:.1f}s")
            else:
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                self.status_text = f"[P5] ⏸️ 마커 로스트 {lost_sec:.1f}s"
            return

        self.marker_lost_since = None

        # --- 2단계 정렬: 위치 먼저, 방향 나중 ---
        # 요 회전은 오차 좌표계를 함께 돌려 병진 제어와 간섭한다.
        e_x, e_y = self.aruco_center_error
        pos_err = max(abs(e_x), abs(e_y))
        yaw_err = (abs(self.aruco_yaw_error)
                   if self.aruco_yaw_error is not None else 999.0)

        if pos_err > POS_TOL_EXIT:
            self.align_stage = "POS"

        if self.align_stage == "POS":
            v_x, v_y, _ = self.compute_servo_velocity()
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(v_x, v_y, 0.0, 0.0))
            self.aligned_since = None
            self.status_text = (f"[P5a] 📍 위치정렬 e{pos_err:.2f} "
                                f"vx{v_x:+.2f} vy{v_y:+.2f}")
            if pos_err <= POS_TOL:
                self.align_stage = "YAW"
            return

        if self.align_stage == "YAW":
            _, _, yaw_rate = self.compute_servo_velocity()
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, yaw_rate))
            self.aligned_since = None
            self.status_text = f"[P5b] 🧭 방향정렬 {yaw_err:.0f}도 yaw{yaw_rate:+.0f}"
            if yaw_err <= YAW_TOL:
                self.align_stage = "HOLD"
            return

        # HOLD: 둘 다 맞음. 미세 보정만 하며 유지 시간을 센다
        v_x, v_y, yaw_rate = self.compute_servo_velocity()
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(v_x, v_y, 0.0, yaw_rate))

        if self.aligned_since is None:
            self.aligned_since = loop_now
        hold_sec = loop_now - self.aligned_since
        self.status_text = f"[P5] ✅ 정렬유지 {hold_sec:.1f}/{ALIGN_HOLD_SEC:.0f}s"

        if hold_sec >= ALIGN_HOLD_SEC:
            self.descent_started_at = loop_now
            self.current_phase = FlightPhase.PHASE_6_DESCENDING

    # ==================================================================
    # P6 : 서보잉 하강
    # ==================================================================
    async def handle_phase_6_descending(self):
        """
        마커가 보이는 동안 수평 보정을 하며 등속 하강한다.
        마커를 잃으면 그 자리에서 PX4 land() 에 넘긴다.
        접지 판정은 PX4 몫이라 고도 정보 없이도 성립한다.
        """
        loop_now = asyncio.get_event_loop().time()

        if (self.descent_started_at is not None
                and loop_now - self.descent_started_at > DESCENT_TIMEOUT_SEC):
            elapsed = loop_now - self.descent_started_at
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            self.status_text = f"[P6] ⏰ 하강 지연 {elapsed:.0f}s (수동 개입 권장)"
            return

        # 한두 프레임 깜빡임은 유실이 아니다. 유예 중에는 직하강만 유지한다.
        if self.aruco_center_error is None:
            if self.marker_lost_since is None:
                self.marker_lost_since = loop_now
            lost_sec = loop_now - self.marker_lost_since

            if lost_sec >= MARKER_GRACE_SEC:
                self.status_text = "[P6] 🛬 마커 유실 -> land() 위임"
                await self._handoff_to_land()
                return

            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, DESCENT_SPEED, 0.0))
            self.status_text = f"[P6] ⬇️ 직하강 (마커 깜빡임 {lost_sec:.1f}s)"
            return

        self.marker_lost_since = None

        v_x, v_y, yaw_rate = self.compute_servo_velocity(
            yaw_scale=DESCENT_YAW_SCALE)
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(v_x, v_y, DESCENT_SPEED, yaw_rate))

        elapsed = (loop_now - self.descent_started_at
                   if self.descent_started_at is not None else 0.0)
        mark = "✅" if self.aruco_aligned else "🎯"
        self.status_text = (f"[P6] ⬇️{mark} 하강 {elapsed:.0f}s | "
                            f"vx{v_x:+.2f} vy{v_y:+.2f} yaw{yaw_rate:+.0f}")

    # ==================================================================
    # P7 : 접지 대기
    # ==================================================================
    async def handle_phase_7_landed(self):
        """PX4 가 접지와 디스암을 처리한다. 실제 접지를 확인하고 넘어간다."""
        if self.in_air:
            self.status_text = "[P7] 🅿️ PX4 착륙 진행 중..."
            self.landed_at = None
            return

        loop_now = asyncio.get_event_loop().time()
        if self.landed_at is None:
            self.landed_at = loop_now
            return

        # 접지 직후 기체가 흔들릴 수 있어 잠시 안정화를 기다린다.
        settle = loop_now - self.landed_at
        if settle < LAND_SETTLE_SEC:
            self.status_text = f"[P7] 🅿️ 접지 확인, 안정화 {settle:.1f}s"
            return

        self.status_text = "[P7->8] 🅿️ 버티포트 시퀀스 시작"
        self.current_phase = FlightPhase.PHASE_8_VERTIPORT

    # ==================================================================
    # P8 : 리프트로 새 상자 체결
    # ==================================================================
    async def handle_phase_8_vertiport(self):
        """
        그리퍼는 P3 이후 계속 열려 있다.
        리프트가 상자를 올리면 닫아서 물고, 리프트를 내린다.
        """
        if self.vertiport_done:
            self.status_text = "[P8] ✅ 상자 체결 완료. 임무 종료"
            return
        self.vertiport_done = True

        # 접지 직후 기체와 프레임이 안정될 때까지 대기
        for i in range(int(P8_DELAY_SEC * 10), 0, -1):
            self.status_text = f"[P8] ⏱️ {i/10:.1f}s 후 시작"
            await asyncio.sleep(0.1)

        self.status_text = "[P8] ⬆️ 리프트 상승 요청"
        # 응답을 놓쳐도 리프트가 실제로는 올라간 경우가 있다.
        # 그리퍼 체결은 그대로 진행하고, 실패는 로그로만 남긴다.
        if not await asyncio.to_thread(esp, "/liftup"):
            print("\n[P8] ⚠️ 리프트 응답 없음. 그리퍼 체결은 계속 진행")
        await asyncio.sleep(1.0)   # 리프트 안착 여유

        self.status_text = "[P8] 🤖 그리퍼 체결 중..."
        await asyncio.to_thread(
            self._set_servo_angle_sync, GRIPPER_CLOSE, True)
        await asyncio.sleep(GRIPPER_SETTLE_SEC)

        self.status_text = "[P8] ⬇️ 리프트 하강 요청"
        if not await asyncio.to_thread(esp, "/liftdown"):
            self.status_text = "[P8] ⚠️ 하강 요청 실패 (25초 후 자동 하강)"
            return

        self.status_text = "[P8] ✅ 상자 체결 완료. 임무 종료"

    # ==================================================================
    async def _handoff_to_land(self):
        """offboard 를 끊고 PX4 land() 에 넘긴다."""
        if self.land_commanded:
            self.current_phase = FlightPhase.PHASE_7_LANDED
            return
        self.land_commanded = True
        try:
            await self.drone.offboard.stop()
        except Exception:
            pass
        try:
            await self.drone.action.land()
        except Exception:
            self.status_text = "[P6] ❌ land() 실패! 수동 개입"
        self.current_phase = FlightPhase.PHASE_7_LANDED
        self.hold_engaged = False

    async def _fallback_to_rtl(self):
        """
        정렬 실패 시 RTL 착륙에 넘기는 폴백. 현재는 호출하지 않는다.
        
        """
        try:
            await self.drone.offboard.stop()
        except Exception:
            pass
        try:
            await self.drone.action.return_to_launch()
        except Exception:
            pass
        self.current_phase = FlightPhase.PHASE_4_RTL
        self.hold_engaged = False
        self.servo_abandoned = True

    # ==================================================================
    async def subscribe_altitude(self):
        """지면 기준 고도를 갱신한다. 없어도 FSM 은 동작한다."""
        async for position in self.drone.telemetry.position():
            self.altitude_m = position.relative_altitude_m

    async def subscribe_in_air(self):
        """공중/지상 상태. P7 접지 판정에 쓴다."""
        async for in_air in self.drone.telemetry.in_air():
            self.in_air = in_air

    def _set_servo_angle_sync(self, angle, quiet=False):
        if not quiet:
            print(f"👉 [그리퍼] {angle}도 이동 중")
        pulse_width = 0.0005 + (0.002 * angle / 180.0)
        end_time = time.time() + 1.5
        while time.time() < end_time:
            GPIO.output(self.SERVO_PIN, GPIO.HIGH)
            time.sleep(pulse_width)
            GPIO.output(self.SERVO_PIN, GPIO.LOW)
            time.sleep(0.02 - pulse_width)


async def main():
    drone = System()
    print("[SYSTEM] 픽스호크 연결 중...")
    await drone.connect(system_address="udpin://127.0.0.1:14551")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[SYSTEM] 연결 완료!")
            break
    fsm = FlightFSM(drone)
    await asyncio.to_thread(fsm._set_servo_angle_sync, GRIPPER_CLOSE)
    asyncio.create_task(fsm.subscribe_altitude())
    asyncio.create_task(fsm.subscribe_in_air())
    await fsm.run_fsm()


if __name__ == "__main__":
    asyncio.run(main())
