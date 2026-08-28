"""
USB UVC 카메라 열기 헬퍼 (ELP-USBGS1200P01 / AR0234 글로벌 셔터).

카메라 확인
    v4l2-ctl --list-devices
    v4l2-ctl -d /dev/video0 --list-formats-ext
    v4l2-ctl -d /dev/video0 --list-ctrls
"""
from __future__ import annotations

import subprocess
import sys

import cv2

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1200
DEFAULT_FPS = 30


def open_usb_camera(
    device: int | str = 0,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    use_mjpg: bool = True,
    buffer_size: int = 1,
    warmup_frames: int = 15,
) -> cv2.VideoCapture | None:
    """
    UVC 카메라를 열고 설정한다. 실패하면 None return.

    device : 0 같은 인덱스, 또는 "/dev/video0" 경로
    use_mjpg : USB 2.0 에서 고해상도/고fps 를 쓰려면 반드시 True
    buffer_size : 1 이면 제어 지연 최소화를 위해 항상 최신 프레임만 사용
    """
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        return None

    if use_mjpg:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    # 내부 버퍼를 1장으로 제한해 오래된 프레임이 쌓이지 않도록 함
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    except Exception:
        pass

    # 초기 프레임 몇 장은 노출이 안정되지 않아 폐기
    for _ in range(warmup_frames):
        capture.read()

    return capture


def describe_capture(capture: cv2.VideoCapture) -> str:
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    return f"{w}x{h} @ {fps:.0f}fps  fourcc={fourcc}"


def set_manual_exposure(
    device_path: str = "/dev/video0",
    exposure: int = 100,
    gain: int | None = None,
) -> bool:
    """
    자동 노출을 끄고 고정 노출로 전환

    수동 확인:
        v4l2-ctl -d /dev/video0 --list-ctrls
        v4l2-ctl -d /dev/video0 -c auto_exposure=1
        v4l2-ctl -d /dev/video0 -c exposure_time_absolute=100
    """
    commands = [
        # UVC 규격: 1 = manual, 3 = aperture priority(자동)
        ["v4l2-ctl", "-d", device_path, "-c", "auto_exposure=1"],
        ["v4l2-ctl", "-d", device_path, "-c",
         f"exposure_time_absolute={exposure}"],
    ]
    if gain is not None:
        commands.append(["v4l2-ctl", "-d", device_path, "-c", f"gain={gain}"])

    ok = True
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, timeout=5)
            if result.returncode != 0:
                print(f"[카메라] 설정 실패: {' '.join(command)}", file=sys.stderr)
                ok = False
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            print(f"[카메라] v4l2-ctl 실행 불가: {exc}", file=sys.stderr)
            return False
    return ok


def set_auto_exposure(device_path: str = "/dev/video0") -> bool:
    """자동 노출로 복귀."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "-c", "auto_exposure=3"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
