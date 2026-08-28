#!/usr/bin/env python3
"""
USB 카메라 초점 조정 도구 (ELP-USBGS1200P01).

실행
    python3 focus_tune.py            # 수동 조정
    python3 focus_tune.py --auto     # 자동 스윕 후 최적값 추천

키
    a / d : 초점 -10 / +10
    z / c : 초점 -50 / +50
    s     : 현재 값에서 자동 미세 스윕
    f     : 오토포커스 토글 (비교용)
    q     : 종료
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import cv2
import numpy as np

DEVICE = "/dev/video0"
WIDTH, HEIGHT = 1920, 1200


def v4l2_set(control: str, value: int) -> bool:
    try:
        r = subprocess.run(
            ["v4l2-ctl", "-d", DEVICE, "-c", f"{control}={value}"],
            capture_output=True, timeout=3,
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def v4l2_get(control: str):
    try:
        r = subprocess.run(
            ["v4l2-ctl", "-d", DEVICE, "-C", control],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and ":" in r.stdout:
            return int(r.stdout.split(":")[1].strip())
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    return None


def sharpness(frame_bgr) -> float:
    """라플라시안 분산. 화면 중앙 영역만 본다."""
    h, w = frame_bgr.shape[:2]
    y0, y1 = int(h * 0.3), int(h * 0.7)
    x0, x1 = int(w * 0.3), int(w * 0.7)
    gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        return None
    for _ in range(15):
        cap.read()
    return cap


def measure_at(cap, focus_value: int, settle_frames: int = 8) -> float:
    """초점 값 설정 후 선명도 측정."""
    v4l2_set("focus_absolute", focus_value)
    time.sleep(0.25)
    scores = []
    for _ in range(settle_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            scores.append(sharpness(frame))
    return max(scores) if scores else 0.0


def auto_sweep(cap, start=0, end=1023, coarse_step=64) -> int:
    """거친 스윕 -> 최고점 주변 미세 스윕 -> 최적값 반환."""
    print("\n[1단계] 거친 스윕", flush=True)
    best_value, best_score = start, 0.0
    for value in range(start, end + 1, coarse_step):
        score = measure_at(cap, value)
        bar = "#" * int(min(score / 20, 50))
        print(f"  focus {value:4d} : {score:8.1f}  {bar}", flush=True)
        if score > best_score:
            best_value, best_score = value, score

    print(f"\n[2단계] {best_value} 주변 미세 스윕", flush=True)
    lo = max(start, best_value - coarse_step)
    hi = min(end, best_value + coarse_step)
    for value in range(lo, hi + 1, 8):
        score = measure_at(cap, value)
        bar = "#" * int(min(score / 20, 50))
        marker = "  <== 최고" if score > best_score else ""
        print(f"  focus {value:4d} : {score:8.1f}  {bar}{marker}", flush=True)
        if score > best_score:
            best_value, best_score = value, score

    return best_value


def main() -> int:
    parser = argparse.ArgumentParser(description="카메라 초점 조정")
    parser.add_argument("--auto", action="store_true", help="자동 스윕 실행")
    args = parser.parse_args()

    cap = open_camera()
    if cap is None:
        print("카메라 열기 실패", file=sys.stderr)
        return 1

    print("오토포커스를 끄기.", flush=True)
    if not v4l2_set("focus_automatic_continuous", 0):
        print("경고: 오토포커스 끄기 실패.",
              file=sys.stderr)

    focus = v4l2_get("focus_absolute")
    if focus is None:
        focus = 483
    print(f"현재 초점 값: {focus}\n", flush=True)

    if args.auto:
        best = auto_sweep(cap)
        v4l2_set("focus_absolute", best)
        print("\n" + "=" * 54)
        print(f"최적 초점 값: {best}")
        print(f'  v4l2_set("focus_automatic_continuous", 0)')
        print(f'  v4l2_set("focus_absolute", {best})')
        print("=" * 54)
        focus = best

    auto_on = False
    cv2.namedWindow("focus", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("focus", 1000, 625)
    print("a/d = -10/+10,  z/c = -50/+50,  s = 미세스윕,  "
          "f = AF토글,  q = 종료\n", flush=True)

    history: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        score = sharpness(frame)
        history.append(score)
        if len(history) > 60:
            history.pop(0)

        h, w = frame.shape[:2]
        # 측정 영역 표시
        cv2.rectangle(frame,
                      (int(w * 0.3), int(h * 0.3)),
                      (int(w * 0.7), int(h * 0.7)),
                      (0, 220, 80), 2)

        peak = max(history) if history else 1.0
        lines = [
            f"focus = {focus}   AF = {'ON' if auto_on else 'off'}",
            f"sharpness = {score:.1f}   peak = {peak:.1f}",
            "higher is sharper - aim at 2-3m target",
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (24, 60 + i * 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3,
                        cv2.LINE_AA)

        # 선명도 막대
        bar_w = int(min(score / peak, 1.0) * (w - 48)) if peak > 0 else 0
        cv2.rectangle(frame, (24, h - 60), (24 + bar_w, h - 24),
                      (0, 220, 80), -1)

        cv2.imshow("focus", frame)

        key = cv2.waitKey(1) & 0xFF
        changed = None
        if key == ord("a"):
            changed = max(0, focus - 10)
        elif key == ord("d"):
            changed = min(1023, focus + 10)
        elif key == ord("z"):
            changed = max(0, focus - 50)
        elif key == ord("c"):
            changed = min(1023, focus + 50)
        elif key == ord("f"):
            auto_on = not auto_on
            v4l2_set("focus_automatic_continuous", 1 if auto_on else 0)
            print(f"오토포커스 {'ON' if auto_on else 'off'}", flush=True)
        elif key == ord("s"):
            print("\n미세 스윕...", flush=True)
            lo, hi = max(0, focus - 80), min(1023, focus + 80)
            best_v, best_s = focus, 0.0
            for value in range(lo, hi + 1, 8):
                s = measure_at(cap, value)
                if s > best_s:
                    best_v, best_s = value, s
            focus = best_v
            v4l2_set("focus_absolute", focus)
            print(f"최적 {focus} (선명도 {best_s:.1f})\n", flush=True)
            history.clear()
        elif key == ord("q"):
            break

        if changed is not None and changed != focus:
            focus = changed
            v4l2_set("focus_absolute", focus)
            history.clear()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n최종 초점 값: {focus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
