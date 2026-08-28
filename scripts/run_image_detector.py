from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from medifly_vision import CameraModel, CasualtyDetector


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    frame = read_image(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}", file=sys.stderr)
        return 1

    height, width = frame.shape[:2]
    camera = CameraModel(
        width_px=width,
        height_px=height,
        horizontal_fov_deg=args.hfov_deg,
        vertical_fov_deg=args.vfov_deg,
    )
    detector = CasualtyDetector(
        model_path=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
        tile_overlap=args.tile_overlap,
    )
    detect_method = detector.detect_tiled if args.tiled else detector.detect
    detect_kwargs = {
        "frame_bgr": frame,
        "altitude_m": args.height_m,
        "camera": camera,
        "drop_distance_m": args.drop_distance_m,
        "drop_direction": args.drop_direction,
    }
    if args.tiled:
        detect_kwargs["iou_threshold"] = args.tile_iou_threshold
    detections = detect_method(**detect_kwargs)

    for index, detection in enumerate(detections, start=1):
        print(
            f"{index}: {detection.class_name} conf={detection.confidence:.3f} "
            f"center_px=({detection.center_px[0]:.1f}, {detection.center_px[1]:.1f}) "
            f"person_m=({detection.ground_m[0]:+.2f}, {detection.ground_m[1]:+.2f}) "
            f"drop_m=({detection.drop.drop_ground_m[0]:+.2f}, "
            f"{detection.drop.drop_ground_m[1]:+.2f})"
        )

    draw_overlay(frame, detections)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(output_path, frame)
    print(f"saved: {output_path}")
    return 0


def draw_overlay(frame: np.ndarray, detections) -> None:
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        center_x, center_y = [int(round(value)) for value in detection.center_px]
        drop_x, drop_y = [int(round(value)) for value in detection.drop.drop_point_px]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 220, 80), -1)
        cv2.circle(frame, (drop_x, drop_y), 7, (0, 80, 255), -1)
        cv2.line(frame, (center_x, center_y), (drop_x, drop_y), (0, 80, 255), 2)
        cv2.putText(
            frame,
            f"{detection.class_name} {detection.confidence:.2f} "
            f"[{detection.inference_source}]",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MEDIFLY person_down detection on one image."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/predictions/image_prediction.jpg")
    parser.add_argument("--height-m", type=float, default=0.8)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default=None, help="Example: cpu, cuda, 0")
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Run full-frame plus complete 2x2 tile validation.",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.15)
    parser.add_argument("--tile-iou-threshold", type=float, default=0.45)
    parser.add_argument("--hfov-deg", type=float, default=70.0)
    parser.add_argument("--vfov-deg", type=float, default=43.0)
    parser.add_argument("--drop-distance-m", type=float, default=1.5)
    parser.add_argument(
        "--drop-direction",
        choices=["image_up", "image_down", "image_left", "image_right"],
        default="image_down",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
