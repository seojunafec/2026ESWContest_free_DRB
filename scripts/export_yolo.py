from __future__ import annotations

import argparse
import sys


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
    }
    if args.device:
        export_kwargs["device"] = args.device
    if args.half:
        export_kwargs["half"] = True
    if args.int8:
        export_kwargs["int8"] = True

    model = YOLO(args.model)
    output = model.export(**export_kwargs)
    print(f"export complete: {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model for deployment.")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "engine", "openvino", "torchscript"],
        help="Use engine on Jetson for TensorRT export.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Example: 0 for Jetson GPU")
    parser.add_argument("--half", action="store_true", help="FP16 export where supported.")
    parser.add_argument("--int8", action="store_true", help="INT8 export where supported.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
