#!/usr/bin/env python3
"""Reproducibly export and validate the YOLO26n 320 classic-head ONNX asset."""

from __future__ import annotations

import sys

import export_yolo26s_person_640 as exporter


exporter.MODEL_VERSION = "yolo26n-person-coco-320-v1.0.0"
exporter.MODEL_VARIANT = "n"
exporter.MODEL_NAME = "yolo26n-person-coco"
exporter.MODEL_DISPLAY_NAME = "Ultralytics YOLO26n COCO"
exporter.CHECKPOINT_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
)
exporter.CHECKPOINT_BYTES = 5_544_453
exporter.CHECKPOINT_SHA256 = (
    "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
)
exporter.INPUT_SHAPE = (1, 3, 320, 320)
exporter.OUTPUT_SHAPE = (1, 84, 2100)
exporter.INPUT_SIZE = 320
exporter.EXPECTED_SAMPLE_DETECTIONS = 4


if __name__ == "__main__":
    sys.exit(exporter.main())
