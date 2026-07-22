#!/usr/bin/env python3
"""Reproducibly export and validate the YOLO26s 640 classic-head ONNX asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO


MODEL_VERSION = "yolo26s-person-coco-640-v1.0.0"
ULTRALYTICS_VERSION = "8.4.34"
ULTRALYTICS_REVISION = "16665db343532a0f94d04bf2489ee0028da10346"
CHECKPOINT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"
CHECKPOINT_BYTES = 20_422_725
CHECKPOINT_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_BYTES = 137_419
SAMPLE_SHA256 = "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63"
INPUT_SHAPE = (1, 3, 640, 640)
OUTPUT_SHAPE = (1, 84, 8400)
SEED = 20_260_722


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} integrity check failed: bytes={actual_bytes}, sha256={actual_sha256}"
        )


def stable_model_bytes(model: onnx.ModelProto) -> bytes:
    """Remove time-varying exporter metadata and serialize deterministically."""
    metadata = {item.key: item.value for item in model.metadata_props if item.key != "date"}
    metadata.update(
        {
            "description": "Ultralytics YOLO26s COCO classic-head model for person-only decoding",
            "source": CHECKPOINT_URL,
            "source_revision": ULTRALYTICS_REVISION,
            "export_contract": "images-fp32-nchw-1x3x640x640_to_output0-fp32-1x84x8400",
            "end2end": "False",
        }
    )
    del model.metadata_props[:]
    for key in sorted(metadata):
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = metadata[key]
    return model.SerializeToString(deterministic=True)


def tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)


def validate_graph(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    if [(item.domain, item.version) for item in model.opset_import] != [("", 11)]:
        raise RuntimeError("Expected only default-domain opset 11")
    if len(model.graph.input) != 1 or model.graph.input[0].name != "images":
        raise RuntimeError("Expected a single input named images")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "output0":
        raise RuntimeError("Expected a single output named output0")
    if tensor_shape(model.graph.input[0]) != INPUT_SHAPE:
        raise RuntimeError(f"Unexpected input shape: {tensor_shape(model.graph.input[0])}")
    if tensor_shape(model.graph.output[0]) != OUTPUT_SHAPE:
        raise RuntimeError(f"Unexpected output shape: {tensor_shape(model.graph.output[0])}")
    if model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("Expected FP32 input")
    if model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("Expected FP32 output")
    if any(initializer.external_data for initializer in model.graph.initializer):
        raise RuntimeError("External tensor data is not allowed")
    return {
        "onnxChecker": "passed/full_check",
        "irVersion": model.ir_version,
        "opset": 11,
        "operators": sorted({node.op_type for node in model.graph.node}),
        "externalDataTensors": 0,
    }


def export_model(checkpoint: Path, output_path: Path) -> None:
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"This release must use ultralytics=={ULTRALYTICS_VERSION}; found {ultralytics.__version__}"
        )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    yolo = YOLO(str(checkpoint))
    exported = Path(
        yolo.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            half=False,
            int8=False,
            simplify=False,
            opset=11,
            nms=False,
            end2end=False,
            device="cpu",
            verbose=True,
        )
    )
    model = onnx.load(str(exported), load_external_data=True)
    stable_bytes = stable_model_bytes(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".onnx.tmp")
    temporary.write_bytes(stable_bytes)
    os.replace(temporary, output_path)


def validate_numerics(checkpoint: Path, onnx_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    input_tensor = rng.random(INPUT_SHAPE, dtype=np.float32)
    yolo = YOLO(str(checkpoint))
    yolo.model.end2end = False
    yolo.model.eval()
    with torch.inference_mode():
        torch_output = yolo.model(torch.from_numpy(input_tensor))
        if isinstance(torch_output, (tuple, list)):
            torch_output = torch_output[0]
        torch_array = torch_output.detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_array = session.run(["output0"], {"images": input_tensor})[0]
    if tuple(torch_array.shape) != OUTPUT_SHAPE or tuple(ort_array.shape) != OUTPUT_SHAPE:
        raise RuntimeError(f"Numeric validation produced invalid shapes: {torch_array.shape}, {ort_array.shape}")
    absolute_error = np.abs(torch_array - ort_array)
    box_error = absolute_error[:, :4, :]
    score_error = absolute_error[:, 4:, :]
    result = {
        "seed": SEED,
        "inputDistribution": "numpy.default_rng.random float32 [0,1]",
        "pytorchShape": list(torch_array.shape),
        "onnxRuntimeShape": list(ort_array.shape),
        "maxAbsError": float(absolute_error.max()),
        "meanAbsError": float(absolute_error.mean()),
        "boxMaxAbsError": float(box_error.max()),
        "scoreMaxAbsError": float(score_error.max()),
        "provider": session.get_providers()[0],
    }
    if result["maxAbsError"] > 0.0025 or result["meanAbsError"] > 0.00001:
        raise RuntimeError(f"PyTorch/ORT error exceeded tolerance: {result}")
    return result


def intersection_over_union(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def validate_sample(sample_path: Path, onnx_path: Path) -> dict[str, Any]:
    require_file(sample_path, SAMPLE_BYTES, SAMPLE_SHA256, "validation sample")
    image = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode sample image: {sample_path}")
    height, width = image.shape[:2]
    scale = min(640 / width, 640 / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    half_pad_x, half_pad_y = (640 - resized_width) / 2, (640 - resized_height) / 2
    left, right = round(half_pad_x - 0.1), round(half_pad_x + 0.1)
    top, bottom = round(half_pad_y - 0.1), round(half_pad_y + 0.1)
    letterboxed = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output = session.run(["output0"], {"images": tensor})[0][0]
    candidates: list[list[float]] = []
    for anchor_index in np.flatnonzero(output[4] >= 0.20):
        center_x, center_y, box_width, box_height = (float(v) for v in output[:4, anchor_index])
        candidates.append(
            [
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
                float(output[4, anchor_index]),
                int(anchor_index),
            ]
        )
    candidates.sort(key=lambda item: (-item[4], item[5]))
    kept: list[list[float]] = []
    for candidate in candidates:
        if all(intersection_over_union(candidate, previous) <= 0.50 for previous in kept):
            kept.append(candidate)
            if len(kept) == 5:
                break
    detections = []
    for box in kept:
        x1 = max(0.0, min(width, (box[0] - left) / scale))
        y1 = max(0.0, min(height, (box[1] - top) / scale))
        x2 = max(0.0, min(width, (box[2] - left) / scale))
        y2 = max(0.0, min(height, (box[3] - top) / scale))
        detections.append(
            {
                "xyxy": [round(value, 3) for value in (x1, y1, x2, y2)],
                "confidence": round(box[4], 7),
                "anchorIndex": int(box[5]),
            }
        )
    if len(detections) != 4:
        raise RuntimeError(f"Expected 4 reference person detections, found {len(detections)}")
    return {
        "imageUrl": SAMPLE_URL,
        "imageBytes": SAMPLE_BYTES,
        "imageSha256": SAMPLE_SHA256,
        "imageSize": [width, height],
        "letterboxScale": scale,
        "letterboxPadding": [left, top, right, bottom],
        "scoreThreshold": 0.20,
        "iouThreshold": 0.50,
        "candidateCount": len(candidates),
        "detections": detections,
    }


def write_release_metadata(
    output_dir: Path,
    graph_validation: dict[str, Any],
    numeric_validation: dict[str, Any],
    sample_validation: dict[str, Any],
) -> None:
    model_path = output_dir / "model.onnx"
    artifact_bytes = model_path.stat().st_size
    artifact_sha256 = sha256_file(model_path)
    manifest = {
        "formatVersion": 1,
        "modelName": "yolo26s-person-coco",
        "version": MODEL_VERSION,
        "modelFile": "model.onnx",
        "modelBytes": artifact_bytes,
        "modelSha256": artifact_sha256,
        "decoder": "yolo26-classic-person-v1",
        "inputName": "images",
        "inputShape": list(INPUT_SHAPE),
        "inputType": "float32",
        "inputLayout": "NCHW",
        "inputSize": 640,
        "resizeMode": "letterbox",
        "resizeInterpolation": "bilinear",
        "colorOrder": "RGB",
        "normalization": "zeroOne",
        "padValue": 114,
        "preprocessVersion": "yolo-rgb-zero-one-letterbox-640-bilinear-pad114-v1",
        "outputName": "output0",
        "outputShape": list(OUTPUT_SHAPE),
        "outputBoxFormat": "cxcywh",
        "classCount": 80,
        "personClassId": 0,
        "scoreThreshold": 0.20,
        "iouThreshold": 0.50,
        "maxDetections": 5,
        "opset": 11,
    }
    metadata = {
        "model": "Ultralytics YOLO26s COCO, classic detection head",
        "sourceRepository": "https://github.com/ultralytics/ultralytics",
        "sourceRevision": ULTRALYTICS_REVISION,
        "checkpointUrl": CHECKPOINT_URL,
        "checkpointBytes": CHECKPOINT_BYTES,
        "checkpointSha256": CHECKPOINT_SHA256,
        "export": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "opencv": cv2.__version__,
            "seed": SEED,
            "format": "onnx",
            "opset": 11,
            "batch": 1,
            "imgsz": 640,
            "dynamic": False,
            "half": False,
            "int8": False,
            "simplify": False,
            "nms": False,
            "end2end": False,
            "device": "cpu",
        },
        "preprocess": {
            "version": manifest["preprocessVersion"],
            "resize": "aspect-preserving fit with centered letterbox padding",
            "resizeInterpolation": "bilinear",
            "padValueRgb": [114, 114, 114],
            "colorOrder": "RGB",
            "range": [0, 1],
            "layout": "NCHW",
        },
        "postprocess": {
            "decoder": manifest["decoder"],
            "boxFormat": "cxcywh",
            "personClassId": 0,
            "scoreThreshold": 0.20,
            "iouThreshold": 0.50,
            "maxDetections": 5,
        },
        "validation": {
            "graph": graph_validation,
            "pytorchVsOnnxRuntime": numeric_validation,
            "fixedImageDecode": sample_validation,
        },
        "artifact": {
            "file": "model.onnx",
            "bytes": artifact_bytes,
            "sha256": artifact_sha256,
            "gitStorage": "ordinary Git blob; no Git LFS",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "export-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sample-image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    sample_image = args.sample_image.resolve()
    output_dir = args.output_dir.resolve()
    require_file(checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256, "official checkpoint")
    require_file(sample_image, SAMPLE_BYTES, SAMPLE_SHA256, "validation sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.onnx"
    export_model(checkpoint, model_path)
    graph_validation = validate_graph(model_path)
    numeric_validation = validate_numerics(checkpoint, model_path)
    sample_validation = validate_sample(sample_image, model_path)
    write_release_metadata(output_dir, graph_validation, numeric_validation, sample_validation)
    print(
        json.dumps(
            {
                "model": str(model_path),
                "bytes": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
                "manifest": str(output_dir / "manifest.json"),
                "metadata": str(output_dir / "export-metadata.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
