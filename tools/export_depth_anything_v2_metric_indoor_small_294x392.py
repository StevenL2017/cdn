#!/usr/bin/env python3
"""Reproducibly export and validate the 294x392 DAv2 metric indoor ONNX asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision


MODEL_DISPLAY_NAME = "Depth Anything V2 Metric Indoor Small (Hypersim ViT-S)"
MODEL_NAME = "depth-anything-v2-metric-indoor-small"
MODEL_VERSION = "depth-anything-v2-metric-indoor-small-294x392-v1.1.0"
SOURCE_REPOSITORY = "https://github.com/DepthAnything/Depth-Anything-V2"
SOURCE_REVISION = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
CHECKPOINT_URL = (
    "https://huggingface.co/depth-anything/"
    "Depth-Anything-V2-Metric-Hypersim-Small/resolve/"
    "3bc65d4e14a6786a61acec16453c50e12bf5f338/"
    "depth_anything_v2_metric_hypersim_vits.pth"
)
CHECKPOINT_BYTES = 99_222_290
CHECKPOINT_SHA256 = "b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545"
INPUT_NAME = "image"
OUTPUT_NAME = "depth"
INPUT_SHAPE = (1, 3, 294, 392)
OUTPUT_SHAPE = (1, 1, 294, 392)
PREPROCESS_VERSION = "depth-anything-v2-metric-imagenet-letterbox-294x392-v1"
FIXTURE_RELATIVE_PATH = "assets/examples/demo01.jpg"
FIXTURE_BYTES = 488_150
FIXTURE_SHA256 = "35ef1bbb63f6540e49aa9b6302b9b938be4fe8b9c08c07c3694b02396b0e87e0"
FOLDED_POSITION_SHAPE = (1, 384, 21, 28)
MAX_GIT_BLOB_BYTES = 104_857_600
OPSET = 11
SEED = 20_260_723
MAX_ABS_ERROR_TOLERANCE_METERS = 0.0005
MEAN_ABS_ERROR_TOLERANCE_METERS = 0.00001

EXPECTED_VERSIONS = {
    "python": "3.13.5",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "onnx": "1.21.0",
    "onnxruntime": "1.24.4",
    "numpy": "2.2.6",
    "opencv": "4.12.0",
}


def base_version(value: str) -> str:
    return value.split("+", 1)[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_environment() -> None:
    actual = {
        "python": platform.python_version(),
        "torch": base_version(torch.__version__),
        "torchvision": base_version(torchvision.__version__),
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    mismatches = {
        name: {"expected": EXPECTED_VERSIONS[name], "actual": value}
        for name, value in actual.items()
        if value != EXPECTED_VERSIONS[name]
    }
    if mismatches:
        raise RuntimeError(f"Pinned exporter dependency mismatch: {mismatches}")


def require_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Official checkpoint not found: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != CHECKPOINT_BYTES or actual_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(
            "Official checkpoint integrity check failed: "
            f"bytes={actual_bytes}, sha256={actual_sha256}"
        )


def git_output(source_dir: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_dir), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def require_official_source(source_dir: Path) -> None:
    if not (source_dir / ".git").exists():
        raise RuntimeError(f"Source directory is not a Git checkout: {source_dir}")
    revision = git_output(source_dir, "rev-parse", "HEAD")
    if revision != SOURCE_REVISION:
        raise RuntimeError(
            f"Depth Anything V2 revision mismatch: expected {SOURCE_REVISION}, got {revision}"
        )
    tracked_changes = git_output(
        source_dir,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_changes:
        raise RuntimeError(
            "Official source checkout has tracked modifications; refusing a non-reproducible export:\n"
            f"{tracked_changes}"
        )
    required = [
        source_dir / "LICENSE",
        source_dir / FIXTURE_RELATIVE_PATH,
        source_dir / "metric_depth" / "depth_anything_v2" / "dpt.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Official source checkout is incomplete: {missing}")


def load_official_model(source_dir: Path, checkpoint: Path) -> nn.Module:
    metric_source = source_dir / "metric_depth"
    sys.path.insert(0, str(metric_source))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    finally:
        sys.path.pop(0)

    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        max_depth=20.0,
    )
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class StaticMetricDepthContract(nn.Module):
    """Expose an explicit NCHW single-channel output for the mini program."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image).unsqueeze(1)


def stabilize_onnx(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=True)
    model.producer_name = "Depth Anything V2 metric indoor static exporter"
    model.producer_version = "1.1.0"
    del model.metadata_props[:]
    metadata = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "export_contract": "image-fp32-nchw-1x3x294x392_to_depth-fp32-1x1x294x392-meter",
        "source": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
    }
    for key in sorted(metadata):
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = metadata[key]
    temporary = path.with_suffix(".stable.tmp")
    temporary.write_bytes(model.SerializeToString(deterministic=True))
    os.replace(temporary, path)


def ort_session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def resize_mode(node: onnx.NodeProto) -> str:
    for attribute in node.attribute:
        if attribute.name == "mode":
            value = onnx.helper.get_attribute_value(attribute)
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return "nearest"


def evaluate_constant_resize(
    model: onnx.ModelProto,
    resize: onnx.NodeProto,
    producers: dict[str, onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
) -> np.ndarray:
    dependency_nodes: list[onnx.NodeProto] = []
    dependency_initializers: list[onnx.TensorProto] = []
    for input_name in resize.input:
        if not input_name:
            continue
        if input_name in producers:
            producer = producers[input_name]
            if producer.op_type != "Constant":
                raise RuntimeError(
                    f"Static cubic Resize depends on non-constant node {producer.name}"
                )
            dependency_nodes.append(producer)
        elif input_name in initializers:
            dependency_initializers.append(initializers[input_name])
        else:
            raise RuntimeError(
                f"Static cubic Resize input is not constant: {input_name}"
            )

    output_info = onnx.helper.make_tensor_value_info(
        resize.output[0],
        onnx.TensorProto.FLOAT,
        FOLDED_POSITION_SHAPE,
    )
    graph = onnx.helper.make_graph(
        [*dependency_nodes, resize],
        "fold_static_position_embedding_resize",
        [],
        [output_info],
        dependency_initializers,
    )
    mini_model = onnx.helper.make_model(
        graph,
        producer_name="DAv2 static Resize evaluator",
        opset_imports=[
            onnx.helper.make_opsetid(item.domain, item.version)
            for item in model.opset_import
        ],
    )
    mini_model.ir_version = model.ir_version
    onnx.checker.check_model(mini_model, full_check=True)
    session = ort.InferenceSession(
        mini_model.SerializeToString(deterministic=True),
        sess_options=ort_session_options(),
        providers=["CPUExecutionProvider"],
    )
    value = session.run([resize.output[0]], {})[0]
    if (
        value.dtype != np.float32
        or tuple(value.shape) != FOLDED_POSITION_SHAPE
        or not np.isfinite(value).all()
    ):
        raise RuntimeError("Folded position embedding is not finite FP32")
    return value


def compare_ort_sessions(
    before: ort.InferenceSession,
    optimized_path: Path,
    source_dir: Path,
) -> dict[str, Any]:
    after = ort.InferenceSession(
        str(optimized_path),
        sess_options=ort_session_options(),
        providers=["CPUExecutionProvider"],
    )
    sample_results: list[dict[str, Any]] = []
    aggregate_max = 0.0
    aggregate_sum = 0.0
    aggregate_values = 0
    for name, input_array in validation_inputs(source_dir):
        before_output = before.run([OUTPUT_NAME], {INPUT_NAME: input_array})[0]
        after_output = after.run([OUTPUT_NAME], {INPUT_NAME: input_array})[0]
        absolute_error = np.abs(before_output - after_output)
        sample_max = float(absolute_error.max())
        sample_mean = float(absolute_error.mean())
        aggregate_max = max(aggregate_max, sample_max)
        aggregate_sum += float(absolute_error.sum(dtype=np.float64))
        aggregate_values += absolute_error.size
        sample_results.append(
            {
                "name": name,
                "maxAbsErrorMeters": sample_max,
                "meanAbsErrorMeters": sample_mean,
            }
        )
    result = {
        "samples": sample_results,
        "maxAbsErrorMeters": aggregate_max,
        "meanAbsErrorMeters": aggregate_sum / aggregate_values,
    }
    if result["maxAbsErrorMeters"] != 0 or result["meanAbsErrorMeters"] != 0:
        raise RuntimeError(f"Static graph optimization changed ORT output: {result}")
    return result


def optimize_static_graph(path: Path, source_dir: Path) -> dict[str, Any]:
    before_session = ort.InferenceSession(
        str(path),
        sess_options=ort_session_options(),
        providers=["CPUExecutionProvider"],
    )
    model = onnx.load(str(path), load_external_data=True)
    before_nodes = len(model.graph.node)
    before_resize = sum(node.op_type == "Resize" for node in model.graph.node)
    before_identity = sum(node.op_type == "Identity" for node in model.graph.node)
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    initializers = {item.name: item for item in model.graph.initializer}
    cubic_resizes = [
        node
        for node in model.graph.node
        if node.op_type == "Resize" and resize_mode(node) == "cubic"
    ]
    identities = [node for node in model.graph.node if node.op_type == "Identity"]
    if len(cubic_resizes) != 1 or len(identities) != 1:
        raise RuntimeError(
            "Expected exactly one fixed cubic Resize and one initializer Identity "
            f"before hardening; got cubic={len(cubic_resizes)}, identity={len(identities)}"
        )

    resize = cubic_resizes[0]
    resize_value = evaluate_constant_resize(model, resize, producers, initializers)
    input_use_count = {
        name: sum(name in node.input for node in model.graph.node)
        for name in resize.input
        if name
    }
    removable_constant_nodes: list[onnx.NodeProto] = []
    for input_name in resize.input[1:]:
        producer = producers.get(input_name)
        if producer is not None:
            if producer.op_type != "Constant" or input_use_count[input_name] != 1:
                raise RuntimeError(
                    f"Resize dependency cannot be safely removed: {producer.name}"
                )
            removable_constant_nodes.append(producer)
    data_initializer_name = resize.input[0]
    if data_initializer_name not in initializers or input_use_count[data_initializer_name] != 1:
        raise RuntimeError(
            f"Resize data initializer cannot be safely replaced: {data_initializer_name}"
        )

    identity = identities[0]
    if len(identity.input) != 1 or len(identity.output) != 1:
        raise RuntimeError(f"Unexpected Identity contract: {identity.name}")
    identity_input = identity.input[0]
    identity_output = identity.output[0]
    if identity_input not in initializers:
        raise RuntimeError(
            f"Identity is not an initializer alias and cannot be removed: {identity.name}"
        )
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == identity_output:
                node.input[index] = identity_input
    for graph_output in model.graph.output:
        if graph_output.name == identity_output:
            graph_output.name = identity_input

    removed_node_names = {
        node.name for node in [resize, identity, *removable_constant_nodes]
    }
    if len(removed_node_names) != 4 or any(not name for name in removed_node_names):
        raise RuntimeError(
            f"Expected four uniquely named nodes to remove, got {removed_node_names}"
        )
    remaining_nodes = [
        node for node in model.graph.node if node.name not in removed_node_names
    ]
    del model.graph.node[:]
    model.graph.node.extend(remaining_nodes)
    remaining_initializers = [
        item
        for item in model.graph.initializer
        if item.name != data_initializer_name
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(remaining_initializers)
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(resize_value, name=resize.output[0])
    )

    onnx.checker.check_model(model, full_check=True)
    temporary = path.with_suffix(".optimized.tmp")
    temporary.write_bytes(model.SerializeToString(deterministic=True))
    os.replace(temporary, path)
    parity = compare_ort_sessions(before_session, path, source_dir)
    after_model = onnx.load(str(path), load_external_data=False)
    after_cubic = sum(
        node.op_type == "Resize" and resize_mode(node) == "cubic"
        for node in after_model.graph.node
    )
    after_identity = sum(node.op_type == "Identity" for node in after_model.graph.node)
    if after_cubic != 0 or after_identity != 0:
        raise RuntimeError(
            f"Graph hardening incomplete: cubic={after_cubic}, identity={after_identity}"
        )
    if (
        before_nodes != 897
        or len(after_model.graph.node) != 893
        or before_resize != 6
        or sum(node.op_type == "Resize" for node in after_model.graph.node) != 5
        or before_identity != 1
    ):
        raise RuntimeError(
            "Unexpected static graph hardening counts: "
            f"nodes {before_nodes}->{len(after_model.graph.node)}, "
            f"Resize {before_resize}->"
            f"{sum(node.op_type == 'Resize' for node in after_model.graph.node)}, "
            f"Identity {before_identity}->{after_identity}"
        )
    return {
        "name": "fixed-position-embedding-resize-and-initializer-identity-fold-v1",
        "beforeNodeCount": before_nodes,
        "afterNodeCount": len(after_model.graph.node),
        "beforeResizeCount": before_resize,
        "afterResizeCount": sum(
            node.op_type == "Resize" for node in after_model.graph.node
        ),
        "beforeIdentityCount": before_identity,
        "afterIdentityCount": after_identity,
        "foldedResizeOutput": resize.output[0],
        "foldedResizeShape": list(resize_value.shape),
        "rawOnnxRuntimeVsHardenedOnnxRuntime": parity,
    }


def export_model(
    contract: nn.Module,
    path: Path,
    source_dir: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".export.tmp.onnx")
    if temporary.exists():
        temporary.unlink()
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            contract,
            dummy,
            str(temporary),
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes=None,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False,
        )
    os.replace(temporary, path)
    optimization = optimize_static_graph(path, source_dir)
    stabilize_onnx(path)
    return optimization


def tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)


def validate_graph(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    opsets = [(item.domain, item.version) for item in model.opset_import]
    if opsets != [("", OPSET)]:
        raise RuntimeError(f"Expected only default-domain opset {OPSET}, got {opsets}")
    if len(model.graph.input) != 1 or model.graph.input[0].name != INPUT_NAME:
        raise RuntimeError(f"Expected one input named {INPUT_NAME}")
    if len(model.graph.output) != 1 or model.graph.output[0].name != OUTPUT_NAME:
        raise RuntimeError(f"Expected one output named {OUTPUT_NAME}")
    if tensor_shape(model.graph.input[0]) != INPUT_SHAPE:
        raise RuntimeError(f"Unexpected input shape: {tensor_shape(model.graph.input[0])}")
    if tensor_shape(model.graph.output[0]) != OUTPUT_SHAPE:
        raise RuntimeError(f"Unexpected output shape: {tensor_shape(model.graph.output[0])}")
    if model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("Expected an FP32 image input")
    if model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("Expected an FP32 depth output")
    external_tensors = [
        initializer.name
        for initializer in model.graph.initializer
        if initializer.external_data
    ]
    if external_tensors:
        raise RuntimeError(f"External tensor data is not allowed: {external_tensors[:3]}")
    cubic_resizes = [
        node.name
        for node in model.graph.node
        if node.op_type == "Resize" and resize_mode(node) == "cubic"
    ]
    identity_nodes = [node.name for node in model.graph.node if node.op_type == "Identity"]
    if cubic_resizes or identity_nodes:
        raise RuntimeError(
            f"Mobile compatibility hardening is incomplete: "
            f"cubic Resize={cubic_resizes}, Identity={identity_nodes}"
        )
    return {
        "onnxChecker": "passed/full_check",
        "irVersion": model.ir_version,
        "opset": OPSET,
        "nodeCount": len(model.graph.node),
        "initializerCount": len(model.graph.initializer),
        "operators": sorted({node.op_type for node in model.graph.node}),
        "cubicResizeNodes": 0,
        "identityNodes": 0,
        "externalDataTensors": 0,
    }


def javascript_round_positive(value: float) -> int:
    return int(np.floor(value + 0.5))


def fixed_image_input(source_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    fixture_path = source_dir / FIXTURE_RELATIVE_PATH
    actual_bytes = fixture_path.stat().st_size
    actual_sha256 = sha256_file(fixture_path)
    if actual_bytes != FIXTURE_BYTES or actual_sha256 != FIXTURE_SHA256:
        raise RuntimeError(
            "Official fixed-image fixture integrity check failed: "
            f"bytes={actual_bytes}, sha256={actual_sha256}"
        )
    bgr = cv2.imread(str(fixture_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not decode official fixture: {fixture_path}")
    source_height, source_width = bgr.shape[:2]
    input_height, input_width = INPUT_SHAPE[2], INPUT_SHAPE[3]
    scale = min(input_width / source_width, input_height / source_height)
    resized_width = max(1, min(input_width, javascript_round_positive(source_width * scale)))
    resized_height = max(1, min(input_height, javascript_round_positive(source_height * scale)))
    pad_left = (input_width - resized_width) // 2
    pad_top = (input_height - resized_height) // 2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0
    normalized = (
        resized
        - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = np.zeros(INPUT_SHAPE, dtype=np.float32)
    tensor[
        0,
        :,
        pad_top:pad_top + resized_height,
        pad_left:pad_left + resized_width,
    ] = normalized.transpose(2, 0, 1)
    fixture = {
        "sourceRepository": SOURCE_REPOSITORY,
        "sourceRevision": SOURCE_REVISION,
        "relativePath": FIXTURE_RELATIVE_PATH,
        "bytes": FIXTURE_BYTES,
        "sha256": FIXTURE_SHA256,
        "sourceSize": [source_width, source_height],
        "resizeInterpolation": "OpenCV INTER_LINEAR with pinned OpenCV",
        "resizedSize": [resized_width, resized_height],
        "paddingLeftTopRightBottom": [
            pad_left,
            pad_top,
            input_width - resized_width - pad_left,
            input_height - resized_height - pad_top,
        ],
        "inputTensorSha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
    }
    return tensor, fixture


def validation_inputs(source_dir: Path) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    random_input = rng.normal(0.0, 1.0, INPUT_SHAPE).astype(np.float32)
    y = np.linspace(-2.0, 2.0, INPUT_SHAPE[2], dtype=np.float32)[None, None, :, None]
    x = np.linspace(-1.5, 1.5, INPUT_SHAPE[3], dtype=np.float32)[None, None, None, :]
    channel = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)[None, :, None, None]
    structured_input = np.broadcast_to(y + x + channel, INPUT_SHAPE).copy()
    fixture_input, _ = fixed_image_input(source_dir)
    return [
        ("seeded-normal", random_input),
        ("structured-gradient", structured_input),
        ("official-demo01-letterbox", fixture_input),
    ]


def validate_numerics(
    contract: nn.Module,
    onnx_path: Path,
    source_dir: Path,
) -> dict[str, Any]:
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=ort_session_options(),
        providers=["CPUExecutionProvider"],
    )
    samples: list[dict[str, Any]] = []
    aggregate_max = 0.0
    aggregate_sum = 0.0
    aggregate_values = 0

    for name, input_array in validation_inputs(source_dir):
        with torch.inference_mode():
            torch_array = contract(torch.from_numpy(input_array)).cpu().numpy()
        ort_array = session.run([OUTPUT_NAME], {INPUT_NAME: input_array})[0]
        if tuple(torch_array.shape) != OUTPUT_SHAPE or tuple(ort_array.shape) != OUTPUT_SHAPE:
            raise RuntimeError(
                f"Numeric validation shape mismatch: torch={torch_array.shape}, ort={ort_array.shape}"
            )
        if not np.isfinite(torch_array).all() or not np.isfinite(ort_array).all():
            raise RuntimeError(f"Numeric validation produced non-finite output for {name}")
        if float(torch_array.min()) <= 0 or float(torch_array.max()) > 20.0:
            raise RuntimeError(
                f"PyTorch metric output is outside (0,20]m for {name}: "
                f"{torch_array.min()}..{torch_array.max()}"
            )
        absolute_error = np.abs(torch_array - ort_array)
        sample_max = float(absolute_error.max())
        sample_mean = float(absolute_error.mean())
        aggregate_max = max(aggregate_max, sample_max)
        aggregate_sum += float(absolute_error.sum(dtype=np.float64))
        aggregate_values += absolute_error.size
        samples.append(
            {
                "name": name,
                "pytorchShape": list(torch_array.shape),
                "onnxRuntimeShape": list(ort_array.shape),
                "pytorchRangeMeters": [
                    float(torch_array.min()),
                    float(torch_array.max()),
                ],
                "maxAbsErrorMeters": sample_max,
                "meanAbsErrorMeters": sample_mean,
            }
        )

    aggregate_mean = aggregate_sum / aggregate_values
    result = {
        "seed": SEED,
        "provider": session.get_providers()[0],
        "samples": samples,
        "maxAbsErrorMeters": aggregate_max,
        "meanAbsErrorMeters": aggregate_mean,
        "maxAbsToleranceMeters": MAX_ABS_ERROR_TOLERANCE_METERS,
        "meanAbsToleranceMeters": MEAN_ABS_ERROR_TOLERANCE_METERS,
    }
    if (
        aggregate_max > MAX_ABS_ERROR_TOLERANCE_METERS
        or aggregate_mean > MEAN_ABS_ERROR_TOLERANCE_METERS
    ):
        raise RuntimeError(f"PyTorch/ORT error exceeded tolerance: {result}")
    return result


def verify_deterministic_reexport(
    contract: nn.Module,
    output_dir: Path,
    published_model: Path,
    source_dir: Path,
) -> dict[str, Any]:
    second_path = output_dir / "model.reexport.onnx"
    try:
        second_optimization = export_model(contract, second_path, source_dir)
        first_sha256 = sha256_file(published_model)
        second_sha256 = sha256_file(second_path)
        if first_sha256 != second_sha256:
            raise RuntimeError(
                "Two exports from the pinned environment produced different hashes: "
                f"{first_sha256} != {second_sha256}"
            )
        return {
            "passed": True,
            "exportsCompared": 2,
            "sha256": first_sha256,
            "secondExportHardenedNodeCount": second_optimization["afterNodeCount"],
        }
    finally:
        if second_path.exists():
            second_path.unlink()


def license_document(source_dir: Path) -> str:
    upstream_license = (source_dir / "LICENSE").read_text(encoding="utf-8").strip()
    return (
        "# Licenses and provenance\n\n"
        f"`model.onnx` is a deterministic ONNX export of **{MODEL_DISPLAY_NAME}**.\n\n"
        f"- Upstream source: [Depth Anything V2]({SOURCE_REPOSITORY}), revision "
        f"`{SOURCE_REVISION}`.\n"
        "- Official checkpoint: "
        "[Depth-Anything-V2-Metric-Hypersim-Small]"
        "(https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small), "
        "revision `3bc65d4e14a6786a61acec16453c50e12bf5f338`.\n"
        "- Upstream license: Apache License 2.0.\n\n"
        "No third-party community ONNX weights are used. The graph is exported from "
        "the pinned official checkpoint with a static `image` "
        "`[1,3,294,392]` input and `depth` `[1,1,294,392]` meter output.\n\n"
        "## Apache License 2.0\n\n"
        "The complete upstream license text is reproduced below.\n\n"
        "```text\n"
        f"{upstream_license}\n"
        "```\n"
    )


def write_release_files(
    output_dir: Path,
    source_dir: Path,
    graph_validation: dict[str, Any],
    numeric_validation: dict[str, Any],
    graph_optimization: dict[str, Any],
    deterministic_validation: dict[str, Any] | None,
) -> None:
    model_path = output_dir / "model.onnx"
    artifact_bytes = model_path.stat().st_size
    artifact_sha256 = sha256_file(model_path)
    if artifact_bytes >= MAX_GIT_BLOB_BYTES:
        raise RuntimeError(
            f"ONNX is too large for an ordinary GitHub blob: {artifact_bytes} bytes"
        )

    manifest = {
        "formatVersion": 1,
        "modelName": MODEL_NAME,
        "version": MODEL_VERSION,
        "modelFile": "model.onnx",
        "modelBytes": artifact_bytes,
        "modelSha256": artifact_sha256,
        "inputName": INPUT_NAME,
        "outputName": OUTPUT_NAME,
        "inputShape": list(INPUT_SHAPE),
        "outputShape": list(OUTPUT_SHAPE),
        "preprocessVersion": PREPROCESS_VERSION,
        "unit": "meter",
        "minDepthMeters": 0.5,
        "maxDepthMeters": 20,
        "opset": OPSET,
    }
    metadata = {
        "model": MODEL_DISPLAY_NAME,
        "sourceRepository": SOURCE_REPOSITORY,
        "sourceRevision": SOURCE_REVISION,
        "checkpointUrl": CHECKPOINT_URL,
        "checkpointBytes": CHECKPOINT_BYTES,
        "checkpointSha256": CHECKPOINT_SHA256,
        "export": {
            "script": "tools/export_depth_anything_v2_metric_indoor_small_294x392.py",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "seed": SEED,
            "opset": OPSET,
            "dynamo": False,
            "doConstantFolding": True,
            "inputName": INPUT_NAME,
            "inputShape": list(INPUT_SHAPE),
            "outputName": OUTPUT_NAME,
            "outputShape": list(OUTPUT_SHAPE),
        },
        "preprocess": {
            "version": PREPROCESS_VERSION,
            "resize": "aspect-preserving fit with centered letterbox padding",
            "paddingRgb01": [0.485, 0.456, 0.406],
            "paddingNormalized": [0, 0, 0],
            "range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "layout": "NCHW",
        },
        "metricDepth": {
            "unit": "meter",
            "minimumAcceptedMeters": 0.5,
            "maximumMeters": 20,
        },
        "validation": {
            "staticGraphOptimization": graph_optimization,
            "graph": graph_validation,
            "fixedImageFixture": fixed_image_input(source_dir)[1],
            "pytorchVsOnnxRuntime": numeric_validation,
            "deterministicReexport": deterministic_validation,
        },
        "artifact": {
            "file": "model.onnx",
            "bytes": artifact_bytes,
            "sha256": artifact_sha256,
            "maximumOrdinaryGitBlobBytes": MAX_GIT_BLOB_BYTES,
            "gitStorage": "ordinary Git blob; no Git LFS",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "export-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "LICENSES.md").write_text(
        license_document(source_dir),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Export a second time and require byte-identical ONNX output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    require_environment()
    require_official_source(source_dir)
    require_checkpoint(checkpoint)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    contract = StaticMetricDepthContract(
        load_official_model(source_dir, checkpoint)
    ).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.onnx"
    graph_optimization = export_model(contract, model_path, source_dir)
    graph_validation = validate_graph(model_path)
    numeric_validation = validate_numerics(contract, model_path, source_dir)
    deterministic_validation = (
        verify_deterministic_reexport(
            contract,
            output_dir,
            model_path,
            source_dir,
        )
        if args.verify_determinism
        else None
    )
    write_release_files(
        output_dir,
        source_dir,
        graph_validation,
        numeric_validation,
        graph_optimization,
        deterministic_validation,
    )
    print(
        json.dumps(
            {
                "model": str(model_path),
                "bytes": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
                "inputShape": list(INPUT_SHAPE),
                "outputShape": list(OUTPUT_SHAPE),
                "onnxChecker": graph_validation["onnxChecker"],
                "staticGraphOptimization": graph_optimization,
                "pytorchVsOnnxRuntime": {
                    "maxAbsErrorMeters": numeric_validation["maxAbsErrorMeters"],
                    "meanAbsErrorMeters": numeric_validation["meanAbsErrorMeters"],
                },
                "deterministicReexport": deterministic_validation,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
