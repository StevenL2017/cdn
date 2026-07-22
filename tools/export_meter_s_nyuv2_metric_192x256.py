#!/usr/bin/env python3
"""Deterministically export and validate the official METER-S NYUv2 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as functional


MODEL_NAME = "meter-s-nyuv2-metric"
MODEL_VERSION = "meter-s-nyuv2-metric-192x256-v1.0.0"
MODEL_DISPLAY_NAME = "METER-S NYUv2 Metric Depth"
SOURCE_REPOSITORY = "https://github.com/lorenzopapa5/METER"
SOURCE_REVISION = "0f9a49ada3af88393ba5dce3839ab471329e37bd"
SOURCE_URL = (
    "https://raw.githubusercontent.com/lorenzopapa5/METER/"
    f"{SOURCE_REVISION}/architecture.py"
)
SOURCE_BYTES = 11_846
SOURCE_SHA256 = "876babc9c42df809016f2257d304b1901b10c57ae1ab06c40be95dfce7afa188"
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/lorenzopapa5/METER/"
    f"{SOURCE_REVISION}/models/build_model_best_nyu_s"
)
CHECKPOINT_BYTES = 13_326_903
CHECKPOINT_SHA256 = "2bd1a7410d311ba5d9346b1d9aa5721d54c446791d9e0b619d830fe1072d4708"
INPUT_SHAPE = (1, 3, 192, 256)
RAW_OUTPUT_SHAPE = (1, 1, 48, 64)
OUTPUT_SHAPE = (1, 1, 192, 256)
PREPROCESS_VERSION = "meter-s-nyuv2-rgb-zero-one-letterbox-192x256-v1"
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


def meter_rearrange(tensor: torch.Tensor, pattern: str, **axes: int) -> torch.Tensor:
    """Implement the four static einops patterns used by upstream METER."""
    if pattern.startswith("b d "):
        batch, channels, padded_height, padded_width = tensor.shape
        patch_height, patch_width = axes["ph"], axes["pw"]
        height = padded_height // patch_height
        width = padded_width // patch_width
        return (
            tensor.reshape(batch, channels, height, patch_height, width, patch_width)
            .permute(0, 3, 5, 2, 4, 1)
            .reshape(batch, patch_height * patch_width, height * width, channels)
        )
    if pattern.startswith("b (ph pw)"):
        batch, _, _, channels = tensor.shape
        patch_height, patch_width = axes["ph"], axes["pw"]
        height, width = axes["h"], axes["w"]
        return (
            tensor.reshape(batch, patch_height, patch_width, height, width, channels)
            .permute(0, 5, 3, 1, 4, 2)
            .reshape(batch, channels, height * patch_height, width * patch_width)
        )
    if pattern.startswith("b p n"):
        batch, patches, tokens, combined = tensor.shape
        heads = axes["h"]
        head_dim = combined // heads
        return tensor.reshape(batch, patches, tokens, heads, head_dim).permute(0, 1, 3, 2, 4)
    if pattern.startswith("b p h"):
        batch, patches, heads, tokens, head_dim = tensor.shape
        return tensor.permute(0, 1, 3, 2, 4).reshape(
            batch, patches, tokens, heads * head_dim
        )
    raise ValueError(f"Unsupported METER rearrange pattern: {pattern}")


def load_upstream_model(source_path: Path, checkpoint_path: Path) -> torch.nn.Module:
    require_file(source_path, SOURCE_BYTES, SOURCE_SHA256, "official architecture source")
    require_file(checkpoint_path, CHECKPOINT_BYTES, CHECKPOINT_SHA256, "official checkpoint")
    source = source_path.read_text(encoding="utf-8")

    cuda_literal = "device='cuda:0'"
    broken_xs_return = (
        "return MobileViT((RGB_img_res[1], RGB_img_res[2])), dims, channels), enc_type"
    )
    if source.count(cuda_literal) != 1 or source.count(broken_xs_return) != 1:
        raise RuntimeError("Pinned upstream source no longer matches the audited export patch")
    source = source.replace(cuda_literal, "device='cpu'")
    source = source.replace(
        broken_xs_return,
        "return MobileViT((RGB_img_res[1], RGB_img_res[2]), dims, channels), enc_type",
    )

    globals_module = types.ModuleType("globals")
    globals_module.RGB_img_res = (3, 192, 256)
    einops_module = types.ModuleType("einops")
    einops_module.rearrange = meter_rearrange
    old_globals = sys.modules.get("globals")
    old_einops = sys.modules.get("einops")
    sys.modules["globals"] = globals_module
    sys.modules["einops"] = einops_module
    try:
        architecture_module = types.ModuleType("meter_pinned_architecture")
        exec(compile(source, str(source_path), "exec"), architecture_module.__dict__)
        model = architecture_module.build_METER_model("cpu", "s")
    finally:
        if old_globals is None:
            sys.modules.pop("globals", None)
        else:
            sys.modules["globals"] = old_globals
        if old_einops is None:
            sys.modules.pop("einops", None)
        else:
            sys.modules["einops"] = old_einops

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    return model.eval()


class MetricDepthWrapper(torch.nn.Module):
    """Expose a stable full-resolution metric-depth contract for the mini program."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        depth_centimeters = self.model(image)
        depth_meters = functional.interpolate(
            depth_centimeters,
            size=(OUTPUT_SHAPE[2], OUTPUT_SHAPE[3]),
            mode="bilinear",
            align_corners=False,
        ) / 100.0
        return torch.clamp(depth_meters, 0.1, 10.0)


def tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)


def canonicalize_model(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=True)
    model.producer_name = "PyTorch"
    model.producer_version = torch.__version__
    model.model_version = 1
    model.doc_string = ""
    metadata = {
        "description": MODEL_DISPLAY_NAME,
        "source": CHECKPOINT_URL,
        "source_revision": SOURCE_REVISION,
        "preprocess_version": PREPROCESS_VERSION,
        "export_contract": "image-fp32-rgb01-nchw-1x3x192x256_to_depth-fp32-meter-1x1x192x256",
        "raw_model_output": "centimeters-1x1x48x64",
        "wrapper": "bilinear-resize-192x256_divide-100_clip-0.1-10-meter",
    }
    del model.metadata_props[:]
    for key in sorted(metadata):
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = metadata[key]
    path.write_bytes(model.SerializeToString(deterministic=True))


def validate_graph(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    if [(item.domain, item.version) for item in model.opset_import] != [("", 11)]:
        raise RuntimeError("Expected only default-domain opset 11")
    if len(model.graph.input) != 1 or model.graph.input[0].name != "image":
        raise RuntimeError("Expected one input named image")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "depth":
        raise RuntimeError("Expected one output named depth")
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
        "nodeCount": len(model.graph.node),
        "operators": sorted({node.op_type for node in model.graph.node}),
        "externalDataTensors": 0,
    }


def validate_numerics(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    sample_results = []
    maximum_error = 0.0
    mean_errors = []
    for offset in range(3):
        rng = np.random.default_rng(SEED + offset)
        input_array = rng.random(INPUT_SHAPE, dtype=np.float32)
        with torch.inference_mode():
            torch_array = model(torch.from_numpy(input_array)).cpu().numpy()
        ort_array = session.run(["depth"], {"image": input_array})[0]
        if tuple(torch_array.shape) != OUTPUT_SHAPE or tuple(ort_array.shape) != OUTPUT_SHAPE:
            raise RuntimeError(f"Unexpected numeric output shape: {torch_array.shape}, {ort_array.shape}")
        error = np.abs(torch_array - ort_array)
        sample_max = float(error.max())
        sample_mean = float(error.mean())
        maximum_error = max(maximum_error, sample_max)
        mean_errors.append(sample_mean)
        sample_results.append(
            {
                "seed": SEED + offset,
                "maxAbsError": sample_max,
                "meanAbsError": sample_mean,
                "outputMinMeters": float(ort_array.min()),
                "outputMedianMeters": float(np.median(ort_array)),
                "outputMaxMeters": float(ort_array.max()),
            }
        )
    if maximum_error > 0.00001:
        raise RuntimeError(f"PyTorch/ORT error exceeded tolerance: {maximum_error}")
    return {
        "provider": session.get_providers()[0],
        "sampleCount": len(sample_results),
        "inputDistribution": "numpy.default_rng.random float32 [0,1]",
        "maxAbsError": maximum_error,
        "meanAbsError": float(np.mean(mean_errors)),
        "samples": sample_results,
    }


def export_model(source_path: Path, checkpoint_path: Path, output_path: Path) -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = MetricDepthWrapper(load_upstream_model(source_path, checkpoint_path)).eval()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    example = torch.rand(INPUT_SHAPE, generator=generator, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".onnx.tmp")
    torch.onnx.export(
        model,
        example,
        temporary,
        input_names=["image"],
        output_names=["depth"],
        opset_version=11,
        dynamo=False,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    os.replace(temporary, output_path)
    canonicalize_model(output_path)
    graph = validate_graph(output_path)
    numeric = validate_numerics(model, output_path)
    return {
        "model": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "graph": graph,
        "numeric": numeric,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = export_model(
        args.source.resolve(),
        args.checkpoint.resolve(),
        args.output.resolve(),
    )
    result["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
