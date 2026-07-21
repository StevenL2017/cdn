# Licenses and provenance

`model.onnx` is a deterministic ONNX export of **Depth Anything V2 Metric Indoor Small (Hypersim ViT-S)**.

- Upstream source: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), revision `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
- Official checkpoint: [Depth-Anything-V2-Metric-Hypersim-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small), revision `3bc65d4e14a6786a61acec16453c50e12bf5f338`.
- Upstream license: Apache License 2.0. See the source repository's [LICENSE](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/LICENSE).

No third-party community ONNX weights are used. The graph is exported from the official checkpoint and then wrapped only to expose the stable `image` input and `depth` `[1,1,392,518]` output contract.
