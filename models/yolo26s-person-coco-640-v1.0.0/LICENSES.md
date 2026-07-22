# Licenses and provenance

`model.onnx` is a deterministic FP32 ONNX export of the official **Ultralytics YOLO26s COCO** checkpoint with the classic detection head enabled.

- Upstream source: [Ultralytics](https://github.com/ultralytics/ultralytics), revision [`16665db343532a0f94d04bf2489ee0028da10346`](https://github.com/ultralytics/ultralytics/tree/16665db343532a0f94d04bf2489ee0028da10346).
- Official checkpoint: [`yolo26s.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt), SHA-256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`.
- Upstream model and software license: [GNU Affero General Public License v3.0](https://github.com/ultralytics/ultralytics/blob/16665db343532a0f94d04bf2489ee0028da10346/LICENSE), identified upstream as `AGPL-3.0`.

No community ONNX weights are used. The graph was exported locally from the verified official checkpoint with Ultralytics `8.4.34`, `end2end=false`, `nms=false`, static input shape `[1,3,640,640]`, and opset 11. Exporter timestamp metadata was removed so identical inputs and tool versions produce an identical model artifact.
