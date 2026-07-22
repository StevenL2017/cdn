# Licenses and provenance

`model.onnx` is a deterministic ONNX export of **METER-S trained on NYU Depth V2**.

- Upstream source and official checkpoint: [lorenzopapa5/METER](https://github.com/lorenzopapa5/METER/tree/0f9a49ada3af88393ba5dce3839ab471329e37bd).
- Official checkpoint file: [`build_model_best_nyu_s`](https://raw.githubusercontent.com/lorenzopapa5/METER/0f9a49ada3af88393ba5dce3839ab471329e37bd/models/build_model_best_nyu_s).
- Upstream license: MIT License. See the pinned upstream [LICENSE](https://github.com/lorenzopapa5/METER/blob/0f9a49ada3af88393ba5dce3839ab471329e37bd/LICENSE).

The release graph uses the official state dictionary without retraining. The export-only compatibility changes are documented in `export-metadata.json`. The graph wraps the upstream quarter-resolution centimeter output with bilinear upsampling, conversion to meters, and clipping to the NYUv2 depth interval.

## MIT License

Copyright (c) 2022 lorenzopapa5

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
