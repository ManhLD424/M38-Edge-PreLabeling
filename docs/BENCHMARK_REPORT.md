# Benchmark report — corrected 2026-10-16

## Runtime consistency, not accuracy

10 holdout images, 97 PyTorch reference boxes, confidence .25 and class-aware NMS IoU .45. Calibration uses 16 different image hashes. This is an illustrative image split, not scene-independent human GT.

| Candidate | Boxes | Matched at IoU >= .5 | Reference retention | Mean matched IoU |
|---|---:|---:|---:|---:|
| ONNX | 97 | 97 | 100.00% | 1.0000 |
| OV FP16 | 97 | 97 | 100.00% | 0.9997 |
| Recalibrated OV INT8 | 99 | 94 | 96.91% | 0.8092 |

These numbers are agreement with another model, not mAP, precision against truth, or proof of useful annotation productivity. AP requires human GT and the COCO evaluator. Old F1-as-mAP claims and derived mAP-drop claims are withdrawn.

## Real-image CPU timing

Python 3.12, Windows host recorded in predictions/latency.json. CPU thread budget 4 for Torch/ORT/OV; 3 warmup passes and 3 repeats of 26 images. Sequential predict() includes preprocessing and NMS, excludes disk read/model startup. No streaming-throughput claim, no guaranteed thermal isolation or stable wall-clock speedup. Browser is a different runtime/host and is measured separately in UI.

| Runtime | Mean ms | P95 ms |
|---|---:|---:|
| torch | 153.64 | 186.65 |
| onnx | 138.58 | 222.79 |
| openvino_fp16 | 119.77 | 145.37 |
| openvino_int8 | 38.80 | 42.05 |

Raw per-image/per-repeat timings, model/image fingerprints and dependency versions: [latency.json](predictions/latency.json). Consistency data: [consistency.json](predictions/consistency.json). Replay's single-image timings are provenance, not an independently controlled comparison.

## Reproduction

```powershell
python -m src.cli.main compare-runtimes --save-json outputs/consistency-v2.json
python -m src.cli.main benchmark-2d --runs 3 --warmup 3 --save-json outputs/latency-v2.json
python tools/plot_benchmarks.py
```

Requires local weights as described in README. Default model paths are explicit in inference/factory.py. No measured annotator-time saving or live CVAT roundtrip is claimed.
