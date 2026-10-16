# Changelog

## Unreleased — 2026-10-16 repair

- Replaced hardcoded demo labels with browser ONNX inference and hash-bound per-image replay.
- Shared RGB/letterbox preprocessing and class-aware NMS across Python runtimes; bounded CPU threads.
- Added zoom, pan, box editing/drawing/deletion, undo, review snapshots and CVAT XML export.
- Recalibrated INT8 using a frozen calibration split, preserving old artifacts.
- Separated cross-model agreement from actual COCO AP requiring human ground truth.
- Hardened loopback API against path traversal, malformed inputs and concurrent threshold mutation.
- Required explicit CVAT ontology mapping; fixed create/update semantics and stopped silent 3D checkpoint fallback.
- Fixed heuristic confidence/full-scene behavior and binary PCD field parsing.
- Added artifact validation and quality gates before Pages deploy; corrected unsupported product/benchmark claims.

Not yet a production release: live CVAT save/reload, ground-truth evaluation and annotator pilot remain unverified.
