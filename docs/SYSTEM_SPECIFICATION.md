# System specification — corrected 2026-10-16

## Supported

- 26 sample images, live browser ONNX CPU, per-image replay from four Python runtimes.
- Shared UI for static Pages and loopback server; local CPU API available when models exist.
- Zoom/pan, choose/move/resize/delete/draw boxes, numeric/label editing, undo.
- Confidence/traffic scope applies equally to viewer and export; manual edits remain visible.
- CVAT for images 1.1 XML for current or reviewed frames; custom Studio session JSON.
- RGB letterbox preprocessing, class-aware NMS, explicit model/input fingerprints.
- NNCF PTQ on real calibration images only; hash-disjoint illustrative image split.
- Model consistency separate from COCO AP requiring human GT; raw real-image timings.

## Limits and acceptance gates

COCO pretrained does not cover full BDD ontology. No validated BDD mAP, annotator time saving, live CVAT import/save/reload or scene-independent holdout evidence. INT8 boxes require review. 3D default is heuristic, no learned-model compression evidence. No multi-frame temporal association. User edits stay in tab memory; JSON export preserves data but there is currently no session-import workflow.

Public deployment requires repository publication. Local/browser checks do not prove public rollout. See REPAIR_2026-10-16.md for exact evidence and pending gates.
