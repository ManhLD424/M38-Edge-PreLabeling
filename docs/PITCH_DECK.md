# Pitch outline — corrected evidence boundaries

1. Problem: model-generated boxes can reduce drawing effort, but incorrect boxes also create review cost. No time-saving percentage has been established here.
2. Implementation: YOLOv8s COCO pretrained -> ONNX/browser CPU or OpenVINO CPU -> reviewer tools -> CVAT XML.
3. Correctness: shared preprocessing/NMS, image/model hash provenance, stale-result protection and explicit replay mode.
4. Demonstration: run inference on an image, zoom, correct a box, change confidence, export XML.
5. Compression: show raw file sizes and actual timing protocol. Label consistency metrics as agreement with FP32, not mAP.
6. Evaluation plan: human GT holdout, CVAT roundtrip and annotator pilot are pending.
7. Limits: full BDD ontology and learned 3D detection are not validated product capabilities.

Use only the verified results in REPAIR_2026-10-16.md. Withdraw old 2.64-point mAP-drop and 70%-time-saving claims.
