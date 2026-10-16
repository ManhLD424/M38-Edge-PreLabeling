# Mentor submission status — corrected

M38 demonstrates a COCO-pretrained 2D detector deployed on CPU with ONNX/browser and OpenVINO, plus reviewer tools and CVAT XML export. Static demo data was replaced with real image-specific predictions and live browser inference after an audit found correctness defects.

Current evidence: hash-bound artifacts, runtime consistency on an illustrative image holdout, real-image latency logs and regression tests. These are not ground-truth detection accuracy or measured annotator productivity.

Required before external claims: human GT evaluation, CVAT import/save/reload on a test task, paired annotator pilot and verification of the published revision. 3D is an experimental geometric proposal module by default, not evidence of optimized learned PointPillars.

Use README.md for actual CLI commands, REPAIR_2026-10-16.md for the local verification record, and SUBMISSION_CHECKLIST.md for pending gates.
