# Testing report — repair 2026-10-16

Local environment: Python 3.12.10, Windows, exact direct dependency versions in requirements.txt. This is a record of executed checks, not 100% source coverage or a production certification.

## Automated checks

- Python suite covers existing geometry/PCD/dataclasses plus RGB input, class-aware NMS, malformed API requests, path containment, response image hashes, heuristic threshold/rear pass, explicit ontology/checkpoint failures, artifact preservation and actual packaged ONNX inference.
- COCO evaluator tests use synthetic human-GT-shaped fixtures for perfect/empty predictions; these prove evaluator wiring, not detector quality.
- JS tests cover class-aware decoding, missing-image replay failure, exact filtered XML/escaping and stale request generations.
- Artifact validator checks 26 images x 4 runtimes, per-image hashes, box bounds, split disjointness and model/vendor fingerprints.
- Canonical LF regression addresses the Windows/Linux license-file hash mismatch found in the GitHub verify job.

Commands and final counts are recorded in REPAIR_2026-10-16.md. CI has not yet validated the final uncommitted patch; its older revision failed vendor provenance validation, while the separate built-in Pages job published.

## Executed runtime checks

- Real Python engines on 26 images; controlled consistency on 10 held-out images.
- CPU timing on 26 real images, 3 repeats, thread budget 4, raw logs versioned.
- Compression smoke: fresh output directory, checkpoint -> ONNX -> OV FP16 -> recalibrated INT8, old artifacts preserved.
- Current local API: all four engines return 200 with the exact requested image hash/dimensions.
- Browser localhost: real ONNX inference, numeric box edit, review and XML export.
- Public Pages: real ONNX inference on BDD06, distinct image-specific FP16 replay, confidence .90 -> zero visible/exported boxes; stale-response navigation checked separately.

## Not established

Authenticated CVAT import/save/reload, human-GT BDD AP/drop, independent driving-scene holdout, annotator productivity, cold full dependency install, mobile/cross-browser matrix and final CI/public rollout of the last hardening patch. Session JSON export does not currently have a session-import workflow.

## Follow-up checks

50 Python tests pass after explicit CVAT revision/authentication guards and pytest collection scope were added. Datumaro 1.13.11 independently imported four Python/browser XML fixture datasets, including the zero-box case. This verifies format compatibility only. Reproduce with an isolated Datumaro environment and `python tools/verify_cvat_export.py --parser-python <environment-python>`.
