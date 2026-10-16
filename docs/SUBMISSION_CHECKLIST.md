# Submission checklist — evidence required

- [ ] Publish repaired revision and inspect the actual public URL after deployment.
- [ ] Import exported XML on a disposable CVAT task with matching filenames/labels; save and reload.
- [ ] Obtain independently annotated holdout with dataset/scene provenance; run actual COCO AP for baseline and INT8.
- [ ] Measure annotator correction time and residual errors in a paired pilot.
- [ ] Review upstream model and runtime license terms before redistribution/submission.
- [ ] Record final revision, dataset/model hashes, benchmark protocol and raw results.

Implemented locally: real ONNX browser inference, per-image replay, manual review tools, XML export, CPU engines, PTQ and regression tests. See REPAIR_2026-10-16.md for verified scope. Do not submit the former mAP/drop/speedup/time-saving claims as measured product outcomes.
