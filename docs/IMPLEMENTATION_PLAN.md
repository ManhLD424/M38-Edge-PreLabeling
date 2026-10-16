# Repair gates

1. RGB/letterbox, class-aware NMS, hash-bound predictions, no cross-image fallback.
2. Actual browser inference, clear replay labeling, request generation and disabled stale export.
3. Reviewer tools and CVAT XML, tested for filtering and XML escaping.
4. Real calibration, disjoint frozen image split, re-quantization without overwriting old models.
5. Correct consistency terminology; COCO AP only with human GT.
6. API containment, loopback default, strict validation and serialized engine access.
7. 3D backend identity, explicit checkpoint errors, threshold/full-scene behavior and safe ontology mapping.
8. Correct documentation, model provenance, fresh regression checks and pre-deploy CI gates.

Local implementation and verification are recorded in REPAIR_2026-10-16.md. Public rollout, authenticated CVAT roundtrip, GT evaluation and annotator pilot are separate pending gates. No deadline or completion claim is inferred from this plan.
