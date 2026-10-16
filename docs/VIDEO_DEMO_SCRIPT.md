# Demo script — show actual supported behavior

1. Open the repaired revision and show the source selection. Explain live Browser CPU versus stored replay.
2. Select BDD06, run ONNX and wait for a real completion state. Show the measured browser inference time without comparing it to a different machine's replay timing.
3. Zoom into a box, edit its coordinates/label, undo once, then make the intended correction. Lower confidence to inspect missed objects and manually add one if appropriate.
4. Mark the frame reviewed and export CVAT XML. Explain the required matching filenames/ontology. Demonstrate live CVAT import/save/reload only after that test is actually performed.
5. Show the RGB/letterbox/NMS contract and model/image hash provenance.
6. Show raw consistency/timing records. State explicitly that consistency is not mAP and the image split has no independent scene/GT validation.
7. Show 3D only as experimental geometry proposals unless a real checkpoint/backend is loaded and separately verified.

Do not narrate invented side-by-side controls, a fixed speedup, a measured mAP drop or annotator productivity that has not been verified.
