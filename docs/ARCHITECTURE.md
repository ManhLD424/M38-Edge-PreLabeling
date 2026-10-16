# Architecture after repair

Browser: shared HTML/CSS + Studio state -> image SHA verification -> worker -> ONNX WASM CPU -> class-aware NMS -> reviewer edits -> CVAT XML.
Replay: same UI -> exact image/runtime key in predictions/manifest.json -> image SHA check -> reviewer. No fallback to another image.
Local: same UI -> loopback /api/predict -> validated sample/runtime/confidence -> serialized engine access -> real CPU inference -> reviewer.

RGB letterbox 640, padding114 and NMS IoU .45 are shared by calibration/ONNX/OpenVINO. Torch uses square inference with equivalent settings. Browser canvas resizing can differ from OpenCV near thresholds. CPU local threads default4, browser WASM1.

Every result is bound to image/runtime; stale responses are discarded. Export disabled while pending/error. Models and WASM are same-origin and fingerprinted. Uploaded images stay in browser. Backend serves only docs assets and allowlisted sample images.

3D is a separate experimental proposal module. Default is geometry heuristic. CVAT adapter requires explicit IDs and does not associate objects across frames. No authenticated CVAT E2E is claimed.
