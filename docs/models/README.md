# Browser model

`yolov8s.onnx` is the locally exported COCO-pretrained Ultralytics YOLOv8s model.
Its SHA-256, byte size and producing checkpoint fingerprint are in
`../predictions/manifest.json`. Browser inference verifies the model hash before
creating a session. No BDD fine-tuning or ground-truth validation is claimed.

Upstream model/code: https://github.com/ultralytics/ultralytics
Licensing information: https://www.ultralytics.com/license
The project's MIT license does not override upstream model/runtime licenses.

Input: float32 RGB NCHW [1,3,640,640], letterbox padding 114, divide by 255.
Output: [1,84,8400], xywh plus 80 COCO class scores. Class-aware NMS is outside
the graph, IoU 0.45, max 300 detections. Browser canvas and OpenCV use different
resize implementations, so predictions near score thresholds can differ.
