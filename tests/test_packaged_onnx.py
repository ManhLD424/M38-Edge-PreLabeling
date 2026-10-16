"""Exercise the actual shipped model and public image, not mock detections."""
import hashlib
import json
from pathlib import Path
import cv2
from src.core.inference.onnx_engine import ONNXInferenceEngine
from src.core.benchmark.accuracy_bench import compute_iou


def test_packaged_model_matches_image_specific_replay():
    manifest=json.loads(Path('docs/predictions/manifest.json').read_text(encoding='utf-8'))
    record=manifest['images']['BDD06.jpg']
    image=Path('docs/images/BDD06.jpg')
    assert hashlib.sha256(image.read_bytes()).hexdigest()==record['sha256']
    engine=ONNXInferenceEngine('docs/models/yolov8s.onnx',conf_thresh=.25)
    actual=engine.predict(cv2.imread(str(image)))
    expected=[d for d in record['runtimes']['onnx']['detections'] if d['confidence']>=.25]
    assert len(actual)==len(expected)
    for detection in actual:
        assert any(detection.class_id==d['class_id'] and compute_iou(detection.box_xyxy,d['box_xyxy'])>.99 for d in expected)
