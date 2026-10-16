"""Regressions for the deployed-demo/inference audit of 2026-10-16."""
import json
import threading
import xml.etree.ElementTree as ET
from http.server import HTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests


def fake_ov(output):
    from src.core.inference.ov_engine import OpenVINOInferenceEngine
    engine = object.__new__(OpenVINOInferenceEngine)
    engine.conf_thresh = .25
    engine.iou_thresh = .45
    engine.output_layer = 'out'
    captured = []
    def infer(inputs):
        captured.append(inputs[0])
        return {'out': output}
    engine.compiled_model = infer
    return engine, captured


def test_openvino_converts_bgr_to_rgb():
    engine, captured = fake_ov(np.zeros((1, 84, 1), np.float32))
    image = np.zeros((640, 640, 3), np.uint8)
    image[:, :, 2] = 255
    engine.predict(image)
    assert captured[0][0, 0, 0, 0] == 1
    assert captured[0][0, 2, 0, 0] == 0


def test_openvino_keeps_overlapping_objects_of_different_classes():
    output = np.zeros((1, 84, 3), np.float32)
    output[0, :4] = [[320]*3, [320]*3, [100]*3, [100]*3]
    output[0, 4, 0] = .9  # person
    output[0, 6, 1] = .8  # car, must survive
    output[0, 4, 2] = .7  # duplicate person, must be suppressed
    engine, _ = fake_ov(output)
    assert [d.class_name for d in engine.predict(np.zeros((640, 640, 3), np.uint8))] == ['person', 'car']


def test_3d_heuristic_respects_score_threshold():
    from src.core.models.detector import PointPillarsDetector, DetectorConfig
    from src.core.pointcloud.pcd_io import read_pcd
    detector = PointPillarsDetector(DetectorConfig(score_thresh=.99))
    assert detector.detect(read_pcd('data/pointclouds/demo.pcd').xyz) == []


def test_requested_missing_checkpoint_cannot_silently_fallback():
    from src.core.models.detector import PointPillarsDetector, DetectorConfig
    with pytest.raises(FileNotFoundError):
        PointPillarsDetector(DetectorConfig(checkpoint_path='missing-weights.pth'))


def test_cvat_adapter_requires_explicit_ontology():
    from src.core.geometry.cuboid import Cuboid3D
    from src.core.geometry.cvat_adapter import CVAT3DAdapter
    with pytest.raises(ValueError, match='label'):
        CVAT3DAdapter().to_shapes_payload([Cuboid3D(0, 0, 0, 4, 2, 1.5)])


@pytest.fixture
def api_server():
    from src.web.app import WebMVPHandler
    server = HTTPServer(('127.0.0.1', 0), WebMVPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def test_api_image_cannot_escape_sample_directory(api_server):
    response = requests.get(api_server+'/api/image', params={'name': '../../README.md'}, timeout=5)
    assert response.status_code == 400
    assert response.json()['error']


def test_api_handles_invalid_json_and_remains_usable(api_server):
    response = requests.post(api_server+'/api/predict', data='{bad', timeout=5)
    assert response.status_code == 400
    assert requests.get(api_server+'/api/samples', timeout=5).status_code == 200


def test_api_rejects_unknown_runtime_before_loading_model(api_server):
    response = requests.post(api_server+'/api/predict', json={'runtime':'unknown','image_name':'BDD01.jpg'}, timeout=5)
    assert response.status_code == 400


def test_prelabel_export_is_cvat_xml_with_filtered_boxes():
    from src.core.cvat.export import export_cvat_xml
    records = [{'image':'road&<1>.jpg','width':1280,'height':720,'shapes':[
        {'label':'car','points':[1,2,30,40],'confidence':.8},
        {'label':'car','points':[50,60,70,80],'confidence':.2}]}]
    xml = ET.fromstring(export_cvat_xml(records, conf_thresh=.5))
    assert xml.findtext('version') == '1.1'
    assert xml.find('image').get('name') == 'road&<1>.jpg'
    assert len(xml.findall('image/box')) == 1
    assert xml.find('image/box').get('xtl') == '1.0'


def test_calibration_rejects_missing_real_data(tmp_path):
    from src.core.compression.quantizer import ModelQuantizer
    quantizer = object.__new__(ModelQuantizer)
    with pytest.raises(ValueError, match='calibration'):
        quantizer._prepare_calibration_data(str(tmp_path), subset_size=5)
