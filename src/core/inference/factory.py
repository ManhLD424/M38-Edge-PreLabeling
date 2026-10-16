"""Explicit runtime selection; importing CLI does not import every ML backend."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = {'torch':'models/raw/yolov8s.pt','onnx':'models/onnx/yolov8s.onnx',
          'openvino_fp16':'models/openvino_fp16/yolov8s_fp16.xml',
          'openvino_int8':'models/openvino_int8_rgb/yolov8s_int8.xml'}


def make_engine(runtime, model_path=None, conf=.25):
    if runtime not in MODELS:
        raise ValueError(f'Unsupported runtime: {runtime}')
    if runtime == 'torch':
        from .torch_engine import TorchInferenceEngine as Engine
    elif runtime == 'onnx':
        from .onnx_engine import ONNXInferenceEngine as Engine
    else:
        from .ov_engine import OpenVINOInferenceEngine as Engine
    path = str(model_path or ROOT/MODELS[runtime])
    return Engine(path,conf_thresh=conf)
