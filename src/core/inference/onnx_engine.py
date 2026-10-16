"""ONNX CPU engine using the same preprocessing/NMS as OpenVINO."""
from pathlib import Path
import onnxruntime as ort
from .base import BaseInferenceEngine
from .preprocessing import prepare_image, decode_predictions


class ONNXInferenceEngine(BaseInferenceEngine):
    def __init__(self, model_path, conf_thresh=.25, iou_thresh=.45):
        super().__init__(model_path, conf_thresh, iou_thresh)
        self.load_model()

    def load_model(self):
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f'Missing ONNX model: {self.model_path}')
        from .settings import CPU_THREADS
        options = ort.SessionOptions()
        options.intra_op_num_threads = CPU_THREADS
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image):
        tensor, ratio, padding = prepare_image(image)
        output = self.session.run(None, {self.input_name:tensor})[0]
        return decode_predictions(output, image.shape, ratio, padding, self.conf_thresh, self.iou_thresh)
