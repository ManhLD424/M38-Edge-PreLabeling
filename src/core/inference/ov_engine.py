"""Native OpenVINO YOLOv8 engine with shared preprocessing and class-aware NMS."""
from pathlib import Path
import openvino as ov
from .base import BaseInferenceEngine
from .preprocessing import prepare_image, decode_predictions


class OpenVINOInferenceEngine(BaseInferenceEngine):
    def __init__(self, model_path, conf_thresh=.25, iou_thresh=.45):
        super().__init__(model_path, conf_thresh, iou_thresh)
        self.core = ov.Core()
        self.load_model()

    def load_model(self):
        path = Path(self.model_path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing OpenVINO model: {path}')
        model = self.core.read_model(str(path))
        self.compiled_model = self.core.compile_model(model, 'CPU',
            {'PERFORMANCE_HINT':'LATENCY', 'NUM_STREAMS':'1'})
        self.output_layer = self.compiled_model.output(0)

    def predict(self, image):
        tensor, ratio, padding = prepare_image(image)
        output = self.compiled_model([tensor])[self.output_layer]
        return decode_predictions(output, image.shape, ratio, padding, self.conf_thresh, self.iou_thresh)
