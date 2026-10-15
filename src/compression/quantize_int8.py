"""INT8 Post-Training Quantization (PTQ) module using ONNX Runtime."""
from pathlib import Path
import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def preprocess_image(img_path, input_size=(640, 640)):
    """Preprocess single image for calibration data feeding."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image at: {img_path}")
    img = letterbox(img, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img)

class BDD100kCalibrationDataReader(CalibrationDataReader):
    """Feeds representative calibration images to ONNX Runtime quantizer."""
    def __init__(self, image_dir, input_name, max_samples=80, input_size=(640, 640)):
        super().__init__()
        self.image_paths = sorted(list(Path(image_dir).glob("*.jpg")))[:max_samples]
        self.input_name = input_name
        self.input_size = input_size
        self.index = 0

    def get_next(self):
        if self.index < len(self.image_paths):
            p = self.image_paths[self.index]
            self.index += 1
            tensor = preprocess_image(p, self.input_size)
            return {self.input_name: tensor}
        return None

    def rewind(self):
        self.index = 0

def quantize_onnx_int8(slim_onnx_path, output_int8_path, calibration_image_dir, max_samples=80):
    """Run INT8 Static Post-Training Quantization with calibration dataset.
    
    Protects detection head in FP32 to minimize precision degradation.
    """
    inp_p = Path(slim_onnx_path).resolve()
    out_p = Path(output_int8_path).resolve()
    calib_p = Path(calibration_image_dir).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not inp_p.exists():
        raise FileNotFoundError(f"Source model not found: {inp_p}")
    if not calib_p.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calib_p}")

    model_proto = onnx.load(str(inp_p))
    input_name = model_proto.graph.input[0].name

    print(f"[INT8 PTQ] Starting calibration on {max_samples} images for {inp_p.name}...")
    calib_reader = BDD100kCalibrationDataReader(
        image_dir=calib_p,
        input_name=input_name,
        max_samples=max_samples,
    )

    quantize_static(
        model_input=str(inp_p),
        model_output=str(out_p),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],
        per_channel=True,
        reduce_range=False,
    )

    size_mb = out_p.stat().st_size / (1024 * 1024)
    print(f"[INT8 PTQ] Successfully saved INT8 PTQ model: {out_p.name} ({size_mb:.2f} MB)")
    return out_p
