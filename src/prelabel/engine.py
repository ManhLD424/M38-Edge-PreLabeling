"""High-performance Edge Pre-Labeling inference engine using ONNX Runtime."""
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

BDD100K_CLASSES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motor", "bike", "traffic light", "traffic sign"
]

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] in original image coordinates
    bbox_xywhn: List[float] # [x_center, y_center, width, height] normalized (0..1)
    status: str             # "AUTO_APPROVED" or "REVIEW_NEEDED"

@dataclass
class InferenceResult:
    image_path: str
    image_shape: Tuple[int, int] # (height, width)
    detections: List[Detection]
    latency_ms: float
    fps: float
    auto_approved_count: int
    review_needed_count: int

class EdgePreLabelEngine:
    """Standalone edge inference engine for compressed ONNX models."""
    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        auto_accept_conf: float = 0.80,
        device: str = "cpu"
    ):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.auto_accept_conf = auto_accept_conf

        # Execution providers
        providers = ["CPUExecutionProvider"]
        if device.lower() in ("gpu", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type  # e.g. 'tensor(float16)' or 'tensor(float)'
        self.classes = BDD100K_CLASSES

    def letterbox(self, img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
        """Standard letterbox resize and padding."""
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Preprocess an input BGR image."""
        letterboxed, ratio, (dw, dh) = self.letterbox(img, (640, 640))
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        target_dtype = np.float16 if "float16" in self.input_type else np.float32
        tensor = rgb.astype(target_dtype) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return np.ascontiguousarray(tensor), ratio, (dw, dh)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Compute Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def predict(self, image: np.ndarray, image_path: str = "") -> InferenceResult:
        """Run end-to-end inference on an image."""
        orig_h, orig_w = image.shape[:2]
        t0 = time.perf_counter()

        tensor, ratio, (dw, dh) = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        raw = outputs[0]  # shape: (1, 14, 8400) or (1, 8400, 14)

        if raw.dtype == np.float16:
            raw = raw.astype(np.float32)

        if raw.shape[1] < raw.shape[2]:
            preds = np.transpose(raw[0], (1, 0))  # (8400, 14)
        else:
            preds = raw[0]

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        max_classes = np.argmax(class_scores, axis=1)

        mask = max_scores >= self.conf_thresh
        filtered_boxes = boxes_cxcywh[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = max_classes[mask]

        detections: List[Detection] = []
        if len(filtered_boxes) > 0:
            # Convert center xywh to xyxy
            x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
            y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
            x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
            y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
            boxes_xyxy = np.column_stack([x1, y1, x2, y2])

            # Multiclass NMS
            for c in np.unique(filtered_classes):
                c_mask = filtered_classes == c
                c_boxes = boxes_xyxy[c_mask]
                c_scores = filtered_scores[c_mask]

                keep_indices = self._nms(c_boxes, c_scores, self.iou_thresh)
                for idx in keep_indices:
                    box = c_boxes[idx]
                    score = float(c_scores[idx])

                    # Rescale back to original image
                    box_orig = [
                        (box[0] - dw) / ratio,
                        (box[1] - dh) / ratio,
                        (box[2] - dw) / ratio,
                        (box[3] - dh) / ratio
                    ]
                    # Clip coordinates
                    bx1 = max(0.0, min(float(orig_w), box_orig[0]))
                    by1 = max(0.0, min(float(orig_h), box_orig[1]))
                    bx2 = max(0.0, min(float(orig_w), box_orig[2]))
                    by2 = max(0.0, min(float(orig_h), box_orig[3]))

                    # Normalized xywh
                    bw = bx2 - bx1
                    bh = by2 - by1
                    bcx = bx1 + bw / 2.0
                    bcy = by1 + bh / 2.0

                    xywhn = [
                        bcx / orig_w,
                        bcy / orig_h,
                        bw / orig_w,
                        bh / orig_h
                    ]

                    status = "AUTO_APPROVED" if score >= self.auto_accept_conf else "REVIEW_NEEDED"
                    cls_name = self.classes[int(c)] if int(c) < len(self.classes) else f"class_{c}"

                    detections.append(Detection(
                        class_id=int(c),
                        class_name=cls_name,
                        confidence=round(score, 4),
                        bbox_xyxy=[round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                        bbox_xywhn=[round(v, 6) for v in xywhn],
                        status=status
                    ))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        auto_approved = sum(1 for d in detections if d.status == "AUTO_APPROVED")
        review_needed = sum(1 for d in detections if d.status == "REVIEW_NEEDED")

        return InferenceResult(
            image_path=str(image_path),
            image_shape=(orig_h, orig_w),
            detections=detections,
            latency_ms=round(latency_ms, 2),
            fps=round(fps, 1),
            auto_approved_count=auto_approved,
            review_needed_count=review_needed
        )
