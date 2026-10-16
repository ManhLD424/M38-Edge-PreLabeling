"""YOLO contract shared by inference and calibration: BGR -> RGB NCHW."""
import cv2
import numpy as np


def prepare_image(image, size=640):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected a nonempty HWC BGR image with 3 channels')
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    rw, rh = round(width * ratio), round(height * ratio)
    left, top = round((size-rw)/2-.1), round((size-rh)/2-.1)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, size-rh-top, left, size-rw-left,
                               cv2.BORDER_CONSTANT, value=(114,114,114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2,0,1)[None], dtype=np.float32) / 255.
    return tensor, ratio, (left, top)


def decode_predictions(output, image_shape, ratio, padding, conf_thresh, iou_thresh, max_det=300):
    from .base import COCO_CLASSES, Detection
    predictions = np.asarray(output)[0].T
    if predictions.shape[1] != 84:
        raise ValueError(f'Expected YOLO COCO output [1,84,N], got {np.shape(output)}')
    scores = predictions[:, 4:]
    classes, confidences = scores.argmax(axis=1), scores.max(axis=1)
    valid = np.isfinite(predictions).all(axis=1) & (confidences >= conf_thresh)
    centers = predictions[valid, :4]
    classes, confidences = classes[valid], confidences[valid]
    if not len(centers):
        return []
    # Suppress in model space, before clipping, as in Ultralytics.
    xywh = np.column_stack((centers[:, :2] - centers[:, 2:]/2, centers[:, 2:]))
    kept = []
    for cid in np.unique(classes):
        indices = np.flatnonzero(classes == cid)
        selected = cv2.dnn.NMSBoxes(xywh[indices].tolist(), confidences[indices].tolist(),
                                    max(0., conf_thresh-1e-7), iou_thresh)
        kept.extend(indices[np.asarray(selected).reshape(-1).astype(int)].tolist())
    kept.sort(key=lambda i: -confidences[i])
    height, width = image_shape[:2]
    left, top = padding
    detections = []
    for i in kept[:max_det]:
        x,y,w,h = xywh[i]
        x1,x2 = np.clip([(x-left)/ratio,(x+w-left)/ratio],0,width)
        y1,y2 = np.clip([(y-top)/ratio,(y+h-top)/ratio],0,height)
        if x2 > x1 and y2 > y1:
            cid = int(classes[i])
            detections.append(Detection(float(x1),float(y1),float(x2),float(y2),
                                        float(confidences[i]),cid,COCO_CLASSES[cid]))
    return detections
