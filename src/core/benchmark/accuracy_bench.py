"""
Module Benchmark Accuracy: Đánh giá mAP@50, mAP@50:95, IoU Error và mAP Drop
giữa mô hình gốc (PyTorch FP32 Baseline) và các mô hình nén (ONNX, OpenVINO FP16, INT8).
Tuân thủ chuẩn khoa học của COCO / Pascal VOC metrics.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

from src.core.inference.base import BaseInferenceEngine, Detection


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 hộp (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


class AccuracyBenchmark:
    """Cross-model consistency only. Use COCO evaluation against GT for actual mAP."""
    def __init__(self, iou_thresholds=None):
        self.iou_thresholds = iou_thresholds or [round(x,2) for x in np.arange(.5,1,.05)]

    def evaluate_dataset(self, ref_engine, cand_engine, image_paths):
        if not image_paths:
            raise ValueError('No evaluation images')
        ref_total = cand_total = 0
        matched = {th:0 for th in self.iou_thresholds}
        ious_at_50 = []
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Unreadable evaluation image: {path}')
            refs, candidates = ref_engine.predict(image), cand_engine.predict(image)
            ref_total += len(refs)
            cand_total += len(candidates)
            for th in self.iou_thresholds:
                used = set()
                for candidate in sorted(candidates,key=lambda d:-d.confidence):
                    choices = [(compute_iou(candidate.box_xyxy,ref.box_xyxy),i)
                               for i,ref in enumerate(refs) if i not in used and ref.class_id==candidate.class_id]
                    if not choices:
                        continue
                    overlap,index = max(choices)
                    if overlap >= th:
                        used.add(index)
                        matched[th] += 1
                        if th == .5:
                            ious_at_50.append(overlap)
        matched_50 = matched.get(.5,0)
        precision = matched_50/cand_total if cand_total else 0.
        recall = matched_50/ref_total if ref_total else 0.
        return {'metric_kind':'cross_model_consistency_not_ground_truth_accuracy',
                'total_images':len(image_paths),'ref_total_boxes':ref_total,'cand_total_boxes':cand_total,
                'matched_at_50':matched_50,'precision_at_50_percent':round(precision*100,2),
                'retention_at_50_percent':round(recall*100,2),
                'f1_at_50_percent':round(200*precision*recall/(precision+recall),2) if precision+recall else 0.,
                'mean_matched_iou':round(float(np.mean(ious_at_50)),4) if ious_at_50 else None,
                'matched_counts_by_iou':matched}
