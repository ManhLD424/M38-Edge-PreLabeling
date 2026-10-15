"""Annotation exporters for YOLO and COCO formats."""
import json
from pathlib import Path
from typing import List
from .engine import InferenceResult, BDD100K_CLASSES

def export_to_yolo(result: InferenceResult, output_txt_path: str, include_confidence: bool = False):
    """Export single image inference result to YOLO format (.txt).
    
    Format per line: <class_id> <x_center> <y_center> <width> <height> [<confidence>]
    """
    out_p = Path(output_txt_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for d in result.detections:
        x, y, w, h = d.bbox_xywhn
        if include_confidence:
            lines.append(f"{d.class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {d.confidence:.4f}")
        else:
            lines.append(f"{d.class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    with open(out_p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_p

def export_to_coco(results: List[InferenceResult], output_json_path: str, categories: List[str] = None):
    """Export list of inference results to standard COCO JSON format."""
    out_p = Path(output_json_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if categories is None:
        categories = BDD100K_CLASSES

    coco_data = {
        "info": {
            "description": "BDD100K Edge Pre-Labeling Output",
            "version": "1.0",
            "year": 2026,
            "contributor": "Team Sakana (Le Duc Manh - 02122, Pham Hoang Anh - 02128)",
        },
        "licenses": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "traffic"}
            for i, name in enumerate(categories)
        ],
        "images": [],
        "annotations": []
    }

    ann_id = 1
    for img_id, res in enumerate(results, start=1):
        file_name = Path(res.image_path).name if res.image_path else f"image_{img_id:06d}.jpg"
        h, w = res.image_shape
        coco_data["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": w,
            "height": h
        })

        for d in res.detections:
            x1, y1, x2, y2 = [float(v) for v in d.bbox_xyxy]
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            coco_data["annotations"].append({
                "id": int(ann_id),
                "image_id": int(img_id),
                "category_id": int(d.class_id),
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "score": round(float(d.confidence), 4),
                "status": str(d.status)
            })
            ann_id += 1

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)
    return out_p
