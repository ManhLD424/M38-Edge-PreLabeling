"""Batch pre-labeling tool for edge annotator workstations."""
import os
import cv2
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from .engine import EdgePreLabelEngine, InferenceResult
from .exporter import export_to_yolo, export_to_coco

COLOR_APPROVED = (0, 200, 0)    # Green for auto-accepted
COLOR_REVIEW = (0, 140, 255)    # Orange for review needed

def draw_annotations(img, detections):
    """Draw bounding boxes with confidence status colors."""
    vis = img.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
        color = COLOR_APPROVED if d.status == "AUTO_APPROVED" else COLOR_REVIEW
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f} [{d.status[:4]}]"
        cv2.putText(vis, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis

def run_batch_prelabel(
    model_path: str,
    source_dir: str,
    output_dir: str,
    export_format: str = "yolo",
    save_visuals: bool = False,
    conf_thresh: float = 0.25,
    auto_accept_conf: float = 0.80,
    device: str = "cpu"
):
    """Process all images in source_dir and export pre-labels."""
    src_p = Path(source_dir).resolve()
    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    
    labels_out = out_p / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)
    
    if save_visuals:
        vis_out = out_p / "visuals"
        vis_out.mkdir(parents=True, exist_ok=True)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(src_p.glob(ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"[Pre-Label] No images found in {src_p}")
        return []

    print(f"\n==================================================")
    print(f"      STARTING EDGE PRE-LABELING PIPELINE         ")
    print(f"==================================================")
    print(f"Model: {Path(model_path).name}")
    print(f"Target Images: {len(image_paths)}")
    print(f"Device: {device.upper()}")
    print(f"Auto-Accept Threshold: {auto_accept_conf}")
    print(f"Output Directory: {out_p}\n")

    engine = EdgePreLabelEngine(
        model_path=model_path,
        conf_thresh=conf_thresh,
        auto_accept_conf=auto_accept_conf,
        device=device
    )

    all_results = []
    total_approved = 0
    total_review = 0
    latencies = []

    for img_path in tqdm(image_paths, desc="Pre-labeling"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        res = engine.predict(img, image_path=str(img_path))
        all_results.append(res)
        latencies.append(res.latency_ms)
        total_approved += res.auto_approved_count
        total_review += res.review_needed_count

        # Export individual YOLO text file
        if export_format in ("yolo", "both"):
            txt_path = labels_out / f"{img_path.stem}.txt"
            export_to_yolo(res, str(txt_path), include_confidence=True)

        # Save visual comparison
        if save_visuals:
            annotated = draw_annotations(img, res.detections)
            cv2.imwrite(str(vis_out / f"{img_path.name}"), annotated)

    # Export COCO format if requested
    if export_format in ("coco", "both"):
        coco_path = out_p / "annotations_coco.json"
        export_to_coco(all_results, str(coco_path))

    # Summary statistics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    total_boxes = total_approved + total_review
    savings_pct = (total_approved / total_boxes * 100.0) if total_boxes > 0 else 0.0

    print("\n==================================================")
    print("             PRE-LABELING REPORT SUMMARY          ")
    print("==================================================")
    print(f"Processed Images:     {len(all_results)}")
    print(f"Total Bounding Boxes: {total_boxes}")
    print(f"Auto-Approved (>=0.8): {total_approved} ({savings_pct:.1f}% annotation time saved)")
    print(f"Review Flagged (<0.8): {total_review}")
    print(f"Average Latency:      {avg_latency:.2f} ms")
    print(f"Throughput:           {avg_fps:.1f} FPS")
    print(f"Outputs saved to:     {out_p}")
    print("==================================================\n")

    return all_results
