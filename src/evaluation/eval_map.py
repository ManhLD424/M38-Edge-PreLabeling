"""Validation accuracy evaluation module (mAP@50 and mAP@50:95)."""
from pathlib import Path
import torch
from ultralytics import YOLO

def evaluate_accuracy(model_path, data_yaml, imgsz=640, device="cpu", batch=4):
    """Evaluate YOLO or ONNX model accuracy on validation dataset."""
    m_path = Path(model_path).resolve()
    yaml_path = Path(data_yaml).resolve()

    if not m_path.exists():
        raise FileNotFoundError(f"Model path not found: {m_path}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {yaml_path}")

    dev = 0 if (device.lower() in ("gpu", "cuda") and torch.cuda.is_available()) else "cpu"
    print(f"\n[Accuracy Eval] Running validation on {m_path.name} (device={dev})...")

    model = YOLO(str(m_path), task="detect")
    results = model.val(
        data=str(yaml_path),
        imgsz=imgsz,
        batch=batch,
        device=dev,
        plots=False,
        verbose=False
    )

    map50 = results.box.map50 * 100
    map50_95 = results.box.map * 100
    precision = results.box.mp * 100
    recall = results.box.mr * 100

    summary = {
        "model": m_path.name,
        "mAP50": round(map50, 2),
        "mAP50_95": round(map50_95, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }

    print(f"[Accuracy Eval] Results for {m_path.name}:")
    print(f"  -> mAP@50:    {summary['mAP50']}%")
    print(f"  -> mAP@50:95: {summary['mAP50_95']}%")
    print(f"  -> Precision: {summary['precision']}%")
    print(f"  -> Recall:    {summary['recall']}%")
    return summary
