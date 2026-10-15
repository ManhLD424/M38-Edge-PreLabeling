"""FP16 half-precision quantization module."""
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO

def export_fp16_onnx(pt_path, output_path, imgsz=640, device=0):
    """Export YOLO model to Float16 half-precision ONNX format.
    
    Halves weight storage requirements with virtually zero mAP degradation (<0.3%).
    """
    pt_p = Path(pt_path).resolve()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    if not pt_p.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pt_p}")
        
    dev = 0 if torch.cuda.is_available() else "cpu"
    print(f"[FP16 Export] Quantizing {pt_p.name} to FP16 ONNX on device={dev}...")
    
    model = YOLO(str(pt_p))
    exported_file = model.export(
        format="onnx",
        half=True,
        simplify=True,
        device=dev,
        imgsz=imgsz
    )
    shutil.move(exported_file, out_p)
    size_mb = out_p.stat().st_size / (1024 * 1024)
    print(f"[FP16 Export] Successfully saved FP16 model: {out_p.name} ({size_mb:.2f} MB)")
    return out_p
