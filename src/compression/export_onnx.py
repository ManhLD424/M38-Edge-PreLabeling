"""Base ONNX export module."""
from pathlib import Path
import shutil
from ultralytics import YOLO

def export_base_onnx(pt_path, output_path, imgsz=640, opset=17):
    """Export PyTorch YOLO checkpoint to standard ONNX FP32 format.
    
    Args:
        pt_path: Path to PyTorch model (.pt)
        output_path: Target path for the exported ONNX model
        imgsz: Image resolution (default 640)
        opset: ONNX operator set version (default 17)
    """
    pt_p = Path(pt_path).resolve()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    if not pt_p.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pt_p}")
        
    print(f"[ONNX Export] Exporting {pt_p.name} to ONNX FP32 (opset {opset})...")
    model = YOLO(str(pt_p))
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=False,
        opset=opset
    )
    shutil.move(exported_file, out_p)
    size_mb = out_p.stat().st_size / (1024 * 1024)
    print(f"[ONNX Export] Successfully saved {out_p.name} ({size_mb:.2f} MB)")
    return out_p
