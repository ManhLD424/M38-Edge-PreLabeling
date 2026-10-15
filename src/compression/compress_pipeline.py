"""Unified compression pipeline for YOLO26 architectures."""
from pathlib import Path
from .export_onnx import export_base_onnx
from .graph_slim import simplify_onnx_graph
from .quantize_fp16 import export_fp16_onnx
from .quantize_int8 import quantize_onnx_int8

def run_compression_pipeline(model_tag, pt_path, output_dir, calibration_dir=None):
    """Run full 4-tier compression workflow for a given model.
    
    1. Base ONNX FP32
    2. ONNX Graph Simplification (onnxslim)
    3. ONNX FP16 Half-Precision (Production Recommendation)
    4. ONNX INT8 PTQ (Optional, if calibration images available)
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fp32_path = out_dir / f"{model_tag}_fp32.onnx"
    slim_path = out_dir / f"{model_tag}_slim.onnx"
    fp16_path = out_dir / f"{model_tag}_fp16.onnx"
    int8_path = out_dir / f"{model_tag}_int8_ptq.onnx"
    
    results = {}
    
    # 1. Base ONNX
    results["fp32"] = export_base_onnx(pt_path, fp32_path)
    
    # 2. Simplified ONNX
    results["slim"] = simplify_onnx_graph(fp32_path, slim_path)
    
    # 3. FP16 ONNX
    results["fp16"] = export_fp16_onnx(pt_path, fp16_path)
    
    # 4. INT8 ONNX
    if calibration_dir and Path(calibration_dir).exists():
        results["int8"] = quantize_onnx_int8(slim_path, int8_path, calibration_dir)
    else:
        print("[Pipeline] No calibration directory provided; skipping INT8 PTQ step.")
        
    return results
