"""ONNX graph simplification module using onnxslim."""
from pathlib import Path
import onnxslim

def simplify_onnx_graph(input_onnx, output_onnx):
    """Simplify ONNX compute graph using onnxslim.
    
    Performs constant folding, operator fusion (Conv+BN+SiLU),
    and dead node elimination without modifying model parameters or losing accuracy.
    """
    inp_p = Path(input_onnx).resolve()
    out_p = Path(output_onnx).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    if not inp_p.exists():
        raise FileNotFoundError(f"Input ONNX file not found: {inp_p}")
        
    print(f"[Graph Slim] Simplifying compute graph: {inp_p.name} -> {out_p.name}...")
    onnxslim.slim(str(inp_p), str(out_p))
    
    size_mb = out_p.stat().st_size / (1024 * 1024)
    print(f"[Graph Slim] Successfully saved simplified graph: {out_p.name} ({size_mb:.2f} MB)")
    return out_p
