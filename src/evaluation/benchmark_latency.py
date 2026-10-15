"""Latency and throughput profiling suite."""
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

def benchmark_onnx_latency(model_path, iterations=50, warmup=10, device="cpu"):
    """Benchmark ONNX model latency with detailed statistical metrics (P50, P90, P99)."""
    m_path = Path(model_path).resolve()
    if not m_path.exists():
        raise FileNotFoundError(f"Model not found: {m_path}")

    providers = ["CPUExecutionProvider"]
    if device.lower() in ("gpu", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(str(m_path), providers=providers)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_dtype = np.float16 if "float16" in input_info.type else np.float32
    
    # Standard dummy input: (1, 3, 640, 640)
    dummy_input = np.random.randn(1, 3, 640, 640).astype(input_dtype)

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    # Benchmark runs
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    mean_lat = float(np.mean(times))
    std_lat = float(np.std(times))
    p50_lat = float(np.percentile(times, 50))
    p90_lat = float(np.percentile(times, 90))
    p99_lat = float(np.percentile(times, 99))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    result = {
        "model": m_path.name,
        "device": device.upper(),
        "iterations": iterations,
        "mean_ms": round(mean_lat, 2),
        "std_ms": round(std_lat, 2),
        "p50_ms": round(p50_lat, 2),
        "p90_ms": round(p90_lat, 2),
        "p99_ms": round(p99_lat, 2),
        "fps": round(fps, 1),
    }

    print(f"\n[Latency Benchmark] {m_path.name} on {device.upper()}:")
    print(f"  -> Mean Latency: {result['mean_ms']} ± {result['std_ms']} ms")
    print(f"  -> P50 / P90 / P99: {result['p50_ms']} / {result['p90_ms']} / {result['p99_ms']} ms")
    print(f"  -> Throughput:   {result['fps']} FPS")
    return result
