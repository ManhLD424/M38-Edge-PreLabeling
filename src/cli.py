"""Unified Command Line Interface for M38: Edge Pre-Labeling."""
import argparse
import sys
from pathlib import Path
from src.data.download_dataset import download_models, download_val_dataset
from src.compression.compress_pipeline import run_compression_pipeline
from src.prelabel.annotator import run_batch_prelabel
from src.evaluation.benchmark_latency import benchmark_onnx_latency
from src.evaluation.eval_map import evaluate_accuracy

def main():
    parser = argparse.ArgumentParser(
        description="M38: Model Compression for Edge Pre-Labeling on BDD100K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Download
    p_dl = subparsers.add_parser("download", help="Download baseline checkpoints or validation subset")
    p_dl.add_argument("--type", choices=["models", "data", "all"], default="all", help="Target assets to download")
    p_dl.add_argument("--output", default="data/bdd100k_val", help="Output directory for dataset")

    # 2. Compress
    p_comp = subparsers.add_parser("compress", help="Compress a PyTorch YOLO model to ONNX FP16 / INT8")
    p_comp.add_argument("--tag", required=True, help="Model tag (e.g. yolo26n, yolo26s, yolo26m)")
    p_comp.add_argument("--weights", required=True, help="Path to PyTorch weights (.pt)")
    p_comp.add_argument("--output", default="models/compressed", help="Output directory for compressed models")
    p_comp.add_argument("--calib-dir", default=None, help="Directory of calibration images for INT8 PTQ")

    # 3. Prelabel
    p_label = subparsers.add_parser("prelabel", help="Run batch edge pre-labeling on image folder")
    p_label.add_argument("--model", required=True, help="Path to compressed ONNX model")
    p_label.add_argument("--source", required=True, help="Input directory containing images")
    p_label.add_argument("--output", default="prelabel_output", help="Directory to save pre-labels")
    p_label.add_argument("--format", choices=["yolo", "coco", "both"], default="yolo", help="Export format")
    p_label.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Compute device")
    p_label.add_argument("--auto-conf", type=float, default=0.80, help="Confidence threshold for auto-approval")
    p_label.add_argument("--visuals", action="store_true", help="Save annotated visualization images")

    # 4. Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark latency and accuracy of a model")
    p_bench.add_argument("--model", required=True, help="Path to ONNX or PyTorch model")
    p_bench.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Compute device")
    p_bench.add_argument("--iters", type=int, default=50, help="Number of benchmark iterations")
    p_bench.add_argument("--eval-yaml", default=None, help="Optional data YAML for mAP accuracy evaluation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "download":
        if args.type in ("models", "all"):
            download_models("models/baseline")
        if args.type in ("data", "all"):
            download_val_dataset(args.output, num_samples=200)

    elif args.command == "compress":
        run_compression_pipeline(
            model_tag=args.tag,
            pt_path=args.weights,
            output_dir=args.output,
            calibration_dir=args.calib_dir
        )

    elif args.command == "prelabel":
        run_batch_prelabel(
            model_path=args.model,
            source_dir=args.source,
            output_dir=args.output,
            export_format=args.format,
            save_visuals=args.visuals,
            auto_accept_conf=args.auto_conf,
            device=args.device
        )

    elif args.command == "benchmark":
        m_path = Path(args.model)
        if m_path.suffix.lower() == ".onnx":
            benchmark_onnx_latency(str(m_path), iterations=args.iters, device=args.device)
        if args.eval_yaml:
            evaluate_accuracy(str(m_path), data_yaml=args.eval_yaml, device=args.device)

if __name__ == "__main__":
    main()
