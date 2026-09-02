"""
Model Training Module
Industrial Defect Detection System using Ultralytics YOLO

This script:
1. Loads pretrained YOLO weights (YOLO11n / YOLOv8n) for transfer learning.
2. Automatically detects hardware acceleration (NVIDIA CUDA GPU or multi-threaded CPU).
3. Trains on the industrial surface defect dataset using configured hyperparameters.
4. Automatically exports performance curves and weights to results/ and results/graphs/.
5. Produces formatted execution logs and student-friendly summaries.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("[!] Ultralytics is not installed. Please run: pip install ultralytics")
    sys.exit(1)


def check_environment():
    """Checks and reports system hardware and software capabilities."""
    print("\n" + "="*60)
    print("           COMPUTING ENVIRONMENT & HARDWARE STATUS")
    print("="*60)
    print(f"* PyTorch Version   : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"* CUDA Accelerated  : {'YES (GPU Enabled)' if cuda_available else 'NO (Running on CPU)'}")
    if cuda_available:
        print(f"* GPU Device        : {torch.cuda.get_device_name(0)}")
        print(f"* VRAM Available    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("* Compute Note      : Standard CPU mode active. Using optimized thread scheduling.")
    print("="*60 + "\n")
    return "0" if cuda_available else "cpu"


def train_defect_detector(
    data_yaml: str = "data.yaml",
    model_name: str = "yolo11n.pt",
    epochs: int = 30,
    batch_size: int = 16,
    img_size: int = 640,
    lr0: float = 0.01,
    project_name: str = "runs/detect",
    run_name: str = "train",
    seed: int = 42
):
    """
    Executes YOLO model training with transfer learning.
    """
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / data_yaml
    
    if not data_path.exists():
        raise FileNotFoundError(f"[!] Dataset config not found at: {data_path}. Run prepare_dataset.py first.")
        
    device = check_environment()
    
    # Adjust default batch size if running on CPU to maintain responsiveness
    if device == "cpu" and batch_size > 8:
        print(f"[*] Adjusted batch size to 8 for smoother CPU training.")
        batch_size = 8

    print(f"[*] Initializing YOLO architecture with pretrained weights: '{model_name}'...")
    try:
        model = YOLO(model_name)
    except Exception:
        # Fallback to YOLOv8n if YOLO11n weights download has network issue
        print(f"[*] Falling back to 'yolov8n.pt'...")
        model = YOLO("yolov8n.pt")

    print(f"[*] Starting model training...")
    print(f"    - Epochs       : {epochs}")
    print(f"    - Batch Size   : {batch_size}")
    print(f"    - Image Size   : {img_size}x{img_size}")
    print(f"    - Initial LR   : {lr0}")
    print(f"    - Device       : {device}")
    print(f"    - Dataset YAML : {data_path}")

    # Train model
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr0,
        device=device,
        project=str(project_root / project_name),
        name=run_name,
        seed=seed,
        exist_ok=True,
        plots=True,
        verbose=True,
        save=True,
        val=True
    )

    print("\n[OK] Training sequence completed successfully!")
    
    # Locate and copy best weights and visual graphs to results directory
    train_output_dir = project_root / project_name / run_name
    best_weight_src = train_output_dir / "weights" / "best.pt"
    results_dir = project_root / "results"
    graphs_dir = results_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    if best_weight_src.exists():
        shutil.copy2(best_weight_src, results_dir / "best_model.pt")
        print(f"[OK] Saved best model weights to: {results_dir / 'best_model.pt'}")
        
    # Copy training curves and plots for assignment documentation
    plot_files = ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", 
                  "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png"]
    for pf in plot_files:
        src_file = train_output_dir / pf
        if src_file.exists():
            shutil.copy2(src_file, graphs_dir / pf)
            print(f"[OK] Exported plot: {graphs_dir / pf}")
            
    print("\n" + "="*60)
    print("              TRAINING ARTIFACTS READY")
    print("="*60)
    print(f"* Best Weights : {results_dir / 'best_model.pt'}")
    print(f"* Graphs Saved : {graphs_dir}")
    print("="*60 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Defect Detector")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model checkpoint")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    args = parser.parse_args()

    train_defect_detector(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        model_name=args.model,
        lr0=args.lr
    )
