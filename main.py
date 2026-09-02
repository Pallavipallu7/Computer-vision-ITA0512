"""
Main Execution & Pipeline Orchestrator
Industrial Defect Detection System

Entry point for running all stages of the computer vision project:
1. Dataset Preparation & Verification
2. Image Preprocessing & Enhancement
3. YOLO Model Training
4. Inference & Bounding Box Prediction
5. Quantitative Evaluation & Metrics
6. Traditional Computer Vision Baseline Comparison
7. Streamlit Web Interface Launch
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.prepare_dataset import setup_dataset, verify_dataset_integrity, print_dataset_statistics
from src.preprocess import generate_preprocessing_visual_report
from src.train import train_defect_detector
from src.predict import predict_batch, predict_image
from src.evaluate import evaluate_defect_detector
from src.traditional_cv import compare_traditional_vs_deep_learning, print_engineering_comparison_table


def print_banner():
    print("""
========================================================================
   🏭 INDUSTRIAL DEFECT DETECTION SYSTEM USING COMPUTER VISION (YOLO)
   Autonomous Quality Control Pipeline for High-Speed Manufacturing
========================================================================
    """)


def interactive_menu():
    """Displays interactive CLI menu for students."""
    print_banner()
    while True:
        print("\nPlease select a pipeline stage to execute:")
        print("  [1] Prepare & Verify Dataset (NEU-DET Benchmark)")
        print("  [2] Run Preprocessing Pipeline & Save Visual Stages")
        print("  [3] Train YOLO Defect Detector (Transfer Learning)")
        print("  [4] Run Defect Detection Inference (Batch / Single)")
        print("  [5] Evaluate Model Performance (mAP, Precision, Recall, F1)")
        print("  [6] Compare Traditional CV vs. Deep Learning Baseline")
        print("  [7] Launch Streamlit Interactive Web Application")
        print("  [8] Run Complete End-to-End Pipeline (Stages 1 -> 6)")
        print("  [0] Exit")
        
        choice = input("\nEnter choice [0-8]: ").strip()
        
        if choice == "1":
            setup_dataset(force_recreate=False)
        elif choice == "2":
            sample_imgs = list((PROJECT_ROOT / "dataset" / "images" / "train").glob("*.jpg"))
            if not sample_imgs:
                setup_dataset()
                sample_imgs = list((PROJECT_ROOT / "dataset" / "images" / "train").glob("*.jpg"))
            out_graph = str(PROJECT_ROOT / "results" / "graphs" / "preprocessing_pipeline_stages.png")
            generate_preprocessing_visual_report(str(sample_imgs[0]), out_graph)
        elif choice == "3":
            epochs_in = input("Enter number of epochs (default: 30): ").strip()
            epochs = int(epochs_in) if epochs_in.isdigit() else 30
            train_defect_detector(epochs=epochs)
        elif choice == "4":
            src_in = input("Enter image or directory path (default: dataset/images/test): ").strip()
            src = src_in if src_in else "dataset/images/test"
            target = Path(src)
            if target.is_dir():
                predict_batch(source_dir=src)
            else:
                predict_image(image_path=src)
        elif choice == "5":
            evaluate_defect_detector(split="test")
        elif choice == "6":
            test_imgs = list((PROJECT_ROOT / "dataset" / "images" / "test").glob("*.jpg"))
            if test_imgs:
                out_chart = str(PROJECT_ROOT / "results" / "graphs" / "traditional_vs_yolo_comparison.png")
                compare_traditional_vs_deep_learning(str(test_imgs[0]), out_chart)
            else:
                print("[!] No test images found. Please run Option 1 first.")
        elif choice == "7":
            print("[*] Launching Streamlit Web App (press Ctrl+C in terminal to stop)...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        elif choice == "8":
            print("\n[*] RUNNING FULL END-TO-END AUTOMATED PIPELINE...")
            setup_dataset(force_recreate=False)
            
            sample_imgs = list((PROJECT_ROOT / "dataset" / "images" / "train").glob("*.jpg"))
            out_prep = str(PROJECT_ROOT / "results" / "graphs" / "preprocessing_pipeline_stages.png")
            generate_preprocessing_visual_report(str(sample_imgs[0]), out_prep)
            
            train_defect_detector(epochs=15, batch_size=8)
            evaluate_defect_detector(split="test")
            predict_batch(source_dir="dataset/images/test")
            
            test_imgs = list((PROJECT_ROOT / "dataset" / "images" / "test").glob("*.jpg"))
            out_comp = str(PROJECT_ROOT / "results" / "graphs" / "traditional_vs_yolo_comparison.png")
            compare_traditional_vs_deep_learning(str(test_imgs[0]), out_comp)
            print("\n[OK] FULL PIPELINE EXECUTION COMPLETE! All results stored in results/")
        elif choice == "0":
            print("Exiting. Good luck with your Computer Vision Assignment!")
            break
        else:
            print("[!] Invalid option. Please enter a number between 0 and 8.")


def main():
    parser = argparse.ArgumentParser(description="Industrial Defect Detection Master Orchestrator")
    parser.add_argument("--prepare", action="store_true", help="Prepare and verify dataset")
    parser.add_argument("--preprocess", action="store_true", help="Generate preprocessing pipeline visual report")
    parser.add_argument("--train", action="store_true", help="Train YOLO model")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--predict", action="store_true", help="Run defect prediction inference")
    parser.add_argument("--source", type=str, default="dataset/images/test", help="Source for prediction")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on test split")
    parser.add_argument("--compare", action="store_true", help="Compare Traditional CV vs YOLO")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web interface")
    parser.add_argument("--full-pipeline", action="store_true", help="Run complete end-to-end pipeline")
    
    args = parser.parse_args()

    if args.prepare:
        setup_dataset(force_recreate=True)
    elif args.preprocess:
        sample_imgs = list((PROJECT_ROOT / "dataset" / "images" / "train").glob("*.jpg"))
        if not sample_imgs:
            setup_dataset()
            sample_imgs = list((PROJECT_ROOT / "dataset" / "images" / "train").glob("*.jpg"))
        out_graph = str(PROJECT_ROOT / "results" / "graphs" / "preprocessing_pipeline_stages.png")
        generate_preprocessing_visual_report(str(sample_imgs[0]), out_graph)
    elif args.train:
        train_defect_detector(epochs=args.epochs)
    elif args.predict:
        target = Path(args.source)
        if target.is_dir():
            predict_batch(source_dir=args.source)
        else:
            predict_image(image_path=args.source)
    elif args.evaluate:
        evaluate_defect_detector(split="test")
    elif args.compare:
        test_imgs = list((PROJECT_ROOT / "dataset" / "images" / "test").glob("*.jpg"))
        if test_imgs:
            out_chart = str(PROJECT_ROOT / "results" / "graphs" / "traditional_vs_yolo_comparison.png")
            compare_traditional_vs_deep_learning(str(test_imgs[0]), out_chart)
    elif args.app:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    elif args.full_pipeline:
        setup_dataset(force_recreate=False)
        sample_imgs = list((PROJECT_ROOT / "dataset" / "images" / "train").glob("*.jpg"))
        out_prep = str(PROJECT_ROOT / "results" / "graphs" / "preprocessing_pipeline_stages.png")
        generate_preprocessing_visual_report(str(sample_imgs[0]), out_prep)
        train_defect_detector(epochs=args.epochs, batch_size=8)
        evaluate_defect_detector(split="test")
        predict_batch(source_dir="dataset/images/test")
        test_imgs = list((PROJECT_ROOT / "dataset" / "images" / "test").glob("*.jpg"))
        out_comp = str(PROJECT_ROOT / "results" / "graphs" / "traditional_vs_yolo_comparison.png")
        compare_traditional_vs_deep_learning(str(test_imgs[0]), out_comp)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
