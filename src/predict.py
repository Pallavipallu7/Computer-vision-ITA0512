"""
Inference & Defect Detection Prediction Module
Industrial Defect Detection System using YOLO

This script:
1. Performs defect detection on a single image, an entire image directory, or video.
2. Extracts bounding boxes, defect classifications, and confidence scores.
3. Renders high-visibility annotations and quality inspection verdicts (PASS / REJECT).
4. Saves predicted images to results/predictions/ and results/sample_outputs/.
5. Outputs formatted logs for assignment documentation.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[!] Ultralytics is not installed. Please run: pip install ultralytics")
    sys.exit(1)

# Color palette for defect classes (BGR format for OpenCV)
CLASS_COLORS = {
    "crazing": (0, 165, 255),       # Orange
    "inclusion": (0, 0, 255),       # Red
    "patches": (255, 0, 255),       # Magenta
    "pitted_surface": (255, 255, 0),# Cyan
    "rolled-in_scale": (0, 255, 255),# Yellow
    "scratches": (0, 255, 0)        # Green
}


def get_model_path(custom_path: str = None) -> Path:
    """Finds available trained model weights or pretrained base weights."""
    project_root = Path(__file__).resolve().parent.parent
    
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
        
    candidates = [
        project_root / "results" / "best_model.pt",
        project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
        project_root / "yolo11n.pt",
        project_root / "yolov8n.pt"
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
            
    # If no local weights exist, default to downloading base weights
    return Path("yolo11n.pt")


def draw_custom_annotations(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Renders clean, high-contrast industrial inspection overlays with bounding boxes,
    class labels, confidence percentages, and inspection status banner.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    # Render each detection box
    for det in detections:
        box = det["box"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        
        x1, y1, x2, y2 = [int(v) for v in box]
        color = CLASS_COLORS.get(cls_name, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
        
        # Label badge background
        label_text = f"{cls_name.upper()} {conf:.1%}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        
        # Ensure label sits nicely above or inside box
        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = y1
        badge_x2 = min(w, x1 + tw + 10)
        
        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(annotated, label_text, (x1 + 5, badge_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Top Status Banner: Quality Inspection Verdict
    has_defects = len(detections) > 0
    banner_color = (0, 0, 180) if has_defects else (0, 150, 0)
    verdict_text = f"INSPECTION: REJECT ({len(detections)} DEFECTS)" if has_defects else "INSPECTION: PASS (NO DEFECTS)"
    
    cv2.rectangle(annotated, (0, 0), (w, 38), (30, 30, 30), -1)
    cv2.rectangle(annotated, (0, 35), (w, 38), banner_color, -1)
    cv2.putText(annotated, verdict_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255) if not has_defects else (100, 100, 255), 2, cv2.LINE_AA)
    
    return annotated


def predict_image(
    image_path: str,
    model_path: str = None,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    save_dir: str = "results/predictions",
    loaded_model = None
) -> dict:
    """
    Runs defect detection on a single image and saves the annotated result.
    """
    project_root = Path(__file__).resolve().parent.parent
    
    if loaded_model is not None:
        model = loaded_model
    else:
        actual_model_path = get_model_path(model_path)
        print(f"\n[*] Loading model from: {actual_model_path}")
        model = YOLO(str(actual_model_path))
    
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"[!] Image not found: {image_path}")
        
    img = cv2.imread(str(img_p))
    if img is None:
        raise ValueError(f"[!] Unable to decode image at: {image_path}")
        
    # Perform YOLO inference
    results = model.predict(
        source=img,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False
    )[0]
    
    # Extract structured detection results
    detections = []
    names = results.names
    
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = names.get(cls_id, f"class_{cls_id}")
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        
        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": round(conf, 4),
            "box": [round(coord, 2) for coord in xyxy]
        })
        
    # Draw custom industrial annotations
    annotated_img = draw_custom_annotations(img, detections)
    
    # Save output
    output_dir = project_root / save_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"pred_{img_p.name}"
    cv2.imwrite(str(out_file), annotated_img)
    
    # Print formatted output for assignment verification
    print("\n" + "="*60)
    print(f"       INSPECTION REPORT: {img_p.name}")
    print("="*60)
    if detections:
        print(f"* Verdict       : DEFECT DETECTED (REJECT)")
        print(f"* Total Defects : {len(detections)}")
        print("-" * 60)
        for idx, d in enumerate(detections, start=1):
            print(f"  [{idx}] Defect Type : {d['class_name']}")
            print(f"      Confidence  : {d['confidence']:.2%}")
            print(f"      Bounding Box: {d['box']}")
    else:
        print("* Verdict       : PASS (No Defects Detected)")
    print(f"* Saved Output  : {out_file}")
    print("="*60 + "\n")
    
    return {
        "image_path": str(img_p),
        "output_path": str(out_file),
        "detections": detections,
        "verdict": "REJECT" if len(detections) > 0 else "PASS"
    }


def predict_batch(
    source_dir: str = "dataset/images/test",
    model_path: str = None,
    conf_thresh: float = 0.25,
    save_dir: str = "results/predictions"
):
    """
    Runs batch inference on all images in a directory and populates sample outputs.
    """
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / source_dir
    
    if not src_path.exists():
        print(f"[!] Directory not found: {src_path}")
        return
        
    img_files = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png")) + list(src_path.glob("*.bmp"))
    if not img_files:
        print(f"[!] No image files found in {src_path}")
        return
        
    print(f"\n[*] Running batch defect inspection on {len(img_files)} images from: {src_path}")
    actual_model_path = get_model_path(model_path)
    print(f"[*] Loading YOLO detector from: {actual_model_path}")
    model = YOLO(str(actual_model_path))
    
    sample_outputs_dir = project_root / "results" / "sample_outputs"
    sample_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    summary_counts = {"PASS": 0, "REJECT": 0}
    
    for idx, img_path in enumerate(img_files, start=1):
        res = predict_image(
            str(img_path),
            model_path=model_path,
            conf_thresh=conf_thresh,
            save_dir=save_dir,
            loaded_model=model
        )
        summary_counts[res["verdict"]] += 1
        
        # Copy first few sample detections to results/sample_outputs/
        if idx <= 6:
            dst_sample = sample_outputs_dir / f"sample_{idx}_{img_path.name}"
            shutil.copy2(res["output_path"], dst_sample)
            
    print(f"\n[OK] Batch Inspection Finished!")
    print(f"* Total Processed : {len(img_files)}")
    print(f"* Accepted (PASS) : {summary_counts['PASS']}")
    print(f"* Rejected (DEFECT): {summary_counts['REJECT']}")
    print(f"* Results Saved In: {project_root / save_dir}")
    print(f"* Samples Saved In: {sample_outputs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Industrial Defect Predictor")
    parser.add_argument("--source", type=str, default="dataset/images/test", help="Path to image or directory")
    parser.add_argument("--model", type=str, default=None, help="Path to custom model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold for NMS")
    args = parser.parse_args()

    target = Path(args.source)
    if target.is_dir():
        predict_batch(args.source, model_path=args.model, conf_thresh=args.conf)
    else:
        predict_image(args.source, model_path=args.model, conf_thresh=args.conf, iou_thresh=args.iou)
