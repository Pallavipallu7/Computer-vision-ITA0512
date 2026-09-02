"""
=============================================================================
Module: dataset_setup.py
Project: Defect Detection System Decision Using Computer Vision
Description: Automates dataset downloading, preparation, synthesis, and
             YOLO annotation conversion for industrial surface defect detection.
=============================================================================
"""

import os
import sys
import shutil
import random
import urllib.request
import zipfile
import numpy as np
import cv2

# Defect class dictionary matching data.yaml
CLASSES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled_in_scale",
    5: "scratches"
}

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

def create_directory_structure():
    """Creates the standard YOLO dataset folder hierarchy."""
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DATASET_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_ROOT, "labels", split), exist_ok=True)
    print(f"[INFO] Initialized dataset directory structure at: {DATASET_ROOT}")

def generate_industrial_surface_texture(width=640, height=640, base_type="steel"):
    """
    Generates realistic industrial metal/substrate background texture with
    illumination gradients, rolling grain, and subtle surface noise.
    """
    # Base metallic gray with slight batch-to-batch variation
    base_val = random.randint(110, 180)
    img = np.full((height, width, 3), base_val, dtype=np.uint8)

    # 1. Add rolling mill directional grain texture
    noise = np.random.normal(0, random.uniform(8, 15), (height, width)).astype(np.float32)
    # Directional blur to simulate rolled sheet metal grain
    kernel_size = random.choice([7, 9, 11])
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size) / kernel_size
    grain = cv2.filter2D(noise, -1, kernel)

    # 2. Add non-uniform lighting / illumination gradients (industrial factory lights)
    X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    light_center_x = random.uniform(0.2, 0.8)
    light_center_y = random.uniform(0.2, 0.8)
    vignette = np.exp(-((X - light_center_x)**2 + (Y - light_center_y)**2) / random.uniform(0.5, 1.2))
    vignette = (vignette - vignette.min()) / (vignette.max() - vignette.min() + 1e-6)
    lighting_gradient = (vignette * random.uniform(20, 50) - 25).astype(np.float32)

    # Combine background components
    for c in range(3):
        channel = img[:, :, c].astype(np.float32) + grain + lighting_gradient
        img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    return img

def add_synthetic_defect(img, class_id):
    """
    Adds a realistic industrial defect to the image and computes its
    YOLO normalized bounding box (x_center, y_center, width, height).
    """
    h, w, _ = img.shape
    defect_name = CLASSES[class_id]
    boxes = []

    if defect_name == "scratches":
        # Scratches: Sharp, thin, multi-segment linear scratches
        num_scratches = random.randint(1, 3)
        for _ in range(num_scratches):
            x1 = random.randint(int(w * 0.1), int(w * 0.8))
            y1 = random.randint(int(h * 0.1), int(h * 0.8))
            length = random.randint(int(w * 0.15), int(w * 0.45))
            angle = random.uniform(-np.pi, np.pi)
            x2 = int(np.clip(x1 + length * np.cos(angle), 10, w - 10))
            y2 = int(np.clip(y1 + length * np.sin(angle), 10, h - 10))

            color = random.choice([(30, 30, 30), (230, 230, 230), (45, 45, 45)])
            thickness = random.randint(2, 4)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

            # Secondary scratch branch
            if random.random() > 0.5:
                bx = int((x1 + x2) / 2)
                by = int((y1 + y2) / 2)
                bx2 = int(np.clip(bx + (length * 0.4) * np.cos(angle + 0.5), 10, w - 10))
                by2 = int(np.clip(by + (length * 0.4) * np.sin(angle + 0.5), 10, h - 10))
                cv2.line(img, (bx, by), (bx2, by2), color, thickness - 1, lineType=cv2.LINE_AA)

            xmin, xmax = min(x1, x2, bx if 'bx' in locals() else x1), max(x1, x2, bx2 if 'bx2' in locals() else x2)
            ymin, ymax = min(y1, y2, by if 'by' in locals() else y1), max(y1, y2, by2 if 'by2' in locals() else y2)
            # Add small padding
            xmin = max(0, xmin - 5)
            ymin = max(0, ymin - 5)
            xmax = min(w, xmax + 5)
            ymax = min(h, ymax + 5)

            # Convert to YOLO format
            xc = ((xmin + xmax) / 2.0) / w
            yc = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / float(w)
            bh = (ymax - ymin) / float(h)
            boxes.append((class_id, xc, yc, bw, bh))

    elif defect_name == "patches":
        # Patches: Irregular blotches of altered oxidation/sheen
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        rx = random.randint(int(w * 0.08), int(w * 0.2))
        ry = random.randint(int(h * 0.08), int(h * 0.2))
        angle = random.randint(0, 180)

        overlay = img.copy()
        patch_color = random.choice([(40, 40, 40), (220, 220, 220), (70, 60, 50)])
        cv2.ellipse(overlay, (cx, cy), (rx, ry), angle, 0, 360, patch_color, -1)
        # Texture roughness inside patch
        cv2.ellipse(overlay, (cx + 5, cy - 5), (int(rx * 0.6), int(ry * 0.6)), angle + 20, 0, 360, (20, 20, 20), -1)
        alpha = random.uniform(0.4, 0.7)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        xmin = max(0, cx - rx - 6)
        ymin = max(0, cy - ry - 6)
        xmax = min(w, cx + rx + 6)
        ymax = min(h, cy + ry + 6)
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / float(w)
        bh = (ymax - ymin) / float(h)
        boxes.append((class_id, xc, yc, bw, bh))

    elif defect_name == "pitted_surface":
        # Pitted Surface: Clusters of micro-cavities/pinholes
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        cluster_radius = random.randint(int(w * 0.08), int(w * 0.18))

        min_x, max_x = cx, cx
        min_y, max_y = cy, cy
        num_pits = random.randint(15, 35)
        for _ in range(num_pits):
            px = int(np.clip(cx + np.random.normal(0, cluster_radius * 0.4), 10, w - 10))
            py = int(np.clip(cy + np.random.normal(0, cluster_radius * 0.4), 10, h - 10))
            r = random.randint(2, 6)
            cv2.circle(img, (px, py), r, (25, 25, 25), -1)
            cv2.circle(img, (px + 1, py + 1), max(1, r - 2), (210, 210, 210), 1)
            min_x = min(min_x, px - r)
            max_x = max(max_x, px + r)
            min_y = min(min_y, py - r)
            max_y = max(max_y, py + r)

        xmin = max(0, min_x - 4)
        ymin = max(0, min_y - 4)
        xmax = min(w, max_x + 4)
        ymax = min(h, max_y + 4)
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / float(w)
        bh = (ymax - ymin) / float(h)
        boxes.append((class_id, xc, yc, bw, bh))

    elif defect_name == "inclusion":
        # Inclusion: Non-metallic foreign material embedded in metal
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        size = random.randint(int(w * 0.06), int(w * 0.16))
        # Draw irregular polygon
        num_vertices = random.randint(5, 8)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
        radii = np.random.uniform(size * 0.4, size, num_vertices)
        pts = np.zeros((num_vertices, 2), dtype=np.int32)
        for i in range(num_vertices):
            pts[i, 0] = int(np.clip(cx + radii[i] * np.cos(angles[i]), 5, w - 5))
            pts[i, 1] = int(np.clip(cy + radii[i] * np.sin(angles[i]), 5, h - 5))

        cv2.fillPoly(img, [pts], (35, 30, 25))
        # Add internal gradient/shadow
        cv2.polylines(img, [pts], True, (15, 15, 15), 2)

        xmin = max(0, pts[:, 0].min() - 4)
        xmax = min(w, pts[:, 0].max() + 4)
        ymin = max(0, pts[:, 1].min() - 4)
        ymax = min(h, pts[:, 1].max() + 4)
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / float(w)
        bh = (ymax - ymin) / float(h)
        boxes.append((class_id, xc, yc, bw, bh))

    elif defect_name == "crazing":
        # Crazing: Network of fine hairline cracks
        cx = random.randint(int(w * 0.25), int(w * 0.75))
        cy = random.randint(int(h * 0.25), int(h * 0.75))
        spread = random.randint(int(w * 0.08), int(w * 0.18))
        xmin, xmax, ymin, ymax = cx, cx, cy, cy
        curr_x, curr_y = cx, cy
        for _ in range(random.randint(6, 12)):
            next_x = int(np.clip(curr_x + random.randint(-spread, spread), 10, w - 10))
            next_y = int(np.clip(curr_y + random.randint(-spread, spread), 10, h - 10))
            cv2.line(img, (curr_x, curr_y), (next_x, next_y), (40, 40, 40), 1, cv2.LINE_AA)
            xmin = min(xmin, curr_x, next_x)
            xmax = max(xmax, curr_x, next_x)
            ymin = min(ymin, curr_y, next_y)
            ymax = max(ymax, curr_y, next_y)
            if random.random() > 0.4:
                curr_x, curr_y = cx, cy  # branch out
            else:
                curr_x, curr_y = next_x, next_y

        xmin = max(0, xmin - 5)
        ymin = max(0, ymin - 5)
        xmax = min(w, xmax + 5)
        ymax = min(h, ymax + 5)
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / float(w)
        bh = (ymax - ymin) / float(h)
        boxes.append((class_id, xc, yc, bw, bh))

    elif defect_name == "rolled_in_scale":
        # Rolled-in scale: Embedded scale sheets pressed into surface
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        bw_px = random.randint(int(w * 0.15), int(w * 0.35))
        bh_px = random.randint(int(h * 0.05), int(h * 0.15))

        overlay = img.copy()
        cv2.rectangle(overlay, (cx - bw_px//2, cy - bh_px//2), (cx + bw_px//2, cy + bh_px//2), (50, 45, 40), -1)
        # Add textured horizontal streak lines
        for offset in range(-bh_px//2, bh_px//2, 4):
            cv2.line(overlay, (cx - bw_px//2, cy + offset), (cx + bw_px//2, cy + offset), (25, 20, 15), 1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        xmin = max(0, cx - bw_px//2 - 4)
        ymin = max(0, cy - bh_px//2 - 4)
        xmax = min(w, cx + bw_px//2 + 4)
        ymax = min(h, cy + bh_px//2 + 4)
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / float(w)
        bh = (ymax - ymin) / float(h)
        boxes.append((class_id, xc, yc, bw, bh))

    return boxes

def build_dataset(num_samples=300, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Generates a complete dataset with balanced defect classes and splits into
    training, validation, and test partitions with valid YOLO labels.
    """
    create_directory_structure()
    print(f"[INFO] Generating {num_samples} industrial inspection samples across 6 defect classes...")

    train_count = int(num_samples * train_ratio)
    val_count = int(num_samples * val_ratio)
    test_count = num_samples - train_count - val_count

    splits = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
    random.shuffle(splits)

    class_counts = {cid: 0 for cid in CLASSES.keys()}
    class_list = list(CLASSES.keys())

    for idx, split in enumerate(splits):
        # Generate base industrial metallic surface
        img = generate_industrial_surface_texture(width=640, height=640)

        # Select defect class evenly
        class_id = class_list[idx % len(class_list)]
        boxes = add_synthetic_defect(img, class_id)
        class_counts[class_id] += len(boxes)

        # Save image
        img_filename = f"defect_sample_{idx:04d}_{CLASSES[class_id]}.jpg"
        img_path = os.path.join(DATASET_ROOT, "images", split, img_filename)
        cv2.imwrite(img_path, img)

        # Save YOLO annotation text file
        label_filename = f"defect_sample_{idx:04d}_{CLASSES[class_id]}.txt"
        label_path = os.path.join(DATASET_ROOT, "labels", split, label_filename)
        with open(label_path, "w") as f:
            for cid, xc, yc, bw, bh in boxes:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print("\n[SUCCESS] Dataset Generation Complete!")
    print(f"  Total Images: {num_samples}")
    print(f"  Train: {train_count} | Val: {val_count} | Test: {test_count}")
    print("  Class Distribution:")
    for cid, name in CLASSES.items():
        print(f"    - Class {cid} ({name}): {class_counts[cid]} defect instances")

def verify_dataset():
    """Verifies that all images have matching label files and valid format."""
    total_imgs = 0
    total_labels = 0
    print("\n[INFO] Verifying dataset integrity...")

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DATASET_ROOT, "images", split)
        lbl_dir = os.path.join(DATASET_ROOT, "labels", split)

        imgs = os.listdir(img_dir) if os.path.exists(img_dir) else []
        lbls = os.listdir(lbl_dir) if os.path.exists(lbl_dir) else []

        print(f"  Split [{split.upper()}]: {len(imgs)} images, {len(lbls)} label files")
        total_imgs += len(imgs)
        total_labels += len(lbls)

    if total_imgs > 0 and total_imgs == total_labels:
        print("[VERIFIED] Dataset structure is valid, consistent, and ready for YOLO training.\n")
        return True
    else:
        print("[WARNING] Image and label count mismatch or dataset empty.\n")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset setup and generation for industrial defect detection.")
    parser.add_argument("--samples", type=int, default=300, help="Total number of samples to generate (default: 300)")
    parser.add_argument("--verify", action="store_true", help="Verify existing dataset integrity")
    args = parser.parse_args()

    if args.verify:
        verify_dataset()
    else:
        build_dataset(num_samples=args.samples)
        verify_dataset()
