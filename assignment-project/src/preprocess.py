"""
Industrial Image Preprocessing & Enhancement Module
Computer Vision Defect Detection System

This module handles:
1. Illumination variation compensation (CLAHE - Contrast Limited Adaptive Histogram Equalization).
2. High-frequency sensor noise suppression (Bilateral Filtering & Gaussian Denoising).
3. Image aspect ratio preservation and spatial letterboxing.
4. Feature map extraction and edge enhancement.
5. Visual pipeline comparison for assignment documentation.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class IndustrialImagePreprocessor:
    """
    Standardized preprocessing pipeline tailored for industrial surface inspection.
    Addresses environmental factory factors: uneven lighting, oil film glares, sensor thermal noise.
    """

    def __init__(self, target_size=(640, 640), clip_limit=2.5, tile_grid_size=(8, 8)):
        """
        Initialize preprocessor with standard parameters.
        :param target_size: Target (width, height) for YOLO model input.
        :param clip_limit: Threshold for contrast limiting in CLAHE.
        :param tile_grid_size: Grid size for local histogram equalization.
        """
        self.target_size = target_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def apply_denoising(self, image: np.ndarray, method="bilateral") -> np.ndarray:
        """
        Applies edge-preserving filtering to eliminate camera sensor grain and high-frequency noise.
        Bilateral filter is preferred because it smooths flat regions while preserving sharp defect edges.
        """
        if method == "bilateral":
            # d: diameter of pixel neighborhood, sigmaColor: filter sigma in color space, sigmaSpace: filter sigma in coordinate space
            if len(image.shape) == 3:
                return cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75)
            else:
                return cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), sigmaX=1.2)
        return image

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Compensates for non-uniform factory illumination, shadows, and reflection highlights.
        """
        if len(image.shape) == 3:
            # Convert to LAB color space to equalize only Lightness (L channel)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_enhanced = self.clahe.apply(l_channel)
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        else:
            return self.clahe.apply(image)

    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Corrects low-frequency illumination gradients using background division.
        Useful when strong light sources cause bright spots on metallic surfaces.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Large Gaussian kernel estimates the low-frequency illumination background
        background = cv2.GaussianBlur(gray, (51, 51), sigmaX=0)
        # Avoid division by zero
        background = np.maximum(background, 1)
        normalized = np.clip((gray.astype(np.float32) / background.astype(np.float32)) * 128.0, 0, 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        return normalized

    def letterbox_resize(self, image: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)) -> np.ndarray:
        """
        Resizes image while preserving aspect ratio using padding (letterboxing).
        Prevents geometrical distortion of defect features.
        """
        shape = image.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return image

    def preprocess_pipeline(self, image: np.ndarray, enhance_contrast: bool = True, denoise: bool = True) -> np.ndarray:
        """
        Executes the complete industrial preprocessing sequence.
        """
        processed = image.copy()
        if denoise:
            processed = self.apply_denoising(processed, method="bilateral")
        if enhance_contrast:
            processed = self.apply_clahe(processed)
        return processed


def generate_preprocessing_visual_report(image_path: str, save_path: str):
    """
    Generates a multi-stage visual comparison grid demonstrating preprocessing stages.
    Crucial deliverable for the assignment's 'Solution / Methodology' and 'Results' sections.
    """
    preprocessor = IndustrialImagePreprocessor()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Cannot read image at {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Denoised
    denoised = preprocessor.apply_denoising(img, method="bilateral")
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    
    # 2. CLAHE enhanced
    clahe_img = preprocessor.apply_clahe(denoised)
    clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)
    
    # 3. Illumination Normalized
    norm_img = preprocessor.normalize_illumination(clahe_img)
    norm_rgb = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
    
    # 4. Edge Highlight (Sobel Magnitude)
    gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_mag = np.clip((edge_mag / edge_mag.max()) * 255.0, 0, 255).astype(np.uint8)
    
    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    
    stages = [
        ("1. Raw Industrial Image", img_rgb, None),
        ("2. Bilateral Denoised", denoised_rgb, None),
        ("3. CLAHE Contrast Enhanced", clahe_rgb, None),
        ("4. Illumination Normalized", norm_rgb, None),
        ("5. Defect Edge Representation", edge_mag, 'gray')
    ]
    
    for ax, (title, img_data, cmap) in zip(axes, stages):
        if cmap:
            ax.imshow(img_data, cmap=cmap)
        else:
            ax.imshow(img_data)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.axis('off')
        
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Preprocessing visual report saved to: {save_path}")


if __name__ == "__main__":
    # Test script on first available image
    project_root = Path(__file__).resolve().parent.parent
    sample_images = list((project_root / "dataset" / "images" / "train").glob("*.jpg"))
    
    if sample_images:
        test_img = str(sample_images[0])
        output_graph = str(project_root / "results" / "graphs" / "preprocessing_pipeline_stages.png")
        generate_preprocessing_visual_report(test_img, output_graph)
    else:
        print("[!] No images found in dataset/images/train. Please run prepare_dataset.py first.")
