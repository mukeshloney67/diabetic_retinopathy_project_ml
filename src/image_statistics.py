"""
Image Statistics & Clinical Quality Assessment Module.
Calculates technical metrics (sharpness, illumination, contrast, RGB channels,
retinal coverage) and verifies prediction accuracy against ground-truth labels.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple


def compute_image_statistics(image_input: Any) -> Dict[str, Any]:
    """
    Calculate comprehensive photometric and structural statistics for a retinal fundus image.

    Args:
        image_input: PIL.Image or NumPy array (RGB).

    Returns:
        Dictionary containing technical and photometric statistics.
    """
    if isinstance(image_input, Image.Image):
        arr = np.array(image_input.convert("RGB"))
    else:
        arr = np.array(image_input)

    h, w, c = arr.shape
    aspect_ratio = round(w / float(h), 3)
    total_pixels = h * w

    # Convert to grayscale for structural & luminance analysis
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # 1. Retinal Field of View (FOV) Mask - exclude black background border
    retina_mask = gray > 15
    retina_pixel_count = int(np.sum(retina_mask))
    retinal_coverage_pct = round((retina_pixel_count / float(total_pixels)) * 100.0, 2)

    # Extract pixels belonging specifically to the retina
    if retina_pixel_count > 0:
        retinal_pixels = gray[retina_mask]
        mean_brightness = round(float(np.mean(retinal_pixels)), 2)
        std_brightness = round(float(np.std(retinal_pixels)), 2)
        median_brightness = round(float(np.median(retinal_pixels)), 2)
        rms_contrast = round(float(np.sqrt(np.mean((retinal_pixels - mean_brightness) ** 2))), 2)

        # Overexposed & Underexposed pixel percentages within retinal FOV
        overexposed_pct = round(float(np.sum(retinal_pixels > 240) / retina_pixel_count) * 100.0, 2)
        underexposed_pct = round(float(np.sum(retinal_pixels < 25) / retina_pixel_count) * 100.0, 2)

        # Illumination uniformity (ratio between 10th and 90th percentiles)
        p10 = float(np.percentile(retinal_pixels, 10))
        p90 = float(np.percentile(retinal_pixels, 90))
        uniformity_ratio = round((p10 / (p90 + 1e-5)), 3)
    else:
        mean_brightness = round(float(np.mean(gray)), 2)
        std_brightness = round(float(np.std(gray)), 2)
        median_brightness = round(float(np.median(gray)), 2)
        rms_contrast = round(float(np.std(gray)), 2)
        overexposed_pct = 0.0
        underexposed_pct = 0.0
        uniformity_ratio = 1.0

    # 2. Blur / Sharpness Index via Laplacian Variance
    laplacian_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    if laplacian_var > 150:
        sharpness_quality = "Optimal / Sharp"
        sharpness_flag = "🟢"
    elif laplacian_var >= 60:
        sharpness_quality = "Acceptable"
        sharpness_flag = "🟡"
    else:
        sharpness_quality = "Degraded / Blurry"
        sharpness_flag = "🔴"

    # 3. Individual RGB Channel Distributions
    r_channel = arr[:, :, 0]
    g_channel = arr[:, :, 1]
    b_channel = arr[:, :, 2]

    # Red is predominant in fundus; Green highlights microaneurysms; Blue is usually low
    rgb_stats = {
        "red": {
            "mean": round(float(np.mean(r_channel[retina_mask] if retina_pixel_count > 0 else r_channel)), 2),
            "std": round(float(np.std(r_channel[retina_mask] if retina_pixel_count > 0 else r_channel)), 2),
        },
        "green": {
            "mean": round(float(np.mean(g_channel[retina_mask] if retina_pixel_count > 0 else g_channel)), 2),
            "std": round(float(np.std(g_channel[retina_mask] if retina_pixel_count > 0 else g_channel)), 2),
        },
        "blue": {
            "mean": round(float(np.mean(b_channel[retina_mask] if retina_pixel_count > 0 else b_channel)), 2),
            "std": round(float(np.std(b_channel[retina_mask] if retina_pixel_count > 0 else b_channel)), 2),
        },
    }

    # Clinical Gradability Assessment
    gradability_reasons = []
    if laplacian_var < 50:
        gradability_reasons.append("Image is excessively blurred (low Laplacian variance).")
    if underexposed_pct > 40:
        gradability_reasons.append("Severe underexposure across retinal field.")
    if overexposed_pct > 25:
        gradability_reasons.append("Significant flash glare / overexposure.")
    if retinal_coverage_pct < 30:
        gradability_reasons.append("Retinal disc occupies insufficient portion of image.")

    is_gradable = len(gradability_reasons) == 0

    return {
        "dimensions": {
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "total_pixels": total_pixels,
        },
        "sharpness": {
            "score": laplacian_var,
            "quality": sharpness_quality,
            "flag": sharpness_flag,
        },
        "photometry": {
            "mean_brightness": mean_brightness,
            "median_brightness": median_brightness,
            "std_brightness": std_brightness,
            "rms_contrast": rms_contrast,
            "uniformity_ratio": uniformity_ratio,
            "overexposed_pct": overexposed_pct,
            "underexposed_pct": underexposed_pct,
            "retinal_coverage_pct": retinal_coverage_pct,
        },
        "channels": rgb_stats,
        "clinical_gradability": {
            "is_gradable": is_gradable,
            "status": "Gradable for Screening" if is_gradable else "Questionable / Poor Quality",
            "warnings": gradability_reasons,
        },
    }


def compute_sample_accuracy(
    predicted_class: int,
    ground_truth_class: int,
    predicted_probabilities: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate diagnostic accuracy, severity distance, and loss metrics
    when comparing a model prediction against a verified ground-truth clinical stage.

    Args:
        predicted_class: Predicted stage (0 to 4).
        ground_truth_class: Verified clinical stage (0 to 4).
        predicted_probabilities: Array of softmax probabilities for all 5 classes.

    Returns:
        Accuracy comparison metrics.
    """
    is_exact_match = int(predicted_class) == int(ground_truth_class)
    stage_difference = abs(int(predicted_class) - int(ground_truth_class))

    # Quadratic Penalty Accuracy Score (1.0 for exact, penalizes far off predictions):
    # Max possible difference is 4 (e.g. stage 0 vs stage 4). 1 - (diff^2 / 16)
    quadratic_accuracy = max(0.0, round(1.0 - (stage_difference ** 2) / 16.0, 4))

    # Cross Entropy Loss on this sample if probabilities are provided
    sample_loss = None
    if predicted_probabilities is not None and len(predicted_probabilities) > ground_truth_class:
        prob_true = max(float(predicted_probabilities[ground_truth_class]), 1e-7)
        sample_loss = round(-np.log(prob_true), 4)

    return {
        "exact_match": is_exact_match,
        "status": "Exact Match ✅" if is_exact_match else ("Minor Stage Discrepancy ⚠️" if stage_difference == 1 else "Misclassified ❌"),
        "predicted_class": int(predicted_class),
        "ground_truth_class": int(ground_truth_class),
        "stage_difference": stage_difference,
        "quadratic_score": quadratic_accuracy,
        "quadratic_score_pct": f"{quadratic_accuracy * 100:.1f}%",
        "sample_cross_entropy": sample_loss,
    }
