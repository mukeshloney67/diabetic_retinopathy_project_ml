"""
CLI Image Statistics & Diagnostic Accuracy Checker.
Runs technical analysis (sharpness, illumination, contrast, RGB) and computes
accuracy against an optional verified ground-truth label.
"""

import os
import argparse
import numpy as np
from PIL import Image

from src.image_statistics import compute_image_statistics, compute_sample_accuracy
from src.predict import RetinopathyPredictor, SEVERITY_STAGES


def analyze_image_cli(image_path: str, model_path: str = "models/best_model.keras", ground_truth: int = None):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    raw_arr = np.array(pil_img)

    # 1. Compute Image Technical Statistics
    stats = compute_image_statistics(raw_arr)

    print("\n" + "=" * 60)
    print("📊 RETINAL IMAGE PHOTOMETRIC & STRUCTURAL STATISTICS")
    print("=" * 60)
    print(f"File:                  {os.path.basename(image_path)}")
    print(f"Dimensions:            {stats['dimensions']['width']} × {stats['dimensions']['height']} px")
    print(f"Aspect Ratio:          {stats['dimensions']['aspect_ratio']}:1")
    print(f"Image Sharpness:       {stats['sharpness']['score']} ({stats['sharpness']['quality']}) {stats['sharpness']['flag']}")
    print(f"Retinal Coverage:      {stats['photometry']['retinal_coverage_pct']}% of field of view")
    print(f"Mean Retinal Luminance:{stats['photometry']['mean_brightness']} / 255 (std: {stats['photometry']['std_brightness']})")
    print(f"RMS Contrast:          {stats['photometry']['rms_contrast']}")
    print(f"Illumination Uniformity:{stats['photometry']['uniformity_ratio']} (10th/90th percentile)")
    print(f"Overexposed Glare:     {stats['photometry']['overexposed_pct']}%")
    print(f"Underexposed Shadow:   {stats['photometry']['underexposed_pct']}%")
    print(f"Clinical Gradability:  {stats['clinical_gradability']['status']}")
    if stats['clinical_gradability']['warnings']:
        for w in stats['clinical_gradability']['warnings']:
            print(f"  ⚠️  Notice: {w}")

    print("\nRGB Channel Distributions (Vascular Analysis):")
    for ch, vals in stats['channels'].items():
        print(f"  • {ch.capitalize():6s} Channel Mean: {vals['mean']:6.2f} (std: {vals['std']:5.2f})")

    # 2. Model Prediction
    print("\n" + "=" * 60)
    print("👁️  MODEL INFERENCE & ACCURACY CHECK")
    print("=" * 60)
    if os.path.exists(model_path):
        predictor = RetinopathyPredictor(model_path=model_path)
        pred_res = predictor.predict(image_path)
        pred_class = pred_res["predicted_class"]
        confidence = pred_res["confidence"]
        stage_name = pred_res["stage_name"]
        print(f"Predicted Diagnosis:   {stage_name} (Stage {pred_class})")
        print(f"Model Confidence:      {confidence:.2%}")
        print(f"Clinical Severity:     {pred_res['severity_level']}")
        print(f"Action Recommendation: {pred_res['clinical_action']}")
        print(f"Key Findings:          {pred_res['findings']}")

        # 3. Ground Truth Accuracy Verification
        if ground_truth is not None:
            acc_data = compute_sample_accuracy(pred_class, ground_truth)
            print("\nAccuracy Verification:")
            print(f"  Ground Truth:        Stage {ground_truth} ({SEVERITY_STAGES.get(ground_truth, {}).get('name', 'Unknown')})")
            print(f"  Exact Match:         {acc_data['status']}")
            print(f"  Stage Discrepancy:   {acc_data['stage_difference']} level(s)")
            print(f"  Quadratic Score:     {acc_data['quadratic_score_pct']}")
    else:
        print(f"Model checkpoint not found at '{model_path}'.")
        print("Train with 'python -m src.train' to generate trained weights.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze retinal image statistics & accuracy")
    parser.add_argument("--image", type=str, required=True, help="Path to retinal image")
    parser.add_argument("--model", type=str, default="models/best_model.keras", help="Model checkpoint path")
    parser.add_argument("--ground_truth", type=int, default=None, choices=[0, 1, 2, 3, 4], help="Verified true stage (0 to 4)")
    args = parser.parse_args()
    analyze_image_cli(args.image, args.model, args.ground_truth)
