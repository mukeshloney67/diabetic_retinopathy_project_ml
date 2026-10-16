"""
Inference module for Diabetic Retinopathy prediction.
Supports single-image and batch image predictions with clinical severity interpretations.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from src.preprocessing import ben_graham_enhancement, inception_preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_STAGES = {
    0: {
        "name": "No DR",
        "severity": "Normal",
        "clinical_action": "Routine annual screening recommended.",
        "findings": "No microaneurysms or retinal abnormalities detected.",
    },
    1: {
        "name": "Mild NPDR",
        "severity": "Mild Non-Proliferative",
        "clinical_action": "Follow up examination in 6-12 months.",
        "findings": "Presence of microaneurysms only.",
    },
    2: {
        "name": "Moderate NPDR",
        "severity": "Moderate Non-Proliferative",
        "clinical_action": "Follow up examination in 3-6 months; optimize glycemic control.",
        "findings": "Microaneurysms, hemorrhages, hard exudates, or cotton wool spots.",
    },
    3: {
        "name": "Severe NPDR",
        "severity": "Severe Non-Proliferative",
        "clinical_action": "Prompt ophthalmologist referral (within 2-4 weeks). High risk of progression.",
        "findings": "Extensive intraretinal hemorrhages (4 quadrants), venous beading (2+ quadrants), or IRMA.",
    },
    4: {
        "name": "Proliferative DR",
        "severity": "Proliferative Diabetic Retinopathy",
        "clinical_action": "Urgent specialist intervention (panretinal photocoagulation or anti-VEGF therapy).",
        "findings": "Neovascularization of retina/disc, preretinal or vitreous hemorrhage.",
    },
}


class RetinopathyPredictor:
    """Predictor engine for diabetic retinopathy using trained InceptionV3 model."""

    def __init__(self, model_path: str = "models/best_model.keras", image_size: int = 299):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        logger.info(f"Loading InceptionV3 model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        self.image_size = (image_size, image_size)

    def preprocess_image(self, image_path: str, apply_enhancement: bool = True) -> np.ndarray:
        """
        Load image from disk and apply Ben Graham enhancement followed by InceptionV3 normalization.
        """
        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img)

        if apply_enhancement:
            img_arr = ben_graham_enhancement(img_arr, target_size=self.image_size)
        else:
            img_arr = cv2.resize(img_arr, self.image_size)

        norm_arr = inception_preprocess(img_arr)
        return np.expand_dims(norm_arr, axis=0)

    def predict(self, image_path: str) -> Dict:
        """
        Run inference on a single retinal fundus image.
        """
        processed_input = self.preprocess_image(image_path)
        probs = self.model.predict(processed_input, verbose=0)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        info = SEVERITY_STAGES.get(class_idx, {})

        return {
            "image_path": image_path,
            "predicted_class": class_idx,
            "stage_name": info.get("name", f"Stage {class_idx}"),
            "severity_level": info.get("severity", "Unknown"),
            "confidence": confidence,
            "clinical_action": info.get("clinical_action", ""),
            "findings": info.get("findings", ""),
            "probabilities": {
                SEVERITY_STAGES[i]["name"]: float(probs[i]) for i in range(len(probs))
            },
        }

    def predict_directory(self, input_dir: str, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Run batch inference over a directory of retinal images.
        """
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(valid_exts)]

        results = []
        logger.info(f"Processing {len(files)} retinal images in {input_dir}...")

        for fname in files:
            fpath = os.path.join(input_dir, fname)
            try:
                res = self.predict(fpath)
                results.append({
                    "filename": fname,
                    "predicted_class": res["predicted_class"],
                    "stage_name": res["stage_name"],
                    "severity_level": res["severity_level"],
                    "confidence": res["confidence"],
                    "clinical_action": res["clinical_action"],
                })
            except Exception as e:
                logger.error(f"Failed to process {fname}: {e}")

        df = pd.DataFrame(results)
        if output_csv:
            df.to_csv(output_csv, index=False)
            logger.info(f"Saved batch prediction results to: {output_csv}")

        return df


def main():
    parser = argparse.ArgumentParser(description="Inference for Diabetic Retinopathy Detection")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--dir", type=str, default=None, help="Directory for batch prediction")
    parser.add_argument("--model", type=str, default="models/best_model.keras", help="Model path")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        sys.exit(1)

    predictor = RetinopathyPredictor(args.model)

    if args.image:
        res = predictor.predict(args.image)
        print("\n" + "=" * 55)
        print("👁️  DIABETIC RETINOPATHY PREDICTION RESULT")
        print("=" * 55)
        print(f"File:            {os.path.basename(res['image_path'])}")
        print(f"Predicted Stage: {res['stage_name']} (Class {res['predicted_class']})")
        print(f"Severity:        {res['severity_level']}")
        print(f"Confidence:      {res['confidence']:.2%}")
        print(f"Clinical Action: {res['clinical_action']}")
        print(f"Key Findings:    {res['findings']}")
        print("\nProbability Distribution:")
        for stage, p in res["probabilities"].items():
            bar = "█" * int(p * 25)
            print(f"  {stage:20s}: {p:6.2%} {bar}")
        print("=" * 55 + "\n")

    elif args.dir:
        predictor.predict_directory(args.dir, args.output_csv)


if __name__ == "__main__":
    main()
