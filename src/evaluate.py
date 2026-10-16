"""
Evaluation module for Diabetic Retinopathy classification.
Computes Accuracy, F1-scores, Cohen's Quadratic Weighted Kappa (QWK),
Multiclass AUC-ROC, and exports plots.
"""

import os
import json
import argparse
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

from src.config_loader import load_config
from src.data_loader import RetinalDataLoader
from src.utils import plot_confusion_matrix, plot_roc_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(config_path: str = "config/config.yaml", model_path: str = None) -> dict:
    config = load_config(config_path)
    paths = config.get("paths", {})
    class_names = config.get("data", {}).get("class_names", [
        "No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"
    ])

    eval_model_path = model_path or paths.get("best_model_path", "models/best_model.keras")
    if not os.path.exists(eval_model_path):
        eval_model_path = paths.get("final_model_path", "models/final_model.keras")

    if not os.path.exists(eval_model_path):
        raise FileNotFoundError(f"Model file not found: {eval_model_path}")

    logger.info(f"Loading trained model from: {eval_model_path}")
    model = tf.keras.models.load_model(eval_model_path)

    loader = RetinalDataLoader(config)
    test_gen = loader.get_test_generator(batch_size=32)

    logger.info("Evaluating on test set...")
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # Metric calculations
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # Quadratic Weighted Kappa (Clinical standard metric for diabetic retinopathy)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # Multi-class AUC-ROC
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    try:
        auc_roc = roc_auc_score(y_true_onehot, y_pred_probs, multi_class="ovr", average="macro")
    except Exception as e:
        logger.warning(f"AUC-ROC calculation error: {e}")
        auc_roc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        "accuracy": float(acc),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "quadratic_weighted_kappa": float(qwk),
        "auc_roc_macro": float(auc_roc),
        "total_test_samples": int(len(y_true)),
        "classification_report": clf_report,
    }

    # Save and display plots
    plot_confusion_matrix(cm, class_names, save_path=paths.get("confusion_matrix_path", "logs/confusion_matrix.png"))
    plot_roc_curves(y_true_onehot, y_pred_probs, class_names, save_path=paths.get("roc_curve_path", "logs/roc_curve.png"))

    # Save metrics JSON
    report_path = paths.get("evaluation_report_path", "logs/evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("DIABETIC RETINOPATHY EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Accuracy:                   {acc:.4f}")
    logger.info(f"Weighted F1-Score:          {f1_weighted:.4f}")
    logger.info(f"Quadratic Weighted Kappa:   {qwk:.4f}")
    logger.info(f"AUC-ROC (macro):            {auc_roc:.4f}")
    logger.info(f"Detailed report saved to:   {report_path}")
    logger.info("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diabetic Retinopathy Model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    evaluate_model(args.config, args.model_path)


if __name__ == "__main__":
    main()
