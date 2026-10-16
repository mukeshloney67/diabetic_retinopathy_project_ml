"""
Utility functions for logging, reproducibility, and plotting.
"""

import os
import random
import logging
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for full reproducibility across NumPy, Python random, and TensorFlow.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    logger.info(f"Random seed set to: {seed}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = "logs/confusion_matrix.png",
    title: str = "Diabetic Retinopathy - Confusion Matrix"
) -> None:
    """
    Plot and save dual confusion matrix: raw counts and normalized percentages.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} (Counts)", fontsize=13)
    axes[0].set_xlabel("Predicted Stage", fontsize=11)
    axes[0].set_ylabel("True Stage", fontsize=11)

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{title} (Normalized)", fontsize=13)
    axes[1].set_xlabel("Predicted Stage", fontsize=11)
    axes[1].set_ylabel("True Stage", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curves(
    y_true_onehot: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str],
    save_path: str = "logs/roc_curve.png"
) -> None:
    """
    Plot and save multiclass One-vs-Rest (OvR) ROC curves.
    """
    from sklearn.metrics import roc_curve, auc

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) - InceptionV3 DR Model", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved to {save_path}")
