"""Evaluation metrics, reports, and plots for held-out test data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import ProjectConfig
from .data import load_split, make_dataset
from .utils import write_json


def evaluate(model_path: str | Path, config: ProjectConfig, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Evaluate a saved model only on the untouched test split."""
    import tensorflow as tf

    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    frame = load_split(config, "test")
    dataset = make_dataset(frame, config, training=False)
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(dataset, verbose=0)
    predicted = probabilities.argmax(axis=1)
    actual = frame["_label"].to_numpy()
    labels = list(range(config.model.num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    metrics: dict[str, Any] = {
        "model_path": str(model_path),
        "examples": int(len(actual)),
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(actual, predicted, weights="quadratic")),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(config.model.class_names)
        },
        "classification_report": classification_report(
            actual,
            predicted,
            labels=labels,
            target_names=list(config.model.class_names),
            output_dict=True,
            zero_division=0,
        ),
    }

    # Referable DR is conventionally grade >= 2; this binary result is supplemental.
    referable_actual = (actual >= 2).astype(int)
    referable_probability = probabilities[:, 2:].sum(axis=1)
    referable_predicted = (predicted >= 2).astype(int)
    metrics["referable_dr"] = {
        "threshold": 0.5,
        "sensitivity": float(
            np.sum((referable_predicted == 1) & (referable_actual == 1))
            / max(np.sum(referable_actual == 1), 1)
        ),
        "specificity": float(
            np.sum((referable_predicted == 0) & (referable_actual == 0))
            / max(np.sum(referable_actual == 0), 1)
        ),
    }
    if len(np.unique(referable_actual)) == 2:
        metrics["referable_dr"]["roc_auc"] = float(roc_auc_score(referable_actual, referable_probability))

    destination = Path(output_dir) if output_dir else model_path.parent / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "metrics.json", metrics)
    write_json(
        destination / "predictions.json",
        {
            "image_ids": frame[config.data.image_column].astype(str).tolist(),
            "actual_labels": actual.tolist(),
            "predicted_labels": predicted.tolist(),
            "probabilities": probabilities.tolist(),
        },
    )
    _save_confusion_matrix(
        actual,
        predicted,
        list(config.model.class_names),
        destination / "confusion_matrix.png",
    )
    return metrics


def _save_confusion_matrix(
    actual: np.ndarray,
    predicted: np.ndarray,
    class_names: list[str],
    destination: Path,
) -> None:
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(class_names))))
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual grade",
        xlabel="Predicted grade",
        title="Diabetic retinopathy severity confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
