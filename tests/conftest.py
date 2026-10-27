from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image


@pytest.fixture
def make_test_config():
    def _make_test_config(tmp_path: Path, patient_column: str | None = "patient_id") -> Path:
        images = tmp_path / "images"
        images.mkdir()
        rows = []
        # Four patients per class permit two sequential stratified group splits.
        for label in range(5):
            for patient_index in range(4):
                image_id = f"patient_{label}_{patient_index}"
                Image.new("RGB", (16, 16), color=(label * 30, 20, 80)).save(images / f"{image_id}.png")
                rows.append({"image_id": image_id, "diagnosis": label, "patient_id": image_id})

        pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
        config = {
        "project": {"name": "test_run"},
        "data": {
            "images_dir": "images",
            "labels_file": "labels.csv",
            "split_file": "splits.csv",
            "image_column": "image_id",
            "label_column": "diagnosis",
            "patient_column": patient_column,
            "image_extension": ".png",
            "train_fraction": 0.5,
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
        },
        "model": {
            "input_size": [299, 299],
            "class_names": ["No DR", "Mild", "Moderate", "Severe", "PDR"],
            "pretrained_weights": "none",
            "dense_units": 8,
            "dropout_rate": 0.2,
            "augmentation": {
                "horizontal_flip": True,
                "rotation_factor": 0.0,
                "zoom_height_factor": 0.0,
                "zoom_width_factor": 0.0,
                "contrast_factor": 0.0,
            },
        },
        "training": {
            "seed": 7,
            "batch_size": 2,
            "freeze_epochs": 1,
            "fine_tune_epochs": 0,
            "initial_learning_rate": 0.001,
            "fine_tune_learning_rate": 0.00001,
            "unfreeze_fraction": 0.2,
            "patience": 1,
            "reduce_lr_factor": 0.5,
            "min_learning_rate": 0.000001,
            "use_class_weights": True,
            "mixed_precision": False,
            "output_dir": "artifacts",
        },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config_path

    return _make_test_config
