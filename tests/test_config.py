from __future__ import annotations

import pytest

from dr_classifier.config import ConfigError, load_config


def test_load_config_resolves_paths_from_config_location(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project: {name: test}
data:
  images_dir: images
  labels_file: labels.csv
  split_file: splits.csv
  image_column: image_id
  label_column: diagnosis
  patient_column: null
  image_extension: png
  train_fraction: 0.7
  validation_fraction: 0.15
  test_fraction: 0.15
model:
  input_size: [299, 299]
  class_names: [No DR, Mild]
  pretrained_weights: none
  dense_units: 4
  dropout_rate: 0.2
  augmentation:
    horizontal_flip: true
    rotation_factor: 0.0
    zoom_height_factor: 0.0
    zoom_width_factor: 0.0
    contrast_factor: 0.0
training:
  seed: 1
  batch_size: 2
  freeze_epochs: 1
  fine_tune_epochs: 0
  initial_learning_rate: 0.001
  fine_tune_learning_rate: 0.00001
  unfreeze_fraction: 0.2
  patience: 1
  reduce_lr_factor: 0.5
  min_learning_rate: 0.000001
  use_class_weights: true
  mixed_precision: false
  output_dir: artifacts
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.images_dir == tmp_path / "images"
    assert config.data.image_extension == ".png"
    assert config.model.num_classes == 2


def test_rejects_invalid_split_fractions(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)
