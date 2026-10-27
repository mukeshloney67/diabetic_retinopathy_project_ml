from __future__ import annotations

from dr_classifier.config import load_config
from dr_classifier.data import class_weights, create_splits, load_split


def test_group_splits_do_not_leak_patients(tmp_path, make_test_config):
    config = load_config(make_test_config(tmp_path))
    splits = create_splits(config)

    assert config.data.split_file.is_file()
    patient_split_counts = splits.groupby("patient_id")["split"].nunique()
    assert patient_split_counts.max() == 1
    assert set(splits["split"]) == {"train", "validation", "test"}
    assert len(load_split(config, "test")) > 0


def test_balanced_weights_are_inverse_to_training_frequency(tmp_path, make_test_config):
    config = load_config(make_test_config(tmp_path))
    splits = create_splits(config)
    train = splits[splits["split"] == "train"]

    weights = class_weights(train, config.model.num_classes)

    assert set(weights) == set(range(config.model.num_classes))
    assert all(value > 0 for value in weights.values())
