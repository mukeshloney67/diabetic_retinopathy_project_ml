"""Manifest validation, split generation, and tf.data input pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ConfigError, ProjectConfig

if TYPE_CHECKING:
    import tensorflow as tf


def _image_path(image_id: object, images_dir: Path, extension: str) -> Path:
    value = Path(str(image_id))
    if value.is_absolute():
        return value
    relative = value if value.suffix else value.with_suffix(extension)
    return images_dir / relative


def load_labels(config: ProjectConfig, require_images: bool = True) -> pd.DataFrame:
    """Read and validate the source manifest without silently discarding rows."""
    if not config.data.labels_file.is_file():
        raise FileNotFoundError(f"Labels file not found: {config.data.labels_file}")
    if require_images and not config.data.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {config.data.images_dir}")

    frame = pd.read_csv(config.data.labels_file)
    required = {config.data.image_column, config.data.label_column}
    if config.data.patient_column:
        required.add(config.data.patient_column)
    missing = required.difference(frame.columns)
    if missing:
        raise ConfigError(f"Labels file is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ConfigError("Labels file contains no rows.")
    if frame[config.data.image_column].isna().any():
        raise ConfigError("The image identifier column contains empty values.")

    labels = pd.to_numeric(frame[config.data.label_column], errors="coerce")
    if labels.isna().any() or not np.allclose(labels, np.round(labels)):
        raise ConfigError("Labels must be integer class IDs.")
    labels = labels.astype(int)
    valid = set(range(config.model.num_classes))
    invalid = sorted(set(labels).difference(valid))
    if invalid:
        raise ConfigError(
            f"Labels outside valid range 0..{config.model.num_classes - 1}: {invalid}"
        )

    normalized = frame.copy()
    normalized["_label"] = labels
    normalized["_image_path"] = [
        str(_image_path(value, config.data.images_dir, config.data.image_extension))
        for value in normalized[config.data.image_column]
    ]
    if normalized["_image_path"].duplicated().any():
        duplicates = normalized.loc[normalized["_image_path"].duplicated(), config.data.image_column]
        raise ConfigError(f"Duplicate image identifiers found, for example: {duplicates.iloc[0]!r}")
    if config.data.patient_column and normalized[config.data.patient_column].isna().any():
        raise ConfigError("The configured patient column contains empty values.")

    if require_images:
        missing_paths = [path for path in normalized["_image_path"] if not Path(path).is_file()]
        if missing_paths:
            preview = ", ".join(missing_paths[:3])
            raise FileNotFoundError(
                f"{len(missing_paths)} image files listed in labels.csv are missing. Examples: {preview}"
            )
    return normalized


def _split_groups(
    frame: pd.DataFrame, config: ProjectConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split whole patients, stratifying on their maximum observed DR grade."""
    group_column = config.data.patient_column
    if not group_column:
        raise ValueError("A group column is required for group splitting.")

    group_frame = (
        frame.groupby(group_column, as_index=False)["_label"]
        .max()
        .rename(columns={"_label": "_group_label"})
    )
    try:
        train_val_groups, test_groups = train_test_split(
            group_frame,
            test_size=config.data.test_fraction,
            random_state=config.training.seed,
            stratify=group_frame["_group_label"],
        )
        validation_relative = config.data.validation_fraction / (
            config.data.train_fraction + config.data.validation_fraction
        )
        train_groups, validation_groups = train_test_split(
            train_val_groups,
            test_size=validation_relative,
            random_state=config.training.seed,
            stratify=train_val_groups["_group_label"],
        )
    except ValueError as exc:
        raise ConfigError(
            "Patient-level stratification needs enough patients in every severity "
            "grade for all splits. Add data, merge rare classes for a research "
            "experiment, or adjust split fractions."
        ) from exc

    def select(groups: pd.DataFrame) -> pd.DataFrame:
        return frame[frame[group_column].isin(groups[group_column])].copy()

    return select(train_groups), select(validation_groups), select(test_groups)


def create_splits(config: ProjectConfig) -> pd.DataFrame:
    """Create deterministic train/validation/test splits and persist them."""
    frame = load_labels(config)
    if config.data.patient_column:
        train_frame, validation_frame, test_frame = _split_groups(frame, config)
    else:
        try:
            train_val, test_frame = train_test_split(
                frame,
                test_size=config.data.test_fraction,
                random_state=config.training.seed,
                stratify=frame["_label"],
            )
            validation_relative = config.data.validation_fraction / (
                config.data.train_fraction + config.data.validation_fraction
            )
            train_frame, validation_frame = train_test_split(
                train_val,
                test_size=validation_relative,
                random_state=config.training.seed,
                stratify=train_val["_label"],
            )
        except ValueError as exc:
            raise ConfigError(
                "Stratified splitting needs at least two examples of each class in "
                "the relevant split. Add data or adjust split fractions."
            ) from exc

    split_frames = []
    for name, split_frame in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        if split_frame.empty:
            raise ConfigError(f"The {name} split is empty.")
        item = split_frame.copy()
        item["split"] = name
        split_frames.append(item)
    combined = pd.concat(split_frames, ignore_index=True)

    if config.data.patient_column:
        group_counts = combined.groupby(config.data.patient_column)["split"].nunique()
        if (group_counts > 1).any():
            raise RuntimeError("Patient leakage detected while creating splits.")

    columns = [
        column
        for column in combined.columns
        if column not in {"_label", "_image_path"}
    ]
    config.data.split_file.parent.mkdir(parents=True, exist_ok=True)
    combined.loc[:, columns].sort_values("split").to_csv(config.data.split_file, index=False)
    return combined


def load_split(config: ProjectConfig, split_name: str) -> pd.DataFrame:
    """Load an existing generated split and reapply manifest validation."""
    if not config.data.split_file.is_file():
        raise FileNotFoundError(
            f"Split file not found: {config.data.split_file}. Run dr-prepare first."
        )
    split_frame = pd.read_csv(config.data.split_file)
    if "split" not in split_frame.columns:
        raise ConfigError("Split file must contain a 'split' column.")
    selected = split_frame[split_frame["split"] == split_name].copy()
    if selected.empty:
        raise ConfigError(f"No rows found for split {split_name!r}.")
    required = {config.data.image_column, config.data.label_column}
    missing = required.difference(selected.columns)
    if missing:
        raise ConfigError(f"Split file is missing columns: {sorted(missing)}")

    labels = pd.to_numeric(selected[config.data.label_column], errors="coerce")
    if labels.isna().any():
        raise ConfigError(f"Split {split_name!r} contains invalid labels.")
    selected["_label"] = labels.astype(int)
    selected["_image_path"] = [
        str(_image_path(value, config.data.images_dir, config.data.image_extension))
        for value in selected[config.data.image_column]
    ]
    missing_paths = [path for path in selected["_image_path"] if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"Split {split_name!r} references missing image files, for example: {missing_paths[0]}"
        )
    return selected


def make_dataset(
    frame: pd.DataFrame,
    config: ProjectConfig,
    training: bool = False,
) -> "tf.data.Dataset":
    """Make an efficient, deterministic image dataset with raw 0-255 float pixels."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required. Install dependencies with pip install -e .") from exc

    paths = frame["_image_path"].astype(str).to_numpy()
    labels = frame["_label"].astype("int32").to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode_and_resize(path: "tf.Tensor", label: "tf.Tensor"):
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, config.model.input_size, antialias=True)
        image = tf.cast(image, tf.float32)
        return image, tf.cast(label, tf.int32)

    options = tf.data.Options()
    options.experimental_deterministic = not training
    dataset = dataset.with_options(options)
    dataset = dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(frame), 4096),
            seed=config.training.seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(config.training.batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(frame: pd.DataFrame, num_classes: int) -> dict[int, float]:
    """Return balanced class weights from the training split only."""
    counts = frame["_label"].value_counts()
    total = len(frame)
    weights: dict[int, float] = {}
    for label in range(num_classes):
        count = int(counts.get(label, 0))
        if count == 0:
            raise ConfigError(f"Training split has no examples for class {label}.")
        weights[label] = total / (num_classes * count)
    return weights
