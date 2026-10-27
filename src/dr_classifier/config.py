"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class DataConfig:
    images_dir: Path
    labels_file: Path
    split_file: Path
    image_column: str
    label_column: str
    patient_column: str | None
    image_extension: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip: bool
    rotation_factor: float
    zoom_height_factor: float
    zoom_width_factor: float
    contrast_factor: float


@dataclass(frozen=True)
class ModelConfig:
    input_size: tuple[int, int]
    class_names: tuple[str, ...]
    pretrained_weights: str | None
    dense_units: int
    dropout_rate: float
    augmentation: AugmentationConfig

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: int
    freeze_epochs: int
    fine_tune_epochs: int
    initial_learning_rate: float
    fine_tune_learning_rate: float
    unfreeze_fraction: float
    patience: int
    reduce_lr_factor: float
    min_learning_rate: float
    use_class_weights: bool
    mixed_precision: bool
    output_dir: Path


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    source_path: Path


def _required(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required configuration key: {key}")
    return section[key]


def _path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def _as_fraction(value: Any, key: str) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be a number.") from exc
    if not 0 < fraction < 1:
        raise ConfigError(f"{key} must be between 0 and 1.")
    return fraction


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML experiment config and resolve its paths relative to the file."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {source_path}")

    with source_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    try:
        project_raw = _required(raw, "project")
        data_raw = _required(raw, "data")
        model_raw = _required(raw, "model")
        training_raw = _required(raw, "training")
    except (TypeError, ConfigError) as exc:
        raise ConfigError("Configuration must contain project, data, model, and training sections.") from exc

    if not all(isinstance(section, dict) for section in (project_raw, data_raw, model_raw, training_raw)):
        raise ConfigError("Each top-level configuration section must be a mapping.")

    config_dir = source_path.parent
    train_fraction = _as_fraction(_required(data_raw, "train_fraction"), "data.train_fraction")
    validation_fraction = _as_fraction(_required(data_raw, "validation_fraction"), "data.validation_fraction")
    test_fraction = _as_fraction(_required(data_raw, "test_fraction"), "data.test_fraction")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-6:
        raise ConfigError("data split fractions must add up to 1.0.")

    class_names = tuple(str(name) for name in _required(model_raw, "class_names"))
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ConfigError("model.class_names must contain at least two unique names.")

    input_size = tuple(int(value) for value in _required(model_raw, "input_size"))
    if len(input_size) != 2 or min(input_size) <= 0:
        raise ConfigError("model.input_size must contain two positive integers.")

    augmentation_raw = _required(model_raw, "augmentation")
    if not isinstance(augmentation_raw, dict):
        raise ConfigError("model.augmentation must be a mapping.")

    patient_column = data_raw.get("patient_column")
    if patient_column is not None:
        patient_column = str(patient_column)

    image_extension = str(_required(data_raw, "image_extension"))
    if image_extension and not image_extension.startswith("."):
        image_extension = f".{image_extension}"

    data = DataConfig(
        images_dir=_path(str(_required(data_raw, "images_dir")), config_dir),
        labels_file=_path(str(_required(data_raw, "labels_file")), config_dir),
        split_file=_path(str(_required(data_raw, "split_file")), config_dir),
        image_column=str(_required(data_raw, "image_column")),
        label_column=str(_required(data_raw, "label_column")),
        patient_column=patient_column,
        image_extension=image_extension,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    model = ModelConfig(
        input_size=input_size,
        class_names=class_names,
        pretrained_weights=model_raw.get("pretrained_weights"),
        dense_units=int(_required(model_raw, "dense_units")),
        dropout_rate=float(_required(model_raw, "dropout_rate")),
        augmentation=AugmentationConfig(
            horizontal_flip=bool(_required(augmentation_raw, "horizontal_flip")),
            rotation_factor=float(_required(augmentation_raw, "rotation_factor")),
            zoom_height_factor=float(_required(augmentation_raw, "zoom_height_factor")),
            zoom_width_factor=float(_required(augmentation_raw, "zoom_width_factor")),
            contrast_factor=float(_required(augmentation_raw, "contrast_factor")),
        ),
    )
    training = TrainingConfig(
        seed=int(_required(training_raw, "seed")),
        batch_size=int(_required(training_raw, "batch_size")),
        freeze_epochs=int(_required(training_raw, "freeze_epochs")),
        fine_tune_epochs=int(_required(training_raw, "fine_tune_epochs")),
        initial_learning_rate=float(_required(training_raw, "initial_learning_rate")),
        fine_tune_learning_rate=float(_required(training_raw, "fine_tune_learning_rate")),
        unfreeze_fraction=float(_required(training_raw, "unfreeze_fraction")),
        patience=int(_required(training_raw, "patience")),
        reduce_lr_factor=float(_required(training_raw, "reduce_lr_factor")),
        min_learning_rate=float(_required(training_raw, "min_learning_rate")),
        use_class_weights=bool(_required(training_raw, "use_class_weights")),
        mixed_precision=bool(_required(training_raw, "mixed_precision")),
        output_dir=_path(str(_required(training_raw, "output_dir")), config_dir),
    )

    if not 0 < model.dropout_rate < 1:
        raise ConfigError("model.dropout_rate must be between 0 and 1.")
    if model.dense_units <= 0 or training.batch_size <= 0:
        raise ConfigError("model.dense_units and training.batch_size must be positive.")
    if not 0 < training.unfreeze_fraction <= 1:
        raise ConfigError("training.unfreeze_fraction must be in (0, 1].")

    return ProjectConfig(
        name=str(_required(project_raw, "name")),
        data=data,
        model=model,
        training=training,
        source_path=source_path,
    )


def config_as_dict(config: ProjectConfig) -> dict[str, Any]:
    """Return a YAML/JSON-serializable representation of a loaded config."""
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(item) for item in value]
        return value

    payload = asdict(config)
    payload.pop("source_path", None)
    return convert(payload)
