"""Training orchestration for the two-phase transfer-learning workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ProjectConfig, config_as_dict
from .data import class_weights, load_split, make_dataset
from .model import build_model, compile_model, configure_tensorflow, fine_tune_top_fraction
from .utils import git_revision, utc_run_id, write_json, write_yaml


def _callbacks(run_dir: Path, patience: int, factor: float, min_lr: float):
    import tensorflow as tf

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=factor,
            patience=max(1, patience // 2),
            min_lr=min_lr,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), update_freq="epoch"),
        tf.keras.callbacks.CSVLogger(str(log_dir / "training.csv"), append=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def _history_payload(*histories: Any) -> dict[str, list[float]]:
    combined: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            combined.setdefault(key, []).extend(float(value) for value in values)
    return combined


def train(config: ProjectConfig) -> Path:
    """Train the classifier and return the created run directory."""
    import tensorflow as tf

    configure_tensorflow(config)
    train_frame = load_split(config, "train")
    validation_frame = load_split(config, "validation")
    train_ds = make_dataset(train_frame, config, training=True)
    validation_ds = make_dataset(validation_frame, config, training=False)

    run_dir = config.training.output_dir / utc_run_id(config.name)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(run_dir / "run_config.yaml", config_as_dict(config))
    write_json(run_dir / "class_names.json", {"class_names": list(config.model.class_names)})

    weights = (
        class_weights(train_frame, config.model.num_classes)
        if config.training.use_class_weights
        else None
    )
    model = build_model(config)
    frozen_history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=config.training.freeze_epochs,
        class_weight=weights,
        callbacks=_callbacks(
            run_dir,
            config.training.patience,
            config.training.reduce_lr_factor,
            config.training.min_learning_rate,
        ),
        verbose=2,
    )

    histories = [frozen_history]
    if config.training.fine_tune_epochs:
        fine_tune_top_fraction(model, config.training.unfreeze_fraction)
        compile_model(model, config.training.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            train_ds,
            validation_data=validation_ds,
            initial_epoch=config.training.freeze_epochs,
            epochs=config.training.freeze_epochs + config.training.fine_tune_epochs,
            class_weight=weights,
            callbacks=_callbacks(
                run_dir,
                config.training.patience,
                config.training.reduce_lr_factor,
                config.training.min_learning_rate,
            ),
            verbose=2,
        )
        histories.append(fine_tune_history)

    model = tf.keras.models.load_model(run_dir / "best_model.keras")
    model.save(run_dir / "final_model.keras")
    write_json(run_dir / "training_history.json", _history_payload(*histories))
    write_json(
        run_dir / "training_summary.json",
        {
            "run_directory": str(run_dir),
            "git_revision": git_revision(),
            "train_examples": len(train_frame),
            "validation_examples": len(validation_frame),
            "class_weights": weights,
            "model_name": model.name,
        },
    )
    return run_dir
