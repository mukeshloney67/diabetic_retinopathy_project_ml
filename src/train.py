"""
Two-phase training script for Diabetic Retinopathy classification with InceptionV3.
Phase 1: Feature extraction (frozen base)
Phase 2: Fine-tuning (unfrozen top Inception blocks)
"""

import os
import json
import argparse
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard,
)

from src.config_loader import load_config
from src.data_loader import RetinalDataLoader
from src.model import InceptionV3RetinopathyClassifier
from src.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def train_pipeline(config_path: str = "config/config.yaml", phase: str = "both") -> None:
    config = load_config(config_path)
    set_seed(config.get("project", {}).get("seed", 42))

    paths = config.get("paths", {})
    os.makedirs(paths.get("model_dir", "models"), exist_ok=True)
    os.makedirs(paths.get("log_dir", "logs"), exist_ok=True)

    # 1. Initialize data generators
    logger.info("Initializing dataset generators...")
    loader = RetinalDataLoader(config)
    batch_size_p1 = config.get("training", {}).get("phase1", {}).get("batch_size", 32)
    train_gen = loader.get_train_generator(batch_size=batch_size_p1)
    val_gen = loader.get_val_generator(batch_size=batch_size_p1)

    # Compute balanced class weights
    class_weights = None
    if config.get("training", {}).get("use_class_weights", True):
        class_weights = loader.calculate_class_weights(train_gen)
        logger.info(f"Calculated class weights: {class_weights}")

    # 2. Build model
    classifier = InceptionV3RetinopathyClassifier(config)
    model = classifier.build()

    # Callbacks
    best_model_path = paths.get("best_model_path", "models/best_model.keras")
    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor=config.get("training", {}).get("early_stopping", {}).get("monitor", "val_accuracy"),
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor=config.get("training", {}).get("early_stopping", {}).get("monitor", "val_accuracy"),
            patience=config.get("training", {}).get("early_stopping", {}).get("patience", 8),
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=config.get("training", {}).get("reduce_lr", {}).get("monitor", "val_loss"),
            factor=config.get("training", {}).get("reduce_lr", {}).get("factor", 0.5),
            patience=config.get("training", {}).get("reduce_lr", {}).get("patience", 3),
            min_lr=config.get("training", {}).get("reduce_lr", {}).get("min_lr", 1e-7),
            verbose=1,
        ),
        CSVLogger(os.path.join(paths.get("log_dir", "logs"), "training_log.csv")),
    ]

    history_records = {}

    # --- PHASE 1: Transfer Learning ---
    if phase in ["both", "phase1"]:
        logger.info("==================================================")
        logger.info("Starting Phase 1: Transfer Learning (Frozen Base)")
        logger.info("==================================================")
        classifier.compile_for_phase1()
        epochs_p1 = config.get("training", {}).get("phase1", {}).get("epochs", 10)

        hist1 = model.fit(
            train_gen,
            epochs=epochs_p1,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        history_records["phase1"] = {k: [float(x) for x in v] for k, v in hist1.history.items()}

    # --- PHASE 2: Fine-Tuning ---
    if phase in ["both", "phase2"]:
        logger.info("==================================================")
        logger.info("Starting Phase 2: Fine-Tuning InceptionV3")
        logger.info("==================================================")
        classifier.compile_for_phase2()
        epochs_p2 = config.get("training", {}).get("phase2", {}).get("epochs", 25)

        hist2 = model.fit(
            train_gen,
            epochs=epochs_p2,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        history_records["phase2"] = {k: [float(x) for x in v] for k, v in hist2.history.items()}

    # Save final model & history
    final_model_path = paths.get("final_model_path", "models/final_model.keras")
    classifier.save(final_model_path)

    history_path = paths.get("history_path", "models/training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=2)

    logger.info("Training complete! Model and logs saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train Diabetic Retinopathy Detection Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--phase", type=str, default="both", choices=["both", "phase1", "phase2"])
    args = parser.parse_args()
    train_pipeline(args.config, args.phase)


if __name__ == "__main__":
    main()
