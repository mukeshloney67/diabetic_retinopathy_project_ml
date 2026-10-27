"""
InceptionV3 Model Architecture for Diabetic Retinopathy Classification.
Implements a 2-phase transfer learning and fine-tuning engine.
"""

import logging
from typing import Dict, Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import InceptionV3

logger = logging.getLogger(__name__)


class InceptionV3RetinopathyClassifier:
    """
    Deep Learning model using InceptionV3 backbone with a custom multi-stage
    classification head designed for multi-class diabetic retinopathy severity grading.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model_cfg = config.get("model", {})
        self.training_cfg = config.get("training", {})
        self.num_classes = config.get("data", {}).get("num_classes", 5)
        self.input_shape = tuple(self.model_cfg.get("input_shape", [299, 299, 3]))

        self.base_model: Optional[tf.keras.Model] = None
        self.model: Optional[tf.keras.Model] = None

    def build(self) -> tf.keras.Model:
        """
        Construct the InceptionV3 feature extractor and attach the classification head.
        Initializes in Phase 1 mode (base model frozen).

        Returns:
            Compiled Keras model ready for Phase 1 transfer learning.
        """
        logger.info("Initializing InceptionV3 backbone with ImageNet weights...")

        # Pretrained InceptionV3 base without ImageNet 1000-class classifier
        self.base_model = InceptionV3(
            weights=self.model_cfg.get("pretrained_weights", "imagenet"),
            include_top=False,
            input_shape=self.input_shape,
        )

        # Freeze all layers of the base model for Phase 1
        self.base_model.trainable = False

        # Build custom classification head
        inputs = layers.Input(shape=self.input_shape, name="retinal_input")
        x = self.base_model(inputs, training=False)

        # Spatial feature pooling
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

        # Dense Block 1
        x = layers.Dense(self.model_cfg.get("dense_units_1", 512), activation="relu", name="dense_1")(x)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Dropout(self.model_cfg.get("dropout_1", 0.4), name="dropout_1")(x)

        # Dense Block 2
        x = layers.Dense(self.model_cfg.get("dense_units_2", 256), activation="relu", name="dense_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Dropout(self.model_cfg.get("dropout_2", 0.3), name="dropout_2")(x)

        # Final 5-class severity classification output
        outputs = layers.Dense(self.num_classes, activation="softmax", name="dr_predictions")(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name="InceptionV3_DR_Classifier")

        logger.info(f"Model successfully constructed. Total parameters: {self.model.count_params():,}")
        return self.model

    def compile_for_phase1(self) -> None:
        """
        Compile for Phase 1: Transfer learning with frozen InceptionV3 feature extractor.
        Uses higher learning rate (default 1e-3).
        """
        if self.model is None:
            self.build()

        lr = self.training_cfg.get("phase1", {}).get("learning_rate", 1e-3)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc", multi_label=True),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        logger.info(f"Model compiled for Phase 1 (Transfer Learning, lr={lr})")

    def compile_for_phase2(self) -> None:
        """
        Compile for Phase 2: Fine-tuning.
        Unfreezes top Inception blocks while keeping early general low-level edge
        detectors frozen. Uses a very small learning rate (default 1e-5).
        """
        if self.base_model is None or self.model is None:
            raise RuntimeError("Base model must be built before fine-tuning.")

        # Unfreeze base model
        self.base_model.trainable = True

        # Keep early layers frozen up to fine_tune_at_layer
        fine_tune_at = self.model_cfg.get("fine_tune_at_layer", 249)
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        trainable_layers = sum(1 for layer in self.base_model.layers if layer.trainable)
        logger.info(f"Fine-tuning enabled: {trainable_layers} layers unfrozen in InceptionV3.")

        lr = self.training_cfg.get("phase2", {}).get("learning_rate", 1e-5)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc", multi_label=True),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        logger.info(f"Model compiled for Phase 2 (Fine-Tuning, lr={lr})")

    def save(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is not None:
            self.model.save(filepath)
            logger.info(f"Saved model to: {filepath}")

    @staticmethod
    def load(filepath: str) -> tf.keras.Model:
        """Load a saved model checkpoint."""
        logger.info(f"Loading model checkpoint from: {filepath}")
        return tf.keras.models.load_model(filepath)
