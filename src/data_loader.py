"""
Data loader and generator pipeline for training, validation, and testing.
Handles augmentation and class weight calculations for imbalanced retinal datasets.
"""

import os
import logging
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.preprocessing import inception_preprocess

logger = logging.getLogger(__name__)


class RetinalDataLoader:
    """
    Data loader providing flow_from_directory generators for retinal image datasets.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.data_cfg = config.get("data", {})
        self.aug_cfg = config.get("augmentation", {})
        self.image_size = (self.data_cfg.get("image_size", 299), self.data_cfg.get("image_size", 299))
        self.train_dir = self.data_cfg.get("train_dir", "data/train")
        self.val_dir = self.data_cfg.get("validation_dir", "data/validation")
        self.test_dir = self.data_cfg.get("test_dir", "data/test")

    def get_train_generator(self, batch_size: int = 32) -> ImageDataGenerator:
        """
        Create training generator with extensive data augmentation.
        """
        train_datagen = ImageDataGenerator(
            preprocessing_function=inception_preprocess,
            rotation_range=self.aug_cfg.get("rotation_range", 30),
            width_shift_range=self.aug_cfg.get("width_shift_range", 0.15),
            height_shift_range=self.aug_cfg.get("height_shift_range", 0.15),
            shear_range=self.aug_cfg.get("shear_range", 0.15),
            zoom_range=self.aug_cfg.get("zoom_range", 0.2),
            horizontal_flip=self.aug_cfg.get("horizontal_flip", True),
            vertical_flip=self.aug_cfg.get("vertical_flip", True),
            fill_mode=self.aug_cfg.get("fill_mode", "nearest"),
        )

        return train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=self.config.get("project", {}).get("seed", 42),
        )

    def get_val_generator(self, batch_size: int = 32) -> ImageDataGenerator:
        """
        Create validation generator (no augmentation, only normalization).
        """
        val_datagen = ImageDataGenerator(
            preprocessing_function=inception_preprocess
        )

        return val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )

    def get_test_generator(self, batch_size: int = 32) -> ImageDataGenerator:
        """
        Create test generator (no augmentation, sequential for evaluation).
        """
        test_datagen = ImageDataGenerator(
            preprocessing_function=inception_preprocess
        )

        return test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )

    @staticmethod
    def calculate_class_weights(generator) -> Dict[int, float]:
        """
        Compute inverse-frequency balanced class weights to compensate for severe
        class imbalance in medical retinopathy datasets.

        Args:
            generator: Training directory iterator.

        Returns:
            Dictionary mapping class index to weight.
        """
        classes = generator.classes
        unique_classes = np.unique(classes)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=classes
        )
        return dict(zip(unique_classes.astype(int), weights.astype(float)))
