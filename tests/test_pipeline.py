"""
Unit tests for Diabetic Retinopathy Machine Learning Pipeline.
Tests configuration loading, image preprocessing, and model architecture.
"""

import os
import pytest
import numpy as np
from src.config_loader import load_config
from src.preprocessing import ben_graham_enhancement, crop_retinal_roi, inception_preprocess


def test_config_loader():
    """Verify config YAML is loaded with required keys."""
    config = load_config("config/config.yaml")
    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert config["data"]["num_classes"] == 5
    assert config["model"]["architecture"] == "InceptionV3"


def test_ben_graham_enhancement():
    """Verify Ben Graham filter output dimensions and values."""
    dummy_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    enhanced = ben_graham_enhancement(dummy_img, target_size=(299, 299), sigma_x=10)

    assert enhanced.shape == (299, 299, 3)
    assert enhanced.dtype == np.uint8


def test_crop_retinal_roi():
    """Verify circular ROI crop strips borders correctly."""
    # Create image with outer black margin
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[50:250, 50:250] = 180  # Central region

    cropped = crop_retinal_roi(img, tolerance=10)
    assert cropped.shape[0] < 300
    assert cropped.shape[1] < 300


def test_inception_preprocessing_range():
    """Verify InceptionV3 scaling normalizes to [-1, 1]."""
    sample = np.array([[[0, 127.5, 255]]], dtype=np.float32)
    processed = inception_preprocess(sample)

    assert processed.min() >= -1.05
    assert processed.max() <= 1.05
