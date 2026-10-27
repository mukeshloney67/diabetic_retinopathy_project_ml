"""
Image preprocessing and enhancement module for retinal fundus images.
Includes Ben Graham contrast enhancement and circular ROI cropping.
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple


def crop_retinal_roi(image: np.ndarray, tolerance: int = 10) -> np.ndarray:
    """
    Crop the circular Region of Interest (ROI) from retinal fundus images,
    eliminating surrounding black border margins.

    Args:
        image: RGB or BGR numpy array representing the retinal image.
        tolerance: Luminance threshold for detecting black border.

    Returns:
        Cropped numpy array of the retinal fundus.
    """
    if len(image.shape) == 2:
        mask = image > tolerance
        return image[np.ix_(mask.any(1), mask.any(0))]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image
    mask = gray > tolerance

    # If entire image is dark or invalid, return original
    if not np.any(mask):
        return image

    check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check_shape == 0:
        return image

    img1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]

    return np.stack([img1, img2, img3], axis=-1)


def ben_graham_enhancement(
    image: np.ndarray, target_size: Tuple[int, int] = (299, 299), sigma_x: int = 10
) -> np.ndarray:
    """
    Apply Ben Graham's retinal image enhancement method.
    Subtracts local average color (Gaussian blurred) to normalize illumination
    and significantly boost visibility of microaneurysms and hemorrhages.

    Args:
        image: RGB image numpy array.
        target_size: (width, height) to resize to.
        sigma_x: Standard deviation for Gaussian kernel.

    Returns:
        Enhanced image array scaled to uint8 [0, 255].
    """
    # Resize to target resolution
    img = cv2.resize(image, target_size)

    # Convert to BGR for OpenCV processing
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Ben Graham formula: 4 * img - 4 * GaussianBlur(img) + 128
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigma_x)
    enhanced = cv2.addWeighted(bgr, 4.0, blurred, -4.0, 128)

    # Return to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return enhanced_rgb


def inception_preprocess(img_array: np.ndarray) -> np.ndarray:
    """
    Apply InceptionV3 specific pixel normalization: scales input to [-1, 1].

    Args:
        img_array: Image array with values in [0, 255].

    Returns:
        Normalized array in range [-1, 1].
    """
    return tf.keras.applications.inception_v3.preprocess_input(img_array.astype(np.float32))
