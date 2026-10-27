"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for InceptionV3.
Generates explainable visual heatmaps highlighting retinal lesions and regions
influencing the model's diagnostic predictions.
"""

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "mixed10",
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a given input image and InceptionV3 model.

    Args:
        img_array: Normalized input image array of shape (1, 299, 299, 3).
        model: Trained InceptionV3 model.
        last_conv_layer_name: Name of target convolutional layer in InceptionV3 (default: 'mixed10').
        pred_index: Class index to compute gradients for (None for argmax predicted class).

    Returns:
        2D numpy array representing the normalized heatmap in range [0, 1].
    """
    # Isolate the InceptionV3 backbone if wrapped inside a sequential/functional model
    try:
        inception_layer = model.get_layer("inception_v3")
    except ValueError:
        inception_layer = None

    if inception_layer is not None:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[inception_layer.get_layer(last_conv_layer_name).output, model.output]
        )
    else:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output class score with respect to the feature map activations
    grads = tape.gradient(class_channel, last_conv_output)

    # Global average pooling of gradients gives feature map weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only positive evidence
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(
    original_img_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Superimpose the Grad-CAM heatmap onto the original retinal image.

    Args:
        original_img_path: Path to original RGB retinal image.
        heatmap: 2D array output from make_gradcam_heatmap.
        alpha: Heatmap transparency factor.
        colormap: OpenCV colormap.

    Returns:
        Superimposed RGB image as uint8 numpy array.
    """
    orig_img = Image.open(original_img_path).convert("RGB")
    orig_img = np.array(orig_img)

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Blend original and heatmap
    superimposed = cv2.addWeighted(orig_img, 1.0 - alpha, colored_heatmap, alpha, 0)
    return superimposed
