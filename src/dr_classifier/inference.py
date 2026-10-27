"""Single-image prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig


def predict_image(model_path: str | Path, image_path: str | Path, config: ProjectConfig) -> dict[str, Any]:
    """Predict one image and return every grade probability."""
    import tensorflow as tf

    model_path = Path(model_path).resolve()
    image_path = Path(image_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    encoded = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, config.model.input_size, antialias=True)
    image = tf.cast(image, tf.float32)
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "image_path": str(image_path),
        "model_path": str(model_path),
        "predicted_grade": predicted_index,
        "predicted_class": config.model.class_names[predicted_index],
        "referable_dr": bool(predicted_index >= 2),
        "probabilities": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(config.model.class_names)
        },
        "notice": (
            "Research-only output. Softmax scores are not clinical certainty and "
            "must not be used as a diagnosis or treatment recommendation."
        ),
    }
