"""InceptionV3 model construction and fine-tuning utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import ProjectConfig

if TYPE_CHECKING:
    import tensorflow as tf


def configure_tensorflow(config: ProjectConfig) -> None:
    """Configure reproducibility and optional mixed precision before model creation."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(config.training.seed)
    if config.training.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")


def build_model(config: ProjectConfig) -> "tf.keras.Model":
    """Build a saved-model-safe InceptionV3 classifier with preprocessing inside."""
    import tensorflow as tf

    layers = tf.keras.layers
    inputs = layers.Input(shape=(*config.model.input_size, 3), name="fundus_image")
    x = inputs
    augmentation = config.model.augmentation
    augmenters: list[tf.keras.layers.Layer] = []
    if augmentation.horizontal_flip:
        augmenters.append(layers.RandomFlip("horizontal", seed=config.training.seed))
    if augmentation.rotation_factor:
        augmenters.append(
            layers.RandomRotation(augmentation.rotation_factor, seed=config.training.seed + 1)
        )
    if augmentation.zoom_height_factor or augmentation.zoom_width_factor:
        augmenters.append(
            layers.RandomZoom(
                height_factor=augmentation.zoom_height_factor,
                width_factor=augmentation.zoom_width_factor,
                seed=config.training.seed + 2,
            )
        )
    if augmentation.contrast_factor:
        augmenters.append(
            layers.RandomContrast(augmentation.contrast_factor, seed=config.training.seed + 3)
        )
    if augmenters:
        x = tf.keras.Sequential(augmenters, name="augmentation")(x)

    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="inception_preprocessing")(x)
    weights = config.model.pretrained_weights
    if isinstance(weights, str) and weights.lower() in {"none", "null", ""}:
        weights = None
    backbone = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=weights,
        input_shape=(*config.model.input_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(config.model.dropout_rate, name="classifier_dropout")(x)
    x = layers.Dense(config.model.dense_units, activation="relu", name="classifier_dense")(x)
    outputs = layers.Dense(
        config.model.num_classes,
        activation="softmax",
        dtype="float32",
        name="severity",
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dr_inception_v3")
    compile_model(model, config.training.initial_learning_rate)
    return model


def compile_model(model: "tf.keras.Model", learning_rate: float) -> None:
    """Compile with sparse labels and metrics appropriate for a multi-class task."""
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def fine_tune_top_fraction(model: "tf.keras.Model", fraction: float) -> None:
    """Unfreeze upper InceptionV3 layers while keeping batch normalization frozen."""
    import tensorflow as tf

    backbone = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model) and layer.name.startswith("inception_v3")
        ),
        None,
    )
    if backbone is None:
        raise RuntimeError("Could not find the InceptionV3 backbone in the model.")

    backbone.trainable = True
    start_index = int(len(backbone.layers) * (1 - fraction))
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= start_index and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
