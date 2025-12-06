import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_shape, num_classes: int):
    """
    Build a simple 1D CNN for posture classification on tabular sensor data.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Reshape((input_shape[0], 1)),
            layers.Conv1D(32, 3, activation="relu"),
            layers.MaxPooling1D(),
            layers.Conv1D(64, 3, activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
