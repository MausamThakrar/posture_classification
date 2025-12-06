import tensorflow as tf
from tensorflow.keras import layers, models

def build_rnn(input_shape, num_classes: int):
    """
    Build a simple RNN (LSTM) model for posture classification.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Reshape((input_shape[0], 1)),
            layers.LSTM(64),
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
