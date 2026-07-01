import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model

from models.attention import AttentionLayer
from configs.config import (
    SEQUENCE_LENGTH,
    DROPOUT,
    NUM_CLASSES,
)


def build_model():

    inputs = layers.Input(
        shape=(SEQUENCE_LENGTH, 1)
    )

    x = layers.Conv1D(
        32,
        5,
        activation="relu",
        padding="same",
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(
        64,
        5,
        activation="relu",
        padding="same",
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(

        layers.LSTM(
            64,
            return_sequences=True,
        )

    )(x)

    x = AttentionLayer()(x)

    x = layers.Dense(
        64,
        activation="relu",
    )(x)

    x = layers.Dropout(
        DROPOUT
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
    )(x)

    model = Model(
        inputs,
        outputs,
        name="CNN_BiLSTM_Attention",
    )

    return model