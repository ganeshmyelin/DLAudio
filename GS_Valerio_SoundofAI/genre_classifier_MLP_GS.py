import os
import sys
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_io as tfio

import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
from IPython.display import Audio
import simpleaudio as sa
import matplotlib.pyplot as plt
import pathlib
import math
import json

from sklearn.model_selection import train_test_split

SAMPLE_RATE = 22050
DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

DATASET_PATH = "data1.json"


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    # convert lists to numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def plot_history(history):
    fig, axes = plt.subplots(2)
    axes[0].plot(history.history["accuracy"],label="train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="test accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Accuracy Eval")

    axes[1].plot(history.history["loss"], label="train error (or loss)")
    axes[1].plot(history.history["val_loss"], label="test error (or loss)")
    axes[1].set_ylabel("Error (or Loss)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Error (or Loss) Eval")

    plt.show()

if __name__ == "__main__":
    # load data
    X, y = load_data(DATASET_PATH)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build the network
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1],X.shape[2])),
        keras.layers.Dense(512,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10,activation='softmax')
    ])

    #compile
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # train model
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=50,
                        batch_size=32,
                        verbose=2)
    plot_history(history)

