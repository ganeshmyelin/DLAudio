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


def prepare_datasets(test_size, validation_size):

    X, y = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,test_size=validation_size)

    # (3d array needed for CNN with h x w x depth) per sample
    X_train = X_train[..., np.newaxis]  # this is a 4d array : (num_samples, 130,13,1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Generates CNN model

    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))


    return model


def predict(model, X, y) :
    X = X[np.newaxis,...] # to add batch size of 1 to X.shape

    #prediction = [ [0.1, 0.2, ...]] 10 classes
    prediction = model.predict(X)  # X.shape -> (1,130,130, 13, 1)

    #extract index with max value
    predicted_index = np.argmax(prediction,axis=1) #index = 4
    print(f'Expected index: {y}, Predicted index: {predicted_index}')


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

    # split the data : train,validation,test

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)


    # build the network
    input_shape = (X_train.shape[1], X_train.shape[2],X_train.shape[3])
    model = build_model(input_shape)

    #compile
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # train model
    history = model.fit(X_train, y_train,
                        validation_data=(X_validation, y_validation),
                        epochs=30,
                        batch_size=32
                        )
    plot_history(history)

    # evaluate model on test set
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy on test set is {test_accuracy}')

    # predict on a sample (inference)
    X = X_test[100]
    print(X.shape)
    y = y_test[100]

    predict(model, X, y)

