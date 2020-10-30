import os
import sys
import tensorflow.keras
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

SAMPLE_RATE = 22050
DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],  # ["classical","blues"...]
        "mfcc": [],  # [mfcc coefficients per segment]
        "labels": []  # [0..9] per segment
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all genre files
    # can get mappings from one command and put in data dict
    # can get all wav files into tf.dataset in 1 command
    # use map for doing operations on tf.dataset ? or just do librosa.mfcc on each element

    # class_mappings = np.array(sorted([item.name for item in dataset_path.glob('*')]))
    # class_mappings = sorted([item.name for item in dataset_path.glob('*')])
    class_mappings = [item.name for item in dataset_path.glob('*')]
    for i in range(len(class_mappings)):
        data["mapping"].append(class_mappings[i])

    list_ds = tf.data.Dataset.list_files(str(dataset_path /'*/*'), shuffle=False)

    for i, f in enumerate(list_ds.take(len(list_ds))):
        signal, sr = librosa.load(f.numpy(), sr=SAMPLE_RATE)
        # process segments extracting mfcc and storing data in
        for s in range(num_segments):
            start_sample = num_samples_per_segment * s
            end_sample = start_sample + num_samples_per_segment

            mfcc = librosa.feature.mfcc(signal[start_sample:end_sample],
                                        sr=sr,
                                        n_fft=n_fft,
                                        n_mfcc=n_mfcc,
                                        hop_length=hop_length)
            mfcc = mfcc.T # very important : transpose the 13 x expected_num_mfcc_vectors_per_segment

            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(int(i/10))
                print("{}, segment:{}".format(f.numpy(), s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)




if __name__ == '__main__':
    # print(f'\n Tfversion : {tf.__version__}\n')

    home_dir = pathlib.Path.cwd()
    print(f'\n ** home_dir now is ** : {home_dir} ')
    DATASET_PATH = pathlib.Path(home_dir / 'data/gtzan_reduced/genres_original')
    JSON_PATH = "data1.json"

    print(str(DATASET_PATH))

    save_mfcc(dataset_path=(DATASET_PATH), json_path=JSON_PATH, num_segments=10)

    # load dataset
    # split into train and validation
    # build netwrok architecture
    # compile model
    # train model

