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
DURATION = 30 #in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path,json_path,n_mfcc=13,n_fft=2048,hop_length=512,num_segments=5):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)


    # loop through all genre files
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure that we are not at root level
        if dirpath is not dataset_path:
            # save semantic_label in mappings of data dictionary
            dirpath_components = dirpath.split("/")  # genre/blues => ["genres","blues"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            # process files for specific gnere
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting mfcc and storing data in
                for s in range(num_segments):
                    start_sample = num_samples_per_segment*s
                    end_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:end_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment :
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s))

    with open(json_path,"w") as fp :
        json.dump(data, fp, indent =4)


if __name__ == '__main__':

    # print(f'\n Tfversion : {tf.__version__}\n')

    home_dir = pathlib.Path.cwd()
    print(f'\n ** home_dir now is ** : {home_dir} ')
    DATASET_PATH = pathlib.Path(home_dir/'data/gtzan_reduced/genres_original')
    JSON_PATH = "data.json"

    print(str(DATASET_PATH))

    save_mfcc(dataset_path=str(DATASET_PATH), json_path=JSON_PATH, num_segments=10)