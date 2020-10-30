# To add a new cell, type '# %%'
import sys
import tensorflow.keras
import tensorflow as tf
import tensorflow_io as tfio

#%matplotlib widget
import os
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
from IPython.display import Audio
import simpleaudio as sa
import matplotlib.pyplot as plt
import pathlib

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


def get_label(file_path):

    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


def load_wav(file_path):

    # Load one second of audio at 44.1kHz sample-rate
    # audio = tf.io.read_file(file_path) #returns tensor of type string
    audio = tfio.audio.AudioIOTensor(file_path)
    # audio, sample_rate = tf.audio.decode_wav(audio,
    #                                         desired_channels=1,
    #                                         desired_samples=44100)
    audio_slice = audio[44100:]
    audio_tensor = tf.squeeze(audio_slice, axis=[-1])
    return audio_tensor, audio.rate


if __name__ == '__main__':

    # print(f'\n Tfversion : {tf.__version__}\n')

    home_dir = pathlib.Path.cwd()
    print(f'\n ** home_dir now is ** : {home_dir} ')
    data_dir = pathlib.Path(home_dir/'data/gtzan/genres_original')
    print(data_dir)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    print(len(list_ds))

    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"]))
    print(class_names)
    print(tf.data.experimental.cardinality(list_ds).numpy())

    x = []
    # for i in list_ds.as_numpy_iterator():

    for i in list_ds.take(len(list_ds)):
        z = get_label(i)
        x.append(z.numpy())    

    audio_list = []
    for i in list_ds.take(1):
        a, r = load_wav(i)

    # Audio(a.numpy(), rate=r.numpy())
    play_obj = sa.play_buffer(a.numpy(), 1, 2, sample_rate=r.numpy())
    play_obj.wait_done()

