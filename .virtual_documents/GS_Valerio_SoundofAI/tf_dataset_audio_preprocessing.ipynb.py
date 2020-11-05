


import sys

import tensorflow.keras
import tensorflow as tf

#get_ipython().run_line_magic("matplotlib", " widget")

import os
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
from IPython.display import Audio
import matplotlib.pyplot as plt
import pathlib

print(f'\n Tensorflow version : {tf.__version__}\n')

home_dir = pathlib.Path.cwd()
print(f'\n ** home_dir now is ** : {home_dir} ')
'''
print(f'\n ** home_dir.ls() ** \n ')
for i in range(len(home_dir.ls().items)):
    print(home_dir.ls()[i].name)
'''


data_dir = pathlib.Path(home_dir/'data/gtzan/genres_original')
print(data_dir)


list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
print(len(list_ds))
for i in list_ds.take(1):
    print(type(i))
for f in list_ds.take(1):
  x=f.numpy()
  print(type(x))


class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name get_ipython().getoutput("= ".DS_Store"]))")
print(class_names)
print(tf.data.experimental.cardinality(list_ds).numpy())


def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_wav(wav_file):
    pass


def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  wav_file = tf.io.read_file(file_path)
  audio = decode_wav(wav_file)
  return audio, label


list_ds = list_ds.map(process_path)

