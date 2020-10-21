import sys

import tensorflow.keras
import tensorflow as tf

get_ipython().run_line_magic("matplotlib", " widget")

import os
import librosa
import librosa.display
import IPython.display as ipd
from IPython.display import Audio

import matplotlib.pyplot as plt

import pathlib




data_dir = pathlib.Path.cwd()
print(data_dir)


x=data_dir.ls()
y=len(x.items)
for i in range(y):
    print(x[i].name)

#files = [print(x[i].name) for i in iter(x)]



len(list(data_dir.glob('*.*')))



