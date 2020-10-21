import sys

import tensorflow.keras
import tensorflow as tf

get_ipython().run_line_magic("matplotlib", " widget")

import os
from fastai.basics import *
#get_ipython().run_line_magic("matplotlib", " widget")

import os
from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
import datetime 
import pytz  

get_ipython().getoutput("pip show fastai")
print(datetime.datetime.now(pytz.timezone('Asia/Kolkata')))


from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *

import librosa
import librosa.display
import IPython.display as ipd
from IPython.display import Audio

import matplotlib.pyplot as plt

path = Path(os.getcwd())
print(f'\n ** path is ** : {path} ')
print(f'\n ** path.ls() ** \n {path.ls()}')


print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")


files = (path/'data/16').ls()


files


duke,_ = librosa.load(files[0])
redhot,_ = librosa.load(files[1])
scale,sr = librosa.load(files[2])
voice,_ = librosa.load(files[3])
debussy,_ = librosa.load(files[4])
noise,_ = librosa.load(files[5])




