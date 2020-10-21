import sys

import tensorflow.keras
import tensorflow as tf

get_ipython().run_line_magic("matplotlib", " widget")

import os
from fastai.basics import *
from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
import datetime 
import pytz  

#get_ipython().getoutput("pip show fastai")
#print(datetime.datetime.now(pytz.timezone('Asia/Kolkata')))

from fastaudio.core.all import *
from fastaudio.augment.all import *

import librosa
import librosa.display
import IPython.display as ipd
from IPython.display import Audio

import matplotlib.pyplot as plt

import pathlib 
from pathlib import Path

#home_dir = Path(os.getcwd())
home_dir = Path.cwd()
print(f'\n ** home_dir is ** : {home_dir} ')
print(f'\n ** home_dir.ls() ** \n ')
for i in range(len(home_dir.ls().items)):
    print(home_dir.ls()[i].name)



#data_dir = (home_dir/'data/gtzan/genres_original').ls()
data_dir = Path(home_dir/'data/gtzan/genres_original')


#Create tf.data.Dataset from list of files
#list_all_files = list(data_dir.glob('*/*'))
files_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
print(f'Type of files_ds {type(files_ds)}') 
print(f'# of files in data_dir is {len(files_ds)}')
#print(list_all_files[0:5])
print(f'first 2 files in Dataset are \n {[f.numpy() for f in files_ds.take(2)]}')



class_names = np.array(sorted(
    [item.name for item in data_dir.glob('*') if item.name get_ipython().getoutput("= ".DS_Store"])")
                      )
print(f"Classes are {class_names}")


#class_indices = [i+1 for i,_ in enumerate(class_names)]
#print(f'Labels are {class_indices}')


#TESTING 

parts = tf.strings.split(Tensor(files_ds), os.path.sep)











#Convert file path to an (wav,label) pair

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_wav(audio):
    pass 
    '''
    audio, sample_rate = tf.audio.decode_wav(wav,
                                             desired_channels=1,
                                             desired_samples=44100)
    return audio,sample_rate
    
    '''
    



def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    '''
    audio = tf.io.read_file(file_path)
    img = decode_img(img)
    '''
    return label


AUTOTUNE = tf.data.experimental.AUTOTUNE

files_ds = files_ds.map(process_path, num_parallel_calls=AUTOTUNE)


for i in files_ds.take(1):
    print(files_ds.label.numpy())





labels_ds = tf.data.Dataset.from_tensor_slices(class_indices)
    
    return tf.data.Dataset.zip((files_ds, labels_ds))



ds = get_dataset(list_all_files,class_indices)


ds



