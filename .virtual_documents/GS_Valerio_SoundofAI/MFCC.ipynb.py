get_ipython().run_line_magic("matplotlib", " widget")

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


files = (path/'data/16').ls()
duke,_ = librosa.load(files[0])
redhot,_ = librosa.load(files[1])
scale,sr = librosa.load(files[2])
voice,_ = librosa.load(files[3])
debussy,_ = librosa.load(files[4])
noise,_ = librosa.load(files[5])



ipd.Audio(files[4])


debussy.shape


#EXTRACT MFCCs
mfccs = librosa.feature.mfcc(debussy,n_mfcc=13,sr=sr)


mfccs.shape


plt.figure()
librosa.display.specshow(mfccs,x_axis='time',sr=sr)
plt.colorbar(format='get_ipython().run_line_magic("2f')", "")


#calculate delta, delta2 MFCCs
delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs,order=2)


delta2_mfccs.shape


plt.figure()
librosa.display.specshow(delta_mfccs,x_axis='time',sr=sr)
plt.colorbar(format='get_ipython().run_line_magic("2f')", "")


plt.figure()
librosa.display.specshow(delta2_mfccs,x_axis='time',sr=sr)
plt.colorbar(format='get_ipython().run_line_magic("2f')", "")


comprehensive_mfccs = np.concatenate([mfccs,delta_mfccs,delta2_mfccs])


comprehensive_mfccs.shape



