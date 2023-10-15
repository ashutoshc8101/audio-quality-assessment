#run
import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

import os
from scipy.io import wavfile

from spafe.utils import vis
from spafe.features.lfcc import lfcc
from spafe.features.pncc import pncc

from spafe.features.bfcc import bfcc
from spafe.features.lpc import lpc, lpcc
from spafe.features.gfcc import gfcc

import time
st = time.time()

Main_folder = os.path.join(os.getcwd(), "udm/")
dataset_path = os.path.join(os.getcwd(), "udm/")

def chroma_cqt_simple_mean_feature(audio, sample_rate):
  C = librosa.feature.chroma_cqt(y = audio, sr = sample_rate)
  C_mean = np.mean(C.T, axis = 0)

  return C_mean


def melspectrogram_mean_feature(audio, sample_rate):
  C = librosa.feature.melspectrogram(y = audio, sr = sample_rate)
  C_mean = np.mean(C.T, axis = 0)

  return C_mean


def mfcc_simple_mean_feature(audio, sample_rate ):
    mfccs_features = librosa.feature.mfcc(
      y = audio, sr = sample_rate, n_mfcc = 40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    return mfccs_scaled_features


def spectral_contrast_mean_feature(audio, sample_rate):
    S = np.abs(librosa.stft(audio))
    contrast = librosa.feature.spectral_contrast(
      S = S, sr = sample_rate)

    return np.mean(contrast, axis=0)

def spectral_centroid_mean_feature(audio, sample_rate):
    cent = librosa.feature.spectral_centroid(
      y = audio, sr = sample_rate)

    return np.mean(cent, axis=0)


def pncc_simple_mean_feature(audio, sample_rate):
  C = pncc(sig = audio, fs = sample_rate);
  C_mean = np.mean(C, axis=0)

  return C_mean


feature_names=["mfcc_simple_mean" , "chroma_simple_mean", "mel_simple_mean"]

Files = os.listdir("./udm/")

def read_audio_clip(file_name):
    audio,sample_rate = librosa.load(file_name)
    return audio, sample_rate

Features = defaultdict(list)


for i in range(0, len(Files)):
  print(dataset_path)
  print(Files[i])
  file_name = dataset_path + Files[i]

  quality = float(Files[i][:4])
  audio, sample_rate = read_audio_clip(file_name)

  if(i%100 == 0):
    print('{} data files have been read'.format(i))

  Features['mfcc_simple_mean_new'].append([mfcc_simple_mean_feature(audio, sample_rate ), quality])
  Features["chroma_cqt_simple_mean_new"].append([chroma_cqt_simple_mean_feature(audio, sample_rate), quality])
  Features["melspectogram_mean_new"].append([melspectrogram_mean_feature(audio, sample_rate), quality])
  Features['pncc_simple_mean'].append([pncc_simple_mean_feature(audio, sample_rate), quality])
  Features["spectral_centroid_mean"].append([spectral_centroid_mean_feature(audio, sample_rate), quality])


print("Feature generation phase complete")

end = time.time()
print((end - st) / len(Files))

feature_names = ["mfcc_simple_mean_new", "chroma_cqt_simple_mean_new", "melspectogram_mean_new"]

for fn in feature_names:
   print(fn)
   print(len(Features[fn]))
   #print(Features[fn])

feature_storage_folder = "./DatabaseDistorted/final_tii_all_finaldata/"+'features/';

features = [
  'melspectogram_mean_new',
  "chroma_cqt_simple_mean_new",
  "mfcc_simple_mean_new",
  "spectral_centroid_mean",
  "pncc_simple_mean"
]

for fn in features:
  extracted_features_df = pd.DataFrame(Features[fn], columns=['feature', 'class'])

  feature_list = extracted_features_df['feature'].tolist()
  class_list = extracted_features_df['class'].tolist()

  new_df = pd.DataFrame(feature_list)
  new_df['class'] = class_list
  feature_filepath = Path(feature_storage_folder + fn + "dm_2075.csv")  # change 998 to length of the data set just to differetiate feature according to the dataset
  feature_filepath.parent.mkdir(parents = True, exist_ok = True)
  new_df.to_csv(feature_filepath)
  print(fn, " csv writing complete")

