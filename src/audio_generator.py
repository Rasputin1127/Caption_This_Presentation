import tensorflow as tf
import os
from scipy.io import wavfile
import librosa
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.audio import decode_wav
from tqdm import tqdm
from glob import glob
import argparse
import warnings
from pydub.utils import mediainfo


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, int(100000), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = librosa.load(path)
            resamp = librosa.resample(rate, wav, self.sr)
            fix_length_samp = librosa.util.fix_length(resamp, 40000)
            X[i,] = fix_length_samp.reshape(-1, 1)
            Y[i,] = label
            

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def get_wav_paths(wav_dir, clip):
    all_wav = []
    mask = []
    idx = 0
    for item in os.walk(wav_dir):
        if item[2]:
            for rec in item[2]:
                if rec.endswith('.wav'):
                        wav = item[0] + '/' + rec
                        sig, freq = librosa.load(wav)
                        if len(sig)/freq > clip:
                            idx += 1
                        else:
                            all_wav.append(wav)
                            mask.append(idx)
                            idx += 1

    return all_wav, mask

def get_labels(dir_path, mask):
    docs = []
    y = []
    for item in os.walk(dir_path):
        if item[2]:
            for rec in item[2]:
                if rec.endswith('.txt'):
                    file = open(f'{item[0]}/' + rec)
                    obj = file.readlines()
                    for _ in obj:
                        text = _.split(maxsplit=1)[1].rstrip('\r\n')
                        docs.append(text)
    for i in mask:
        y.append(docs[i])
    tfidf = TfidfVectorizer()
    y_fit = tfidf.fit_transform(y)
    vocab = tfidf.get_feature_names()
    labels = y_fit.toarray()
    labels[labels>0] = 1

    return labels, vocab

