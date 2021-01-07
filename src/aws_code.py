import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import wavfile
import os
from tensorflow.audio import decode_wav
from tensorflow.io import read_file
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from scipy.signal import spectrogram
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import librosa
from audio_generator import DataGenerator, get_wav_paths, get_labels
from lstm_model import LSTM




if __name__ == '__main__':
    dir_path = '../data/LibriSpeech/dev-clean/'
    all_wavs, mask = get_wav_paths(dir_path, 5)
    labels, vocab = get_labels(dir_path, mask)
    
    with open('paths.pkl', 'wb') as f:
        pickle.dump([all_wavs, mask])

    datagen = DataGenerator(all_wavs, labels, 8000, 5, len(vocab), batch_size=32, shuffle=True)

    # with open('variables.pkl', 'rb') as f:
        # X,_ = pickle.load(f)

    print("Making x and y complete")

    # x_tr, x_val, y_tr, y_val = train_test_split(X,
                                            # targets,
                                            # test_size = 0.2,
                                            # random_state=777,
                                            # shuffle=True)

    with tf.device('/:XLA_GPU:0'):

        K.clear_session()

        model = LSTM(len(vocab))

        model.summary()

        es = EarlyStopping(monitor='val_loss',
                        mode='min', 
                        verbose=1, 
                        patience=50, 
                        min_delta=0.0001)
        mc = ModelCheckpoint('best_model.hdf5',
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

        history = model.fit(datagen, epochs=1000, callbacks=[es,mc], batch_size=32, verbose=1)