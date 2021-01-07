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

        inputs = Input(shape=(40000,1))

        #First Conv1D layer
        conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
        conv = MaxPooling1D(3)(conv)
        conv = Dropout(0.2)(conv)

        #Second Conv1D layer
        conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
        conv = MaxPooling1D(3)(conv)
        conv = Dropout(0.2)(conv)

        #Third Conv1D layer
        conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
        conv = MaxPooling1D(3)(conv)
        conv = Dropout(0.2)(conv)

        #Fourth Conv1D layer
        conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
        conv = MaxPooling1D(3)(conv)
        conv = Dropout(0.2)(conv)

        #Flatten layer
        conv = Flatten()(conv)

        #Dense Layer 1
        conv = Dense(256, activation='relu')(conv)
        conv = Dropout(0.3)(conv)

        #Dense Layer 2
        conv = Dense(128, activation='relu')(conv)
        conv = Dropout(0.3)(conv)

        outputs = Dense(len(vocab), activation='softmax')(conv)

        model = Model(inputs, outputs)
        model.summary()

        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

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