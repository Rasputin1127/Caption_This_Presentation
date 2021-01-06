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
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import librosa


def get_data(dir_path, new_sr = 8000, final_length = 100000):
    samps = []
    y = []
    x = []
    for item in os.walk(dir_path):
        if item[2]:
            for rec in item[2]:
                if rec.endswith('.wav'):
                        samp, sig = librosa.load(item[0] + '/' + rec)
                        resamp = librosa.resample(samp, 16000, new_sr)
                        fix_length_samp = librosa.util.fix_length(resamp, final_length)
                        samps.append(fix_length_samp)
                elif rec.endswith('.txt'):
                    file = open(f'{item[0]}/' + rec)
                    obj = file.readlines()
                    for _ in obj:
                        text = _.split(maxsplit=1)[1].rstrip('\r\n')
                        y.append(text)
#     for item1,item2 in zip(sigs,samps):
#         x.append([item1[0:-1], item2])
#     y = np.array(y).reshape(-1,1)
    x = np.array(samps)
    samp_freq = new_sr
    return x, y, samp_freq


if __name__ == '__main__':
    dir_path = 'data/LibriSpeech/dev-clean/'
    X, Y, freq = get_data(dir_path)
    
    print('Y: ' + f'{Y[0]}')
    print("Done getting data")

    tfidf = TfidfVectorizer()
    # docs = [''.join(x[0]) for x in Y]
    tfidf.fit(Y[0:20])
    vocab = tfidf.get_feature_names()

    test_y = tfidf.transform(Y[0:20])
    targets = test_y.toarray()

    print("Making x and y complete")

    x_tr, x_val, y_tr, y_val = train_test_split(X,
                                            targets,
                                            test_size = 0.2,
                                            random_state=777,
                                            shuffle=True)

    K.clear_session()

    inputs = Input(shape=(100000,1))

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
                        monitor='val_acc',
                        verbose=1,
                        save_best_only=True,
                        mode='max')

    history = model.fit(x_tr, y_tr ,epochs=1000, callbacks=[es,mc], batch_size=32, validation_data=(x_val, y_val))