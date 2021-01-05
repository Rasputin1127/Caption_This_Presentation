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
from tensorflow.keras.layers import BatchNormalization, SimpleRNN, Input, TimeDistributed, Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential, Model

def get_data(dir_path):
    sigs = []
    samps = []
    y = []
    x = []
    for item in os.walk(dir_path):
        if item[2]:
            for rec in item[2]:
                if rec.endswith('.wav'):
                        samp, sig = wavfile.read(item[0] + '/' + rec)
                        samps.append(samp)
                        sigs.append(sig)
                elif rec.endswith('.txt'):
                    file = open(f'{item[0]}/' + rec)
                    obj = file.readlines()
                    for _ in obj:
                        text = _.split(maxsplit=1)[1].rstrip('\r\n')
                        y.append(text)
#     for item1,item2 in zip(sigs,samps):
#         x.append([item1[0:-1], item2])
#     y = np.array(y).reshape(-1,1)
    x = np.array(sigs)
    samp_freq = samps[0]
    return x, y, samp_freq


if __name__ == '__main__':
    dir_path = 'data/LibriSpeech/dev-clean/'
    X, Y, freq = get_data(dir_path)
    short_x = tf.ragged.constant(X[0:20])
    print('Y: ' + f'{Y[0]}')
    print("Done getting data")

    tfidf = TfidfVectorizer()
    # docs = [''.join(x[0]) for x in Y]
    tfidf.fit(docs)
    vocab = tfidf.get_feature_names()

    test_y = tfidf.transform(docs)
    targets = test_y.toarray()

    print("Making x and y complete")

    input1 = Input(shape=(None,), ragged=True)
    x = Embedding(8173, 2703)(input1)
    x = LSTM(64)(x)
    out = Dense(8173, activation='softmax')(x)
    model = Model(inputs=[input1], outputs=[out])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(short_x, targets[0:20], epochs=10)