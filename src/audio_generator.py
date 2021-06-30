import tensorflow as tf
import os
import librosa
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, n_classes,
                 batch_size=32, shuffle=True):
        '''
        Tensorflow Sequence class to generate and feed audio files and labels into models for training.

        Parameters:
        wav_paths: List --> list of all full-paths to .wav files for training
        labels: List --> list of all text that accompanies .wav files for training
        sr: Int --> desired sample rate for final .wav file data before training
        n_classes: Int --> number of classes that will exist in y-label set
        batch_size: Int --> desired batch size for training
        shuffle: Bool --> whether to shuffle the training data or not

        Output:
        Two tensors each time it is called of X = (batch size, 40000, 1) and Y = (batch size, n classes)
        '''

        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        '''
        Returns number of batches required per epoch.
        '''

        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        '''
        Generates the X, Y tensors for training when called by Tensorflow's API.

        Paramters:
        index: Int --> indicates which batch is being called, sets indexing appropriately

        Output:
        X, Y tensors where X is the .wav file data appropriately preprocessed and Y is the accompanying labels
        '''

        # sets indexes for current batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # grabs the appropriate .wav files and accompanying labels
        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data holders for audio files and labels
        X = np.empty((self.batch_size, int(100000), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        # loads file, resamples down to appropriate rate, and fixes length of tensors for feeding into model
        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = librosa.load(path)
            resamp = librosa.resample(rate, wav, self.sr)
            fix_length_samp = librosa.util.fix_length(resamp, 40000)
            X[i, ] = fix_length_samp.reshape(-1, 1)
            Y[i, ] = label

        return X, Y

    # if shuffle is True, shuffles the data after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


# helper function that uses the directory of wav files to populate a master audio files list
# clip is used so that only audio files under a specified duration will be grabbed
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

    # returns a mask used in get_labels so that the appropriate labels are grabbed
    return all_wav, mask

# helper function that uses the directory of wav files to populate a master labels list
# mask is used to ensure labels and audio files match up appropriately


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
    labels[labels > 0] = 1

    # vocab is returned so that some models I was testing would have a feature list
    return labels, vocab
