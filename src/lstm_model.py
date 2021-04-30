import tensorflow as tf
from tf.keras.models import Model
from tf.keras import layers
from tf.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.regularizers import l2
import kapre
from kapre.composed import get_melspectrogram_layer

def LSTM(N_CLASSES, SR=8000, DT=5.0):
    input_shape = (int(SR*DT), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                     n_mels=128,
                                     pad_end=True,
                                     n_fft=512,
                                     win_length=400,
                                     hop_length=160,
                                     sample_rate=SR,
                                     return_decibel=True,
                                     input_data_format='channels_last',
                                     output_data_format='channels_last',
                                     name='2d_convolution')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = TimeDistributed(layers.Reshape((-1,)), name='reshape')(x)
    s = TimeDistributed(layers.Dense(64, activation='tanh'),
                        name='td_dense_tanh')(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
                             name='bidirectional_lstm')(s)
    x = layers.concatenate([s, x], axis=2, name='skip_connection')
    x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
    x = layers.MaxPooling1D(name='max_pool_1d')(x)
    x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(32, activation='relu',
                         activity_regularizer=l2(0.001),
                         name='dense_3_relu')(x)
    o = layers.Dense(N_CLASSES, activation='sigmoid', name='softmax')(x)
    model = Model(inputs=i.input, outputs=o, name='long_short_term_memory')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def deep_LSTM(n_classes, sr=8000, dt=10.0):
    input_shape = (int(sr*dt), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                    n_mels=128,
                                    pad_end=True,
                                    n_fft=512,
                                    win_length=400,
                                    hop_length=160,
                                    sample_rate=sr,
                                    return_decibel=True,
                                    input_data_format='channels_last',
                                    output_data_format='channels_last',
                                    name='spec_layer')
    x = LayerNormalization(axis=2, name_'batch_norm')(i.output)
    x = TimeDistributed(layers.Reshape((-1)), name='reshape')(x)
    s1 = TimeDistributed(layers.Dense(128, activation='tanh'), name='td_dense1_tanh')(x)
    x = layers.BiDirectional(layers.LSTM(64, return_sequences=True), name='bidirectional_lstm1')(s1)
    x = layers.concatenate([s1,x], axis=2, name='skip_connection1')
    s2 = TimeDistributed(layers.Dense(256, activation='tanh'), name='td_dense2_tanh')(x)
    x = layers.BiDirectional(layers.LSTM(128, return_sequences=True), name='bidirectional_lstm2')(s2)
    x = layers.concatenate([s2,x], axis=2, name='skip_connection2')
    s3 = TimeDistributed(layers.Dense(512, activation='tanh'), name='td_dense3_tanh')(x)
    x = layers.BiDirectional(layers.LSTM(256, return_sequences=True), name='bidirectional_lstm3')(s3)
    x = layers.concatenate([s3,x], axis=2, name='skip_connection3')
    x = layers.Dense(512, activation='relu', name='dense_1_relu')(x)
    x = layers.Dense(256, activation='relu', name='dense_2_relu')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(256, activation='relu', activity_regularizer=l2(0.001), name="dense_regularizer")(x)
    o = layers.Dense(n_classes, activation='sigmoid', name='sigmoid')(x)
    model = Model(inputs=i.input, outputs=o, name='deep_lstm')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model