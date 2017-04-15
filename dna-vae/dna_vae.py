#! /usr/bin/env python

from __future__ import print_function

import numpy as np

from Bio import SeqIO

from keras.models import Model
from keras.layers import Input, Activation, Dense
from keras.layers import TimeDistributed, RepeatVector, LSTM, recurrent
from keras.layers import Conv1D, Lambda, Flatten
from keras import backend as K


epochs = 100
batch_size = 100
maxlen = 60
filters = 10
intermediate_dim = 500
latent_dim = 100
epsilon_std = 1.0
rnn_layers = 1
chars = 'atgc '


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


def load_data(maxlen=30, head_only=False, val_split=0.2):
    chars = 'atgc '
    ctable = CharacterTable(chars, maxlen)

    fname = '511145.12.PATRIC.ffn'
    fasta = SeqIO.parse(open(fname), 'fasta')

    if head_only:
        seqs = [str(s.seq)[:maxlen].lower() for s in fasta]
    else:
        seqs = [str(s.seq)[i:i+10].lower() for s in fasta for i in range(0, len(str(s.seq)), maxlen)]

    np.random.shuffle(seqs)

    X = np.zeros((len(seqs), maxlen, len(chars)), dtype=np.byte)
    for i, seq in enumerate(seqs):
        X[i] = ctable.encode(seq)

    # train_size = int(len(seqs) * (1 - val_split))
    train_size = int(len(seqs) * (1 - val_split) / batch_size) * batch_size
    val_size = int(len(seqs) * val_split / batch_size) * batch_size
    x_train = X[:train_size]
    x_val = X[train_size:train_size+val_size]

    return (x_train, x_val), (x_train, x_val)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon


def main():

    (x_train, x_val), (y_train, y_val) = load_data(maxlen=maxlen)

    x = Input(batch_shape=(batch_size, maxlen, len(chars)))
    h = Conv1D(filters, kernel_size=3, activation='relu')(x)
    h = Flatten()(h)
    h = Dense(intermediate_dim, activation='relu')(h)

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    h = Dense(intermediate_dim, activation='relu')(z)
    h = RepeatVector(maxlen)(h)
    for _ in range(rnn_layers):
        h = LSTM(intermediate_dim, return_sequences=True)(h)

    h = TimeDistributed(Dense(len(chars), activation='softmax'))(h)
    # h = Activation('softmax')(h)

    vae = Model(x, h)
    vae.summary()

    vae.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    vae.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_val, y_val))


if __name__ == '__main__':
    main()
