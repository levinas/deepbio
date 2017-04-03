#! /usr/bin/env python

from __future__ import print_function

import numpy as np

from Bio import SeqIO

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent



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


def load_data(maxlen=30, val_split=0.2):
    chars = 'atgc '
    ctable = CharacterTable(chars, maxlen)

    fname = '511145.12.PATRIC.ffn'
    fasta = SeqIO.parse(open(fname), 'fasta')
    seqs = [str(x.seq)[:maxlen].lower() for x in fasta]
    np.random.shuffle(seqs)

    X = np.zeros((len(seqs), maxlen, len(chars)), dtype=np.byte)
    for i, seq in enumerate(seqs):
        X[i] = ctable.encode(seq)

    train_size = int(len(seqs) * (1 - val_split))
    x_train = X[:train_size]
    x_val = X[train_size:]

    return (x_train, x_val), (x_train, x_val)



def main():
    RNN = recurrent.LSTM

    MAXLEN = 30
    LAYERS = 1
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128

    chars = 'atgc '
    (x_train, x_val), (y_train, y_val) = load_data(maxlen=MAXLEN)

    model = Sequential()
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    model.add(RepeatVector(MAXLEN))

    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    model.add(TimeDistributed(Dense(len(chars))))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=10,
              validation_data=(x_val, y_val))


if __name__ == '__main__':
    main()
