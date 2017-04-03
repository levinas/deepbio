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


def load_data(maxlen=30):
    chars = 'atgc '
    ctable = CharacterTable(chars, maxlen)

    fname = '511145.12.PATRIC.ffn'
    seqs = SeqIO.parse(open(fname), 'fasta')
    # for record in seqs:
    #         name, seq = fasta.id, fasta.seq.tostring()
    #         new_sequence = some_function(sequence)
    #         write_fasta(out_file)


def main():
    load_data()


if __name__ == '__main__':
    main()
