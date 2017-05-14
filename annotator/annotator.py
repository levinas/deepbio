#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
import logging

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.layers import Input, Lambda, Layer, LSTM, Conv1D, GlobalMaxPooling1D, Flatten
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.callbacks import Callback, ModelCheckpoint, ProgbarLogger


logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


CHARS = ' atgc'
CHARLEN = len(CHARS)

# MAXLEN = 3000
MAXLEN = 3600

ACTIVATION = 'relu'
OPTIMIZER = 'adam'
EPOCHS = 1000
BATCH_SIZE = 128
DENSE_SIZE = 1024
LAYERS = 1
CLASSES = 100


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


def get_parser():
    parser = argparse.ArgumentParser(prog='dna2.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-a", "--activation",
                        default=ACTIVATION,
                        help="keras activation function to use in inner layers: relu, tanh, sigmoid...")
    parser.add_argument("-e", "--epochs", type=int,
                        default=EPOCHS,
                        help="number of training epochs")
    parser.add_argument('-l', '--log', dest='logfile',
                        default=None,
                        help="log file")
    parser.add_argument("-z", "--batch_size", type=int,
                        default=BATCH_SIZE,
                        help="batch size")
    parser.add_argument("--dense_size", type=int,
                        default=DENSE_SIZE,
                        help="number of neurons in intermediate dense layers")
    parser.add_argument("--layers", type=int,
                        default=LAYERS,
                        help="number of RNN layers to use")
    parser.add_argument("--optimizer",
                        default=OPTIMIZER,
                        help="keras optimizer to use: sgd, rmsprop, ...")
    parser.add_argument("--save",
                        default='save',
                        help="prefix of output files")
    parser.add_argument("--maxlen", type=int,
                        default=MAXLEN,
                        help="DNA chunk length")

    return parser


def load_data(maxlen=1000, val_split=0.2, batch_size=128):
    ctable = CharacterTable(CHARS, maxlen)

    df = pd.read_csv('ref.100ec.pgf.seqs.filter', sep='\t', engine='c', dtype={'genome':str})
    df_ref = pd.read_csv('ref.patric_ids', header=None, dtype=str)
    # ref_ids = df_ref[0].tolist()

    mask = df['genome'].isin(df_ref[0].sample(100))
    # df_train = df[mask]
    # df_val = df[~mask]

    n = df.shape[0]
    x = np.zeros((n, maxlen, CHARLEN), dtype=np.byte)
    for i, seq in enumerate(df['feature.na_sequence']):
        x[i] = ctable.encode(seq[:1000])

    y = pd.get_dummies(df.iloc[:, 0]).as_matrix()

    x_train = x[mask]
    x_val = x[~mask]

    y_train = y[mask]
    y_val = y[~mask]

    return (x_train, y_train), (x_val, y_val)


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    # ext += '.D={}'.format(args.drop)
    ext += '.E={}'.format(args.epochs)
    ext += '.O={}'.format(args.optimizer)
    ext += '.DS={}'.format(args.dense_size)
    ext += '.LEN={}'.format(args.maxlen)

    return ext


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fn=print):
        Callback.__init__(self)
        self.print_fn = print_fn

    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fn(msg)


def simple_model():
    model = Sequential()
    model.add(Conv1D(200, 3, padding='valid', activation='relu', strides=1, input_shape=(MAXLEN, CHARLEN)))
    # model.add(Flatten())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(100))
    model.add(Activation('softmax'))
    return model


def get_model(name, args):
    name = name.lower()
    if name == 'vgg' or name == 'vgg16':
        from vgg16_nt import VGG16NT
        model = VGG16NT(input_shape=(args.maxlen, CHARLEN), dense_size=args.dense_size, classes=CLASSES)
        name = 'vgg16'
    else:
        model = simple_model()
        name = 'simple'
    return model, name


def main():
    parser = get_parser()
    args = parser.parse_args()

    model, model_name = get_model('vgg', args)

    ext = extension_from_parameters(args)
    prefix = args.save + '.' + model_name + ext
    logfile = args.logfile if args.logfile else prefix + '.log'

    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)

    (x_train, y_train), (x_val, y_val) = load_data(maxlen=args.maxlen)

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=args.optimizer,
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(prefix + '.h5', save_best_only=True)

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_val, y_val),
              callbacks=[checkpointer,
                         LoggingCallback(logging.debug)])


if __name__ == '__main__':
    main()