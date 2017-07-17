#! /usr/bin/env python

from __future__ import print_function

import argparse
import functools
import logging
import os

import numpy as np
import pandas as pd
import keras
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Activation, Conv1D, Dense, GlobalMaxPooling1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


CHARS = ' atgc'
CHARLEN = len(CHARS)

# MAXLEN = 3000
MAXLEN = 60 * 60
# MAXLEN = 36 * 36

ACTIVATION = 'relu'
OPTIMIZER = 'sgd'
EPOCHS = 100
BATCH_SIZE = 100
DENSE_LAYERS = [0]
LAYERS = 1
CLASSES = 100
DROPOUT = 0.5
LEARNING_RATE = None
SEED = 2017
DATA = '1K'
MODEL = 'res'
MODEL_VARIATION = 'v1'
SAVE = 'save/save'


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

    def encode(self, C, maxlen=None, snake2d=False):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        if snake2d:
            a = int(np.sqrt(maxlen))
            X2 = np.zeros((a, a, len(self.chars)))
            for i in range(a):
                for j in range(a):
                    k = i * a
                    k += a - j - 1 if i % 2 else j
                    X2[i, j] = X[k]
            X = X2
        return X

    def decode(self, X, snake2d=False):
        X = X.argmax(axis=-1)
        if snake2d:
            a = X.shape[0]
            X2 = np.zeros(a * a)
            for i in range(a):
                for j in range(a):
                    k = i * a
                    k += a - j - 1 if i % 2 else j
                    X2[k] = X[i, j]
            X = X2
        C = ''.join(self.indices_char[x] for x in X)
        return C


def get_parser():
    parser = argparse.ArgumentParser(prog='annotator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='increase output verbosity')
    parser.add_argument('-a', '--activation',
                        default=ACTIVATION,
                        help='keras activation function to use in inner layers: relu, tanh, sigmoid...')
    parser.add_argument('-e', '--epochs', type=int,
                        default=EPOCHS,
                        help='number of training epochs')
    parser.add_argument('-l', '--log', dest='logfile',
                        default=None,
                        help='log file')
    parser.add_argument('-z', '--batch_size', type=int,
                        default=BATCH_SIZE,
                        help='batch size')
    parser.add_argument('-d', '--dense_layers', nargs='+', type=int,
                        default=DENSE_LAYERS,
                        help='number of neurons in intermediate dense layers')
    parser.add_argument('--dropout', type=float,
                        default=DROPOUT,
                        help='dropout ratio')
    parser.add_argument('--layers', type=int,
                        default=LAYERS,
                        help='number of RNN layers to use')
    parser.add_argument('--lr', dest='learning_rate', type=float,
                        default=LEARNING_RATE,
                        help='learning rate')
    parser.add_argument('-m', '--model',
                        default=MODEL,
                        help='DNN model to use: res, res2d, ...')
    parser.add_argument('--mv', dest='model_variation',
                        default=MODEL_VARIATION,
                        help='Model variation')
    parser.add_argument('--optimizer',
                        default=OPTIMIZER,
                        help='keras optimizer to use: sgd, rmsprop, ...')
    parser.add_argument('--save',
                        default=SAVE,
                        help='prefix of output files')
    parser.add_argument('--tb', action='store_true',
                        help='use tensorboard')
    parser.add_argument('--maxlen', type=int,
                        default=MAXLEN,
                        help='DNA chunk length')
    parser.add_argument('--data',
                        default=DATA,
                        help='data')

    return parser


def load_data_100(maxlen=1000, val_split=0.2, batch_size=128, snake2d=False):
    ctable = CharacterTable(CHARS, maxlen)

    df = pd.read_csv('ref.100ec.pgf.seqs.filter', sep='\t', engine='c', dtype={'genome': str})

    n = df.shape[0]
    if snake2d:
        a = int(np.sqrt(maxlen))
        x = np.zeros((n, a, a, CHARLEN), dtype=np.byte)
    else:
        x = np.zeros((n, maxlen, CHARLEN), dtype=np.byte)

    for i, seq in enumerate(df['feature.na_sequence']):
        x[i] = ctable.encode(seq[:maxlen], snake2d=snake2d)

    y = pd.get_dummies(df.iloc[:, 0]).values
    classes = df.iloc[:, 0].nunique()

    # df_ref = pd.read_csv('ref.patric_ids', header=None, dtype=str)
    # mask = df['genome'].isin(df_ref[0].sample(100))
    # x_train = x[mask]
    # y_train = y[mask]
    # x_val = x[~mask]
    # y_val = y[~mask]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                      random_state=SEED,
                                                      stratify=df.iloc[:, 0])

    return (x_train, y_train), (x_val, y_val), classes


def load_data_1K(maxlen=1000, val_split=0.2, batch_size=128, snake2d=False):
    ctable = CharacterTable(CHARS, maxlen)

    df = pd.read_csv('rep.1000ec.pgf.seqs.filter', sep='\t', engine='c', dtype={'genome': str})
    # df = pd.read_csv('rep.1000ec.pgf.seqs.filter', nrows=10000, sep='\t', engine='c', dtype={'genome':str})

    n = df.shape[0]
    if snake2d:
        a = int(np.sqrt(maxlen))
        x = np.zeros((n, a, a, CHARLEN), dtype=np.byte)
    else:
        x = np.zeros((n, maxlen, CHARLEN), dtype=np.byte)

    for i, seq in enumerate(df['feature.na_sequence']):
        x[i] = ctable.encode(seq[:maxlen].lower(), snake2d=snake2d)

    y = pd.get_dummies(df.iloc[:, 0]).values
    classes = df.iloc[:, 0].nunique()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                      random_state=SEED,
                                                      stratify=df.iloc[:, 0])
    return (x_train, y_train), (x_val, y_val), classes


# def reshape_snake_2d(x1d, axis=1):
    # side = int(np.sqrt(x1d))


def load_data_coreseed(maxlen=1000, val_split=0.2, batch_size=128, snake2d=False):
    ctable = CharacterTable(CHARS, maxlen)

    df = pd.read_csv('coreseed.train.tsv', sep='\t', engine='c',
                     usecols=['function_index', 'dna'])

    n = df.shape[0]
    if snake2d:
        a = int(np.sqrt(maxlen))
        x = np.zeros((n, a, a, CHARLEN), dtype=np.byte)
    else:
        x = np.zeros((n, maxlen, CHARLEN), dtype=np.byte)

    for i, seq in enumerate(df['dna']):
        x[i] = ctable.encode(seq[:maxlen].lower(), snake2d=snake2d)

    y = pd.get_dummies(df.iloc[:, 0]).values
    classes = df.iloc[:, 0].nunique()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                      random_state=SEED,
                                                      stratify=df.iloc[:, 0])
    return (x_train, y_train), (x_val, y_val), classes


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.DATA={}'.format(args.data)
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.E={}'.format(args.epochs)
    ext += '.M={}'.format(args.model)
    if args.model.lower().startswith('res'):
        ext += args.model_variation
    ext += '.O={}'.format(args.optimizer)
    ext += '.LEN={}'.format(args.maxlen)
    ext += '.R={}'.format(args.learning_rate)
    if args.dropout != DROPOUT:
        ext += '.D={}'.format(args.dropout)
    for i, n in enumerate(args.dense_layers):
        if n > 0:
            ext += '.D{}={}'.format(i+1, n)

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


def simple_model(classes=100):
    model = Sequential(name='simple')
    model.add(Conv1D(200, 3, padding='valid', activation='relu', strides=1, input_shape=(MAXLEN, CHARLEN)))
    # model.add(Flatten())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    return model


def get_model(name, classes, args):
    name = name.lower()
    if name == 'vgg':
        from vgg16_nt import VGG16NT
        model = VGG16NT(input_shape=(args.maxlen, CHARLEN),
                        dense_size=args.dense_size,
                        dropout=args.dropout,
                        activation=args.activation,
                        classes=classes)
    elif name == 'res':
        from res50_nt import Res50NT
        model = Res50NT(input_shape=(args.maxlen, CHARLEN),
                        dense_layers=args.dense_layers,
                        dropout=args.dropout,
                        activation=args.activation,
                        variation=args.model_variation,
                        classes=classes)
    elif name == 'res2d':
        from res50_nt_2d import Res50NT2D
        a = int(np.sqrt(args.maxlen))
        model = Res50NT2D(input_shape=(a, a, CHARLEN),
                          dense_layers=args.dense_layers,
                          dropout=args.dropout,
                          activation=args.activation,
                          variation=args.model_variation,
                          classes=classes)
    elif name == 'inception':
        from inception_nt import InceptionV3NT
        model = InceptionV3NT(input_shape=(args.maxlen, CHARLEN),
                              classes=classes)
    else:
        model = simple_model(classes)

    return model


def set_up_logger(logfile, verbose):
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)


def main():
    parser = get_parser()
    args = parser.parse_args()

    ext = extension_from_parameters(args)
    prefix = args.save + '.' + args.model + ext
    logfile = args.logfile if args.logfile else prefix + '.log'
    set_up_logger(logfile, args.verbose)

    logger.info(args)

    args.dense_layers = [x for x in args.dense_layers if x > 0]
    snake2d = args.model.endswith('2d')

    if args.data == 'core':
        (x_train, y_train), (x_val, y_val), classes = load_data_coreseed(maxlen=args.maxlen, snake2d=snake2d)
    elif args.data == '100':
        (x_train, y_train), (x_val, y_val), classes = load_data_100(maxlen=args.maxlen, snake2d=snake2d)
    else:
        (x_train, y_train), (x_val, y_val), classes = load_data_1K(maxlen=args.maxlen, snake2d=snake2d)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    # np.savetxt('{}.y_val{}.txt'.format(args.save, ext), np.argmax(y_val, axis=1), fmt='%g')

    checkpointer = ModelCheckpoint(prefix + '.h5', save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(log_dir="tb/tb{}".format(ext))
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    file_log = LoggingCallback(logger.debug)
    callbacks = [checkpointer, reduce_lr, file_log]
    if args.tb:
        callbacks.append(tensorboard)

    top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = 'top5_acc'

    opt_config = {'lr': args.learning_rate} if args.learning_rate else {}
    optimizer = keras.optimizers.deserialize({'class_name': args.optimizer, 'config': opt_config})

    model = get_model(args.model, classes, args)
    print('Model:', model.name)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', top5_acc])

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_val, y_val),
              callbacks=callbacks)


if __name__ == '__main__':
    main()
