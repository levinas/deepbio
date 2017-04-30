#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import os

import numpy as np

from Bio import SeqIO

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.layers import Input, Lambda, Layer, LSTM
from keras.models import Model
from keras import backend as K
from keras import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


CHARS = 'atgc '
CHARLEN = len(CHARS)
MAXLEN = 30
STEP = 100
DISPLAY_SAMPLES = 5
EPSILON_STD = 1.0

ACTIVATION = 'relu'
OPTIMIZER = 'adam'
EPOCHS = 1000
LAYERS = 1
BATCH_SIZE = 128
HIDDEN_DIM = 256
LATENT_DIM = 128


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


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


def load_data(maxlen=30, step=30, val_split=0.2, batch_size=128, flatten=False, head_only=False):
    ctable = CharacterTable(CHARS, maxlen)

    fname = '511145.12.PATRIC.ffn'
    fasta = SeqIO.parse(open(fname), 'fasta')

    if head_only:
        seqs = [str(s.seq)[:maxlen].lower() for s in fasta]
    else:
        seqs = [str(s.seq)[i:i+maxlen].lower() for s in fasta for i in range(0, len(str(s.seq))-maxlen+1, step)]

    np.random.shuffle(seqs)

    X = np.zeros((len(seqs), maxlen, CHARLEN), dtype=np.byte)
    for i, seq in enumerate(seqs):
        X[i] = ctable.encode(seq)

    train_size = int(len(seqs) * (1 - val_split) / batch_size) * batch_size
    val_size = int((len(seqs) - train_size) / batch_size) * batch_size
    x_train = X[:train_size]
    x_val = X[-val_size:]

    if flatten:
        flat_dim = maxlen * CHARLEN
        x_train = x_train.reshape(train_size, flat_dim)
        x_val = x_val.reshape(val_size, flat_dim)

    return x_train, x_val


def get_parser():
    parser = argparse.ArgumentParser(prog='p1b3_baseline',
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
    parser.add_argument("--step", type=int,
                        default=STEP,
                        help="step size for generating DNA chunks")
    parser.add_argument("--hidden_dim", type=int,
                        default=HIDDEN_DIM,
                        help="number of neurons in hidden layer")
    parser.add_argument("--latent_dim", type=int,
                        default=LATENT_DIM,
                        help="number of neurons in bottleneck layer")
    parser.add_argument("--eps", type=float,
                        default=EPSILON_STD,
                        help="epsilon std")
    parser.add_argument("--display_samples", type=int,
                        default=DISPLAY_SAMPLES,
                        help="number of validation samples to display")

    return parser


def predict_classes(model, x, batch_size=32, verbose=1):
    proba = model.predict(x, batch_size=batch_size, verbose=verbose)
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')


def train(model, x_train, x_val, args):
    ctable = CharacterTable(CHARS, args.maxlen)
    y_train, y_val = x_train, x_val
    for iteration in range(1, args.epochs):
        print('-' * 50)
        print('Iteration {}/{}'.format(iteration, args.epochs))
        model.fit(x_train, y_train, batch_size=args.batch_size, epochs=1,
                  validation_data=(x_val, y_val))
        for i in range(args.display_samples):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            preds = predict_classes(model, rowx, batch_size=1, verbose=0)
            # q = ctable.decode(rowx[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print('T', correct)
            print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)


def train_simple_ae(args):
    x_train, x_val = load_data(maxlen=args.maxlen, step=args.step)
    rnn = recurrent.LSTM

    model = Sequential()
    model.add(rnn(args.hidden_dim, input_shape=(args.maxlen, CHARLEN)))
    model.add(RepeatVector(args.maxlen))

    for _ in range(args.layers):
        model.add(rnn(args.latent_dim, return_sequences=True))

    model.add(TimeDistributed(Dense(CHARLEN)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=args.optimizer,
                  metrics=['accuracy'])

    train(model, x_train, x_val, args)


def train_vae(args):

    # x_train, x_val = load_data(maxlen=args.maxlen, step=args.step, batch_size=args.batch_size, flatten=True)
    x_train, x_val = load_data(maxlen=args.maxlen, step=args.step)

    input_dim = args.maxlen
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim

    x = Input(shape=(args.maxlen, CHARLEN))
    h = LSTM(hidden_dim)(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    encoder = Model(x, z_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = metrics.binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean))
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    def sampling(params):
        z_mean_, z_log_var_ = params
        batch_size = K.shape(z_mean_)[0]
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=EPSILON_STD)
        return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder_input = Input(shape=(latent_dim,))
    h = RepeatVector(args.maxlen)(decoder_input)
    h = LSTM(hidden_dim, return_sequences=True)(h)
    decoder_out = TimeDistributed(Dense(CHARLEN, activation='softmax'))(h)

    decoder = Model(decoder_input, decoder_out)

    x_decoded_mean = decoder(z)

    vae = Model(x, x_decoded_mean)

    encoder.summary()
    decoder.summary()
    vae.summary()

    vae.compile(optimizer=args.optimizer, loss=vae_loss, metrics=['accuracy'])
    # vae.compile(optimizer=args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    train(vae, x_train, x_val, args)

    # for iteration in range(1, args.epochs):
    #     print('-' * 50)
    #     print('Iteration {}/{}'.format(iteration, args.epochs))
    #     model.fit(x_train, x_train, batch_size=args.batch_size, epochs=1,
    #               validation_data=(x_val, x_val))
    #     for i in range(args.display_samples):
    #         ind = np.random.randint(0, len(x_val))
    #         rowx, rowy = x_val[np.array([ind])], x_val[np.array([ind])]
    #         preds = model.predict_classes(rowx, verbose=0)
    #         # q = ctable.decode(rowx[0])
    #         correct = ctable.decode(rowy[0])
    #         guess = ctable.decode(preds[0], calc_argmax=False)
    #         print('T', correct)
    #         print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)


def main():
    parser = get_parser()
    args = parser.parse_args()

    # train_simple_ae(args)
    train_vae(args)


if __name__ == '__main__':
    main()
