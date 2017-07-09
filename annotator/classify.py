#! /usr/bin/env python

from __future__ import print_function

import functools

import numpy as np
import pandas as pd
from keras import metrics
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


CHARS = ' atgc'
CHARLEN = len(CHARS)

SEED = 2017


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


def load_data_coreseed(maxlen=1000, val_split=0.2, batch_size=128):
    ctable = CharacterTable(CHARS, maxlen)

    df = pd.read_csv('coreseed.train.tsv', sep='\t', engine='c', usecols=['function_index', 'dna'])

    n = df.shape[0]
    x = np.zeros((n, maxlen, CHARLEN), dtype=np.byte)
    for i, seq in enumerate(df['dna']):
        x[i] = ctable.encode(seq[:maxlen].lower())

    y = pd.get_dummies(df.iloc[:, 0]).as_matrix()
    classes = df.iloc[:, 0].nunique()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                      random_state=SEED,
                                                      stratify=df.iloc[:, 0])
    return (x_train, y_train), (x_val, y_val), classes


def class_statistics(y_true, y_pred, class_names):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    sens = []
    spec = []
    most_conf_for = []
    most_conf_by = []

    for x in range(int(max(y_true))+1):
        row = cnf_matrix[x]
        col = cnf_matrix[:, x]
        tp = row[x]
        fp = sum(col)-tp
        fn = sum(row)-tp
        tn = len(y_true)-tp-fp
        sens.append(float(tp)/(tp+fn))
        spec.append(float(tn)/(tn+fp))
        sorted_row_idxs = np.argsort(row)
        sorted_col_idxs = np.argsort(col)
        mcf = sorted_row_idxs[-2] if sorted_row_idxs[-1] == x else sorted_row_idxs[-1]
        mcb = sorted_col_idxs[-2] if sorted_col_idxs[-1] == x else sorted_col_idxs[-1]
        most_conf_for.append(class_names[mcf])
        most_conf_by.append(class_names[mcb])

    stats_df = pd.DataFrame({'PGF': class_names, 'Sensitivity': sens, 'Specicifity': spec,
                             'Most FN': most_conf_for, 'Most FP': most_conf_by})

    return stats_df


def main():
    top5_acc = functools.partial(metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = 'top5_acc'

    model_file = 'save.res50_nt.DATA=core.A=relu.B=128.D=0.5.E=1000.O=sgd.DS=1024.LEN=1296.h5'
    model = load_model(model_file, custom_objects={'top5_acc': top5_acc})

    maxlen = 1296
    # (x_train, y_train), (x_val, y_val), classes = load_data_coreseed(maxlen)

    df_func = pd.read_csv('func.top1000', sep='\t', names=['count', 'function'], engine='c')
    df_func['function_index'] = range(1, len(df_func) + 1)
    func_dict = df_func.set_index('function_index')['function'].to_dict()

    df = pd.read_csv('coreseed.train.tsv', sep='\t', engine='c',
                     usecols=['peg', 'function_index', 'dna'])

    ctable = CharacterTable(CHARS, maxlen)
    n = df.shape[0]
    x = np.zeros((n, maxlen, CHARLEN), dtype=np.byte)
    for i, seq in enumerate(df['dna']):
        x[i] = ctable.encode(seq[:maxlen].lower())

    y_proba = model.predict(x, verbose=1)
    y_conf = y_proba.max(axis=-1)
    y_pred = y_proba.argmax(axis=-1) + 1
    y_true = df['function_index']

    df_pred = pd.DataFrame({'peg': df['peg'], 'score': y_conf,
                            'true_index': y_true,
                            'pred_index': y_pred})
    # df_pred = df_pred[df_pred['true_index'] != df_pred['pred_index']]
    df_pred = df_pred[['peg', 'score', 'true_index', 'pred_index']]
    df_pred['true_function'] = df_pred.apply(lambda row: func_dict[row['true_index']], axis=1)
    df_pred['pred_function'] = df_pred.apply(lambda row: func_dict[row['pred_index']], axis=1)

    df_pred.to_csv('train.pred.tsv', index=False, float_format='%.3f', sep='\t')

    df_stats = class_statistics(y_true, y_pred, func_dict)
    df_stats.to_csv('train.pred.stats.tsv', index=False, float_format='%.3f', sep='\t')

    # df = df_pred
    df = pd.read_csv('train.pred.tsv', sep='\t', engine='c')
    df = df[df['true_index'] != df['pred_index']]
    df['count'] = df.groupby(['true_index', 'pred_index'])['peg'].transform('count')
    df = df.sort_values(['count', 'true_index', 'score'], ascending=False)
    # df.to_csv('train.pred.error.tsv', index=False, float_format='%.3f', sep='\t')
    df[['peg', 'score', 'true_function', 'pred_function', 'count']].to_csv('train.pred.error.tsv', index=False, float_format='%.3f', sep='\t')


if __name__ == '__main__':
    main()
