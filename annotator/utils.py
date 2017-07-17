import numpy as np
import pandas as pd


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


def load_data_100(maxlen=1000, val_split=0.2, batch_size=128, snake2d=False, seed=SEED):
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
                                                      random_state=seed,
                                                      stratify=df.iloc[:, 0])

    return (x_train, y_train), (x_val, y_val), classes


def load_data_1K(maxlen=1000, val_split=0.2, batch_size=128, snake2d=False, seed=SEED):
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
                                                      random_state=seed,
                                                      stratify=df.iloc[:, 0])
    return (x_train, y_train), (x_val, y_val), classes


# def reshape_snake_2d(x1d, axis=1):
    # side = int(np.sqrt(x1d))


def load_data_coreseed(maxlen=1000, val_split=0.2, batch_size=128, snake2d=False, seed=SEED):
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
                                                      random_state=seed,
                                                      stratify=df.iloc[:, 0])
    return (x_train, y_train), (x_val, y_val), classes
