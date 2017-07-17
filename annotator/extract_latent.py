from __future__ import print_function

import numpy as np
import pandas as pd

import keras

from res50_nt import Res50NT
from utils import CharacterTable
from utils import load_data_coreseed

CHARS = ' atgc'
CHARLEN = len(CHARS)
MAXLEN = 3600


base_model = Res50NT(input_shape=(MAXLEN, CHARLEN),
                     dense_layers=[1000],
                     dropout=0.1,
                     activation='selu',
                     variation='v1',
                     classes=1000)

weights_fname = 'save.res.DATA=core.A=selu.B=100.E=100.M=resv1.O=sgd.LEN=3600.R=0.01.D=0.1.D1=1000.h5'
# weights_url = 'file:///home/fangfang/big/deepbio/annotator/save/' + weights_fname
weights_url = 'http://bioseed.mcs.anl.gov/~fangfang/tmp/' + weights_fname
weights_path = keras.utils.data_utils.get_file(weights_fname, weights_url,
                                               cache_dir='.',
                                               cache_subdir='weights')

base_model.load_weights(weights_path)
model = keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# We can load core seed data with fixed train/validation partitions using the default random seed (2017)
#
# (x_train, y_train), (x_val, y_val), classes = load_data_coreseed(maxlen=args.maxlen, snake2d=snake2d)
#
# x_val_latent = base_model.predict(x)

# Or we can load the sequences with their functions
df_func = pd.read_csv('func.top1000', sep='\t', names=['count', 'function'], engine='c')
df_func['function_index'] = range(1, len(df_func) + 1)
func_dict = df_func.set_index('function_index')['function'].to_dict()

df = pd.read_csv('coreseed.train.tsv', sep='\t', engine='c', nrows=10,
                 usecols=['peg', 'function_index', 'dna'])

ctable = CharacterTable(CHARS, MAXLEN)
n = df.shape[0]
x = np.zeros((n, MAXLEN, CHARLEN), dtype=np.byte)
for i, seq in enumerate(df['dna']):
    x[i] = ctable.encode(seq[:MAXLEN].lower())

y_proba = base_model.predict(x, verbose=1)
y_conf = y_proba.max(axis=-1)
y_pred = y_proba.argmax(axis=-1) + 1
y_true = df['function_index']

x_latent = model.predict(x, verbose=1)

for i in range(n):
    print('X[{}]: {}'.format(i, x[i]))
    print('Latent features:', x_latent[i])
    print('y_pred: {}, {}', y_pred[i], func_dict[y_pred[i]])
    print('y_true: {}, {}', y_true[i], func_dict[y_true[i]])
    print('\n')
