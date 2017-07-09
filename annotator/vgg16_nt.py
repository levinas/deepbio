from __future__ import print_function

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dropout
from keras.engine.topology import get_source_inputs
from keras import backend as K


def VGG16NT(include_top=True, weights='',
            input_tensor=None, input_shape=None,
            pooling=None, classes=1000, activation='relu',
            dense_size=4096, dropout=0):

    if input_tensor is None:
        seq_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            seq_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            seq_input = input_tensor

    # Block 1
    x = Conv1D(64, 3, activation=activation, padding='same', name='block1_conv1')(seq_input)
    x = Conv1D(64, 3, activation=activation, padding='same', name='block1_conv2')(x)
    x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

    # Block 2
    x = Conv1D(128, 3, activation=activation, padding='same', name='block2_conv1')(x)
    x = Conv1D(128, 3, activation=activation, padding='same', name='block2_conv2')(x)
    x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

    # Block 3
    x = Conv1D(256, 3, activation=activation, padding='same', name='block3_conv1')(x)
    x = Conv1D(256, 3, activation=activation, padding='same', name='block3_conv2')(x)
    x = Conv1D(256, 3, activation=activation, padding='same', name='block3_conv3')(x)
    x = MaxPooling1D(2, strides=2, name='block3_pool')(x)

    # Block 4
    x = Conv1D(512, 3, activation=activation, padding='same', name='block4_conv1')(x)
    x = Conv1D(512, 3, activation=activation, padding='same', name='block4_conv2')(x)
    x = Conv1D(512, 3, activation=activation, padding='same', name='block4_conv3')(x)
    x = MaxPooling1D(2, strides=2, name='block4_pool')(x)

    # Block 5
    x = Conv1D(512, 3, activation=activation, padding='same', name='block5_conv1')(x)
    x = Conv1D(512, 3, activation=activation, padding='same', name='block5_conv2')(x)
    x = Conv1D(512, 3, activation=activation, padding='same', name='block5_conv3')(x)
    x = MaxPooling1D(2, strides=2, name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(dense_size, activation=activation, name='fc1')(x)
        x = Dropout(dropout, name='drop1')(x)
        x = Dense(dense_size, activation=activation, name='fc2')(x)
        x = Dropout(dropout, name='drop2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = seq_input
    # Create model.
    model = Model(inputs, x, name='vgg16_nt')

    return model
