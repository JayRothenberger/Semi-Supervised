"""
Network building code - each function returns a compiled keras model

Jay Rothenberger (jay.c.rothenberger@ou.edu)

"""

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D, \
    MultiHeadAttention
from time import time
from inception import *


def build_patchwise_vision_transformer(conv_filters,
                                       conv_size,
                                       attention_heads,
                                       image_size=(28, 28, 1),
                                       learning_rate=1e-3,
                                       n_classes=10,
                                       activation='selu',
                                       l1=None,
                                       l2=None,
                                       dropout=0):
    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': tf.keras.initializers.LecunNormal(),
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
    }
    # define the input layer (required)
    inputs = Input(image_size)
    x = inputs
    # set reference x separately to keep track of the input layer
    import math
    layers = max(int(math.log(image_size[0], 4)), int(math.log(image_size[1], 4)))
    lense_filters = 2*x.shape[-1]*layers
    for layer in range(layers + 2):
        # here we keep track of the input of each block
        x = Conv2D(filters=lense_filters, kernel_size=(4, 4), **conv_params, padding='same')(x)

    # construct the convolutional block
    for (filters, kernel) in zip(conv_filters, conv_size):
        # here we keep track of the input of each block
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), strides=(kernel, kernel), activation=activation,
                   **conv_params)(x)

    x = SpatialDropout2D(dropout)(x)
    print(attention_heads)
    for i, heads in enumerate(attention_heads):
        # for all layers except the last one, we return sequences
        key_dim = value_dim = x.shape[1]
        if i == len(attention_heads) - 1:
            # at the last layer of attention set the output to be a vector instead of a matrix
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(2, 3),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout, output_shape=(1,))(x, x)
        else:
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(2, 3),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(x, x)

    x = Flatten()(x)

    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'vat_model_{"%02d" % time()}')
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])

    return model


def build_sequential_model(conv_filters,
                           conv_size,
                           dense_layers,
                           image_size=(28, 28, 1),
                           learning_rate=1e-3,
                           n_classes=10,
                           activation='selu'):
    conv_params = {
        'activation': activation,
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': tf.keras.initializers.LecunNormal(),
        'padding': 'same',
    }

    # create the model object
    model = tf.keras.Sequential()
    # add an input layer (this step is only needed for the summary)
    model.add(Input(image_size))
    # add the convolutional layers
    for (filters, kernel) in zip(conv_filters, conv_size):
        model.add(Conv2D(filters=filters, kernel_size=(kernel, kernel), **conv_params))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # flatten
    model.add(Flatten())
    # add dense layers
    for neurons in dense_layers:
        model.add(Dense(neurons, activation=activation))
    # classification output
    model.add(Dense(n_classes, activation=tf.keras.activations.softmax))
    # optimizer
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])

    return model


def build_functional_model(conv_filters,
                           conv_size,
                           dense_layers,
                           image_size=(28, 28, 1),
                           learning_rate=1e-3,
                           n_classes=10,
                           activation='selu'):
    conv_params = {
        'activation': activation,
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': tf.keras.initializers.LecunNormal(),
        'padding': 'same',
    }

    # define the input layer (required)
    inputs = Input(image_size)
    # set reference x separately to keep track of the input layer
    x = inputs
    # construct the convolutional part
    for (filters, kernel) in zip(conv_filters, conv_size):
        # each layer is a function of the previous layer, we can reuse reference x
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), **conv_params)(x)
        # pooling after a convolution (or two) is a standard simple technique
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # flatten
    x = Flatten()(x)
    # construct the dense part
    for neurons in dense_layers:
        x = Dense(neurons, activation=activation)(x)
    # classification output
    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # optimizer
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    # when we compile the model we must specify inputs and outputs
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'cnn_model_{"%02d" % time()}')
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])

    return model


def build_parallel_functional_model(conv_filters,
                                    conv_size,
                                    dense_layers,
                                    image_size=(28, 28, 1),
                                    learning_rate=1e-3,
                                    n_classes=10,
                                    activation='selu',
                                    l1=None,
                                    l2=None,
                                    dropout=None):
    reg = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

    conv_params = {
        'activation': activation,
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': tf.keras.initializers.LecunNormal(),
        'padding': 'same',
        'kernel_regularizer': reg,
        'bias_regularizer': reg,
    }

    # define the input tensor
    inputs = Input(image_size)

    x = inputs
    # construct the convolutional block
    for (filters, kernel) in zip(conv_filters, conv_size):
        # here we keep track of the input of each block
        ins = x
        # there are two paths through which the data and gradient can flow
        # 1st path is x:
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), **conv_params)(ins)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        # 2nd path is y:
        y = Conv2D(filters=filters, kernel_size=(1, 1), **conv_params)(ins)
        y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y)
        # both paths' outputs are concatenated across the filter dimension
        x = Concatenate()([x, y])

        # Counter overfitting by randomly setting 2D feature maps to 0 (with probability rate) (change description later)
        x = SpatialDropout2D(dropout)(x)

        # and then an additional convolution that reduces the total filter dimension
        # is performed
        x = Conv2D(filters=filters, kernel_size=(1, 1), activation=activation)(x)

    # flatten
    x = Flatten()(x)
    # construct the dense part
    for neurons in dense_layers:
        x = Dense(neurons, activation=activation,
                  kernel_regularizer=reg, bias_regularizer=reg)(x)
    # classification output
    outputs = Dense(n_classes, activation=tf.keras.activations.softmax,
                    kernel_regularizer=reg, bias_regularizer=reg)(x)

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'cnn_model_{"%02d" % time()}')
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])

    return model


def build_inception_model(conv_filters,
                          conv_size,
                          dense_layers,
                          image_size=(28, 28, 1),
                          learning_rate=1e-3,
                          n_classes=10,
                          activation='selu',
                          l1=None,
                          l2=None,
                          dropout=0):
    reg = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

    conv_params = {
        'activation': activation,
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': tf.keras.initializers.LecunNormal(),
        'padding': 'same',
        'kernel_regularizer': reg,
        'bias_regularizer': reg
    }

    # define the input tensor
    inputs = Input(image_size)

    x = inputs
    # construct the convolutional block
    for (filters, kernel) in zip(conv_filters, conv_size):
        # here we keep track of the input of each block
        x = Inception(filters, activation=activation, kernel_regularizer=reg, bias_regularizer=reg, use_bias=True)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = SpatialDropout2D(dropout)(x)

    # flatten
    x = Flatten()(x)
    # construct the dense part
    for neurons in dense_layers:
        x = Dense(neurons, activation=activation)(x)
    # classification output
    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'cnn_model_{"%02d" % time()}')
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])

    # Generate an ASCII representation of the architecture
    print(model.summary())
    # generate a graphical plot

    return model
