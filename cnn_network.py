"""
Network building code - each function returns a compiled keras model

Jay Rothenberger (jay.c.rothenberger@ou.edu)

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D, \
    MultiHeadAttention, Add, BatchNormalization, LayerNormalization, Conv1D, Reshape, Cropping2D, ZeroPadding3D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Average, SeparableConv2D, DepthwiseConv2D
from time import time
import math
from inception import *
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16, MobileNetV3Small
import cai.layers
import cai.datasets
import cai.efficientnet
import cai.mobilenet_v3
import cai.densenet
import cai.util
import keras.backend as K

"""
Each function in this file returns a keras model and has to take arbitrary arguments (needs a **kwargs argument).
These functions must be PURELY FUNCTIONAL e.g. they cannot modify external data, and external data cannot effect
the model building process.  The one exception is using the timestamp to generate the name of the model.

A typical function will accept at least:
image_size, learning_rate, loss, n_classes

Where loss is the name of the loss function and n_classes is the number of output neurons

below is given a bare-bones template:

def build_net(dense_layers,
              learning_rate,
              image_size,
              loss='categorical_crossentropy',
              activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
              n_classes=10,
              **kwargs):
    # convert string to list of ints
    # NOTE: list arguments will be passed as their string representation for compatability with Hyperband
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in conv_filters.strip('[]').split(', ')]

    inputs = Input(image_size)

    x = inputs

    x = Flatten()(x)
    # construct the MLP
    for units in dense_layers:
        x = Dense(units, activation=activation)(x)
    # output classification
    outputs = Dense(n_classes, activation='softmax')(x)
    # metric to store every epoch
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # the gradient descent optimizer
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    # just leave this here
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    # construct the model instance
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'model_{"%02d" % time()}')
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])
    # return it
    return model
"""


def build_keras_application(application, image_size=(256, 256, 3), learning_rate=1e-4, loss='categorical_crossentropy',
                            n_classes=10, dropout=0, trainable_layers=4, **kwargs):
    inputs = Input(image_size)

    try:
        model = application(input_tensor=inputs, include_top=False, weights='imagenet', pooling='avg',
                            include_preprocessing=False)
    except TypeError as t:
        # no preprocessing for ResNet
        model = application(input_tensor=inputs, include_top=False, weights='imagenet', pooling='avg')

    outputs = Dense(n_classes, activation='softmax')(Dropout(dropout)(Flatten()(model.output)))

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[::-1]:
        if trainable_layers and len(layer.get_weights()) > 0:
            layer.trainable = True
            trainable_layers -= 1

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_EfficientNetB0(**kwargs):
    return build_keras_application(EfficientNetB0, **kwargs)


def build_ResNet50V2(**kwargs):
    return build_keras_application(ResNet50V2, **kwargs)


def build_MobileNetV3Small(**kwargs):
    return build_keras_application(MobileNetV3Small, **kwargs)


def build_keras_kapplication(application, image_size=(256, 256, 3), learning_rate=1e-4, loss='categorical_crossentropy',
                             n_classes=10, dropout=0, channels=16, skip_stride_cnt=0, **kwargs):
    """
    in order to get this to work I have made several changes to my training procedure

    1. learning rate is now cyclic
    2. optimizer is now RMSProp
    3. learning rate is now 1e-3
    4. batch size is now 64
    5. dropout is now .2
    6. now using the loss-scale optimizer
    7. now using a less stinky learning rate schedule
    8. less validation data
    9. skipping pooling steps for cifar

    """

    switch = {
        12: cai.layers.D6_12ch(),
        16: cai.layers.D6_16ch(),
        32: cai.layers.D6_32ch()
    }

    model = application(include_top=True,
                        alpha=1.0,
                        minimalistic=False,
                        input_shape=image_size,
                        classes=n_classes,
                        dropout_rate=dropout,
                        drop_connect_rate=dropout,
                        kType=switch.get(channels, cai.layers.D6_12ch()),
                        skip_stride_cnt=skip_stride_cnt)

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    opt = tf.keras.optimizers.RMSprop(learning_rate)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_kMobileNetV3(**kwargs):
    """
            'cifar10': {
            'params': {'learning_rate': args.lrate,
                       'conv_filters': args.filters,
                       'conv_size': args.kernels,
                       'attention_heads': args.hidden,
                       'image_size': (32, 32, 3),
                       'n_classes': 10,
                       'l1': args.l1,
                       'l2': args.l2,
                       'dropout': args.dropout,
                       'loss': 'categorical_crossentropy',
                       'pad': 4,
                       'overlap': 4,
                       'skip_stride_cnt': 3},
            'network_fn': build_kMobileNetV3},
    """
    return build_keras_kapplication(cai.mobilenet_v3.kMobileNetV3Large, channels=16, **kwargs)


def build_kDensenetBCL40(learning_rate=1e-4, loss='categorical_crossentropy', blocks=6, growth_rate=12, bottleneck=48,
                         compression=0.5, l2_decay=0.0001 / 1024, image_size=(32, 32, 3), **kwargs):
    model = cai.densenet.simple_densenet(image_size, blocks=blocks, growth_rate=growth_rate, bottleneck=bottleneck,
                                         compression=compression, l2_decay=l2_decay)

    opt = tf.keras.optimizers.RMSprop(learning_rate)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_hallucinetv4(conv_filters,
                       conv_size,
                       attention_heads,
                       learning_rate,
                       image_size,
                       iterations=24,
                       loss='categorical_crossentropy',
                       pooling='max',
                       l1=None, l2=None,
                       activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                       n_classes=10,
                       downsample=3,
                       dropout=0.1,
                       depth=2,
                       skips=6,
                       **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        dense_layers = [int(i) for i in attention_heads.strip('[]').split(', ')]

    inputs = Input(image_size)

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    expand_dims = Lambda(lambda z: tf.expand_dims(z, -1))
    squeeze = Lambda(lambda z: tf.squeeze(z, -1))

    x = inputs

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)
    add_to_end = []
    for block in range(len(conv_filters)):
        thrifty_exp = Conv2D(filters=conv_filters[block] * 4, kernel_size=1, activation=None, **conv_params)

        thrifty_convs = [DepthwiseConv2D(kernel_size=3,
                                         activation=activation, **conv_params),
                         DepthwiseConv2D(kernel_size=5,
                                         activation=activation, **conv_params),
                         MaxPooling2D(3, 1, padding='same')]

        thrifty_squeeze = Conv2D(filters=conv_filters[block], kernel_size=1, activation=activation, **conv_params)

        def thrifty_imb(z, exp, exc, sqz):
            inp = z
            z = exp(z)
            z = Concatenate()([e(z) for e in exc] + [inp])
            z = sqz(z)
            z = BatchNormalization()(z)
            return z
        prev_layers = [x]
        for i in range(iterations):
            x = thrifty_imb(x, thrifty_exp, thrifty_convs, thrifty_squeeze)
            prev = Dense(1, activation=None)(Concatenate(axis=-1)([expand_dims(l) for l in prev_layers[-skips:]]))
            prev = squeeze(prev)
            prev_layers.append(x)
            x = Add()([prev, x])
            x = BatchNormalization()(x)
            if downsample:
                if (i % math.ceil(iterations / downsample)) == 0:
                    add_to_end.append(Dense(n_classes, activation=activation)(GlobalMaxPooling2D()(x)))
                    add_to_end.append(Dense(n_classes, activation=activation)(GlobalAveragePooling2D()(x)))
                    x = MaxPooling2D((2, 2))(x)
                    # activation=None, **conv_params)
                    prev_layers = [MaxPooling2D((2, 2))(squeeze(
                        Dense(1, activation=None)(
                            Concatenate(axis=-1)([expand_dims(l) for l in prev_layers[-skips:]]))))]
                    # x = thrifty_attention(x, prev_layer)
                    x = BatchNormalization()(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalMaxPooling2D()(x)

    add_to_end.append(x)

    x = Concatenate()(add_to_end)

    x = Flatten()(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_hallucinetv3(conv_filters,
                       conv_size,
                       attention_heads,
                       learning_rate,
                       image_size,
                       iterations=24,
                       loss='categorical_crossentropy',
                       pooling='max',
                       l1=None, l2=None,
                       activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                       n_classes=10,
                       downsample=3,
                       dropout=0.1,
                       depth=2,
                       skips=6,
                       **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        dense_layers = [int(i) for i in attention_heads.strip('[]').split(', ')]

    inputs = Input(image_size)

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    x = inputs

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)
    add_to_end = []
    for block in range(len(conv_filters)):
        thrifty_exp = Conv2D(filters=conv_filters[block] * 4, kernel_size=1, activation=None, **conv_params)

        thrifty_convs = [DepthwiseConv2D(kernel_size=3,
                                         activation=activation, **conv_params),
                         DepthwiseConv2D(kernel_size=5,
                                         activation=activation, **conv_params),
                         MaxPooling2D(3, 1, padding='same')]

        thrifty_squeeze = Conv2D(filters=conv_filters[block], kernel_size=1, activation=activation, **conv_params)

        def thrifty_imb(z, exp, exc, sqz):
            inp = z
            z = exp(z)
            z = Concatenate()([e(z) for e in exc] + [inp])
            z = sqz(z)
            z = BatchNormalization()(z)
            return z
        prev_layers = [x]
        for i in range(iterations):
            x = thrifty_imb(x, thrifty_exp, thrifty_convs, thrifty_squeeze)
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
            if downsample:
                if (i % math.ceil(iterations / downsample)) == 0:
                    add_to_end.append(Dense(n_classes, activation=activation)(GlobalMaxPooling2D()(x)))
                    add_to_end.append(Dense(n_classes, activation=activation)(GlobalAveragePooling2D()(x)))
                    x = MaxPooling2D((2, 2))(x)
                    # activation=None, **conv_params)
                    prev_layer = MaxPooling2D((2, 2))(Add()(prev_layers[-skips:]))
                    prev_layers = [prev_layer]
                    # x = thrifty_attention(x, prev_layer)
                    x = BatchNormalization()(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalMaxPooling2D()(x)

    add_to_end.append(x)

    x = Concatenate()(add_to_end)

    x = Flatten()(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_hallucinetv2(conv_filters,
                       conv_size,
                       attention_heads,
                       learning_rate,
                       image_size,
                       iterations=24,
                       loss='categorical_crossentropy',
                       pooling='max',
                       l1=None, l2=None,
                       activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                       n_classes=10,
                       downsample=3,
                       dropout=0.1,
                       depth=2,
                       skips=6,
                       **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        dense_layers = [int(i) for i in attention_heads.strip('[]').split(', ')]

    inputs = Input(image_size)

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    x = inputs

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        thrifty_exp = Conv2D(filters=conv_filters[block] * 2, kernel_size=1, activation=None, **conv_params)

        thrifty_convs = [DepthwiseConv2D(kernel_size=min(x.shape[1], 3),
                                         activation=activation, **conv_params),
                         DepthwiseConv2D(kernel_size=min(x.shape[1], 5),
                                         activation=activation, **conv_params),
                         MaxPooling2D(3, 1, padding='same')]

        thrifty_squeeze = Conv2D(filters=conv_filters[block], kernel_size=1, activation=activation, **conv_params)

        def thrifty_imb(z, exp, exc, sqz):
            inp = z
            z = exp(z)
            z = Concatenate()([e(z) for e in exc] + [inp])
            z = sqz(z)
            z = BatchNormalization()(z)
            return z

        prev_layers = [x]
        for i in range(iterations):
            x = thrifty_imb(x, thrifty_exp, thrifty_convs, thrifty_squeeze)
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
            if downsample:
                if (i % math.ceil(iterations / downsample)) == 0:
                    x = MaxPooling2D((2, 2))(x)
                    # activation=None, **conv_params)
                    prev_layer = MaxPooling2D((2, 2))(Add()(prev_layers[-skips:]))
                    prev_layers = [prev_layer]
                    # x = thrifty_attention(x, prev_layer)
                    x = BatchNormalization()(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)

    for neurons in dense_layers:
        x = Dense(neurons, activation=activation)(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_hallucinet(conv_filters,
                     conv_size,
                     attention_heads,
                     learning_rate,
                     image_size,
                     iterations=24,
                     loss='categorical_crossentropy',
                     pooling='max',
                     l1=None, l2=None,
                     activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                     n_classes=10,
                     downsample=3,
                     dropout=0.1,
                     skips=6,
                     **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        dense_layers = [int(i) for i in attention_heads.strip('[]').split(', ')]

    inputs = Input(image_size)

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    x = inputs

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        # thrifty mb_conv
        thrifty_exp = Conv2D(filters=conv_filters[block] * 4, kernel_size=1, activation=None, **conv_params)

        thrifty_excite = DepthwiseConv2D(kernel_size=(conv_size[block], conv_size[block]),
                                         activation=activation, **conv_params)

        thrifty_squeeze = Conv2D(filters=conv_filters[block], kernel_size=1, activation=activation, **conv_params)

        def thrifty_mb(z, exp, exc, sqz):
            inp = z
            z = exp(z)
            z = exc(z)
            z = Concatenate()([z, inp])
            z = sqz(z)
            z = BatchNormalization()(z)
            return z

        x = thrifty_mb(x, thrifty_exp, thrifty_excite, thrifty_squeeze)
        prev_layers = [x]
        for i in range(iterations - 1):
            x = thrifty_mb(x, thrifty_exp, thrifty_excite, thrifty_squeeze)
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
            if downsample:
                if (i % math.ceil(iterations / downsample)) == 0:
                    thrifty_exp = Conv2D(filters=conv_filters[block] * 4, kernel_size=1, activation=None, **conv_params)

                    thrifty_excite = DepthwiseConv2D(kernel_size=(conv_size[block], conv_size[block]),
                                                     activation=activation, **conv_params)

                    thrifty_squeeze = Conv2D(filters=conv_filters[block], kernel_size=1, activation=activation,
                                             **conv_params)
                    x = MaxPooling2D((2, 2))(x)
                    prev_layer = MaxPooling2D((2, 2))(Add()(prev_layers[-skips:]))
                    prev_layers = [prev_layer]
                    # x = thrifty_attention(x, prev_layer)
                    x = BatchNormalization()(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)

    for neurons in dense_layers:
        x = Dense(neurons, activation=activation)(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_thriftynet_sep(conv_filters,
                         conv_size,
                         attention_heads,
                         learning_rate,
                         image_size,
                         iterations=40,
                         loss='categorical_crossentropy',
                         pooling='max',
                         l1=None, l2=None,
                         activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                         downsample=3,
                         n_classes=10,
                         dropout=0.1,
                         skips=5,
                         **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        dense_layers = [int(i) for i in attention_heads.strip('[]').split(', ')]

    inputs = Input(image_size)

    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    expand_dims = Lambda(lambda z: tf.expand_dims(z, -1))
    squeeze = Lambda(lambda z: tf.squeeze(z, -1))

    x = inputs

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        thrifty_conv = SeparableConv2D(filters=conv_filters[block], kernel_size=(conv_size[block], conv_size[block]),
                                       activation=activation, **conv_params)

        x = thrifty_conv(x)
        prev_layers = [x]
        for i in range(iterations - 1):
            x = thrifty_conv(x)
            prev_layers.append(x)
            x = Dense(1, activation=None)(Concatenate(axis=-1)([expand_dims(l) for l in prev_layers[-skips:]]))
            x = squeeze(x)
            x = BatchNormalization()(x)

            if (i % math.ceil(iterations / downsample)) == 0:
                x = MaxPooling2D((2, 2))(x)
                prev_layers = [MaxPooling2D((2, 2))(squeeze(
                    Dense(1, activation=None)(Concatenate(axis=-1)([expand_dims(l) for l in prev_layers[-skips:]]))))]
                x = BatchNormalization()(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)

    for neurons in dense_layers:
        x = Dense(neurons, activation=activation)(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_kthriftynetv3(conv_filters,
                        conv_size,
                        attention_heads,
                        learning_rate,
                        image_size,
                        iterations=16,
                        loss='categorical_crossentropy',
                        pooling='max',
                        l1=None, l2=None,
                        activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                        downsample=3,
                        n_classes=10,
                        dropout=0.1,
                        skips=5,
                        min_ch=32,
                        **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        dense_layers = [int(i) for i in attention_heads.strip('[]').split(', ')]

    inputs = Input(image_size)

    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    x = inputs

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        # min input channels = 12, 24,
        # gcd(pre_ch, ch) - they are, of course, the same so this is easy it is min ch
        def interleave_convs(convs, step=2):
            z = Concatenate()(convs)
            interleaves = [Lambda(lambda u: u[:, :, :, shift_pos::step])(z) for shift_pos in range(step)]
            z = Concatenate()(interleaves)
            return z

        def group_pointwise(z, Kconvs, Lconvs):
            n = len(Kconvs)
            k_filt = z.shape[-1] // n
            partitions = [Kconvs[i](Lambda(lambda conv: conv[:, :, :, i * k_filt:(i + 1) * k_filt])(z)) for i in
                          range(n)]
            z = Concatenate()(partitions)
            n = len(Lconvs)
            l_filt = z.shape[-1] // n
            offset = min_ch // 2
            partitions = [
                Lconvs[i](Lambda(lambda conv: conv[:, :, :, (i * l_filt) + offset:((i + 1) * l_filt) + offset])(z)) for
                i in
                range(n - 1)]
            end = Lconvs[-1](Concatenate()([Lambda(lambda conv: conv[:, :, :, 0:offset])(z),
                                            Lambda(lambda conv: conv[:, :, :, (n * l_filt) + offset:])(z)]))

            top = Lambda(lambda conv: conv[:, :, :, :offset])(end)
            bot = Lambda(lambda conv: conv[:, :, :, offset:])(end)

            z_1 = Concatenate()([top, *partitions, bot])
            return Add()([z, z_1])

        n = conv_filters[block] // min_ch

        thrifty_K_exp = [Conv2D(filters=min_ch * 2, kernel_size=1, activation=None, **conv_params)
                         for i in range(n)]

        thrifty_L_exp = [Conv2D(filters=min_ch * 2, kernel_size=1, activation=None, **conv_params)
                         for i in range(n)]

        thrifty_K_squeeze = [Conv2D(filters=min_ch, kernel_size=1, activation=activation, **conv_params)
                             for i in range(n)]

        thrifty_L_squeeze = [Conv2D(filters=min_ch, kernel_size=1, activation=activation, **conv_params)
                             for i in range(n)]

        thrifty_excite = DepthwiseConv2D(kernel_size=(conv_size[block], conv_size[block]),
                                         activation=activation, **conv_params)

        def thrifty_mb(z):
            z = group_pointwise(z, thrifty_K_exp, thrifty_L_exp)
            z = thrifty_excite(z)
            return group_pointwise(z, thrifty_K_squeeze, thrifty_L_squeeze)

        x = thrifty_mb(x)
        prev_layers = [x]
        for i in range(iterations - 1):
            x = thrifty_mb(x)
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)

            if (i % math.ceil(iterations / downsample)) == 0:
                x = MaxPooling2D((2, 2))(x)
                prev_layer = MaxPooling2D((2, 2))(Add()(prev_layers[-skips:]))
                prev_layers = [prev_layer]
                # x = thrifty_attention(x, prev_layer)
                x = BatchNormalization()(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)

    for neurons in dense_layers:
        x = Dense(neurons, activation=activation)(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_thriftynetv3(conv_filters,
                       conv_size,
                       attention_heads,
                       learning_rate,
                       image_size,
                       iterations=15,
                       loss='categorical_crossentropy',
                       pooling='max',
                       l1=None, l2=None,
                       activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                       downsample=0,
                       n_classes=10,
                       dropout=0.1,
                       skips=10,
                       **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        dense_layers = [int(i) for i in attention_heads.strip('[]').split(', ')]

    inputs = Input(image_size)

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    x = inputs

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        # thrifty mb_conv
        thrifty_exp = Conv2D(filters=conv_filters[block] * 4, kernel_size=1, activation=None, **conv_params)

        thrifty_excite = DepthwiseConv2D(kernel_size=(conv_size[block], conv_size[block]),
                                         activation=activation, **conv_params)

        thrifty_squeeze = Conv2D(filters=conv_filters[block], kernel_size=1, activation=activation, **conv_params)

        def thrifty_mb(z):
            z = thrifty_exp(z)
            z = thrifty_excite(z)
            return thrifty_squeeze(z)

        x = thrifty_mb(x)
        prev_layers = [x]
        for i in range(iterations - 1):
            x = thrifty_mb(x)
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
            if downsample:
                if (i % math.ceil(iterations / downsample)) == 0:
                    x = MaxPooling2D((2, 2))(x)
                    prev_layer = MaxPooling2D((2, 2))(Add()(prev_layers[-skips:]))
                    prev_layers = [prev_layer]
                    # x = thrifty_attention(x, prev_layer)
                    x = BatchNormalization()(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)

    for neurons in dense_layers:
        x = Dense(neurons, activation=activation)(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_thriftynet(conv_filters,
                     conv_size,
                     attention_heads,
                     learning_rate,
                     image_size,
                     iterations=40,
                     loss='categorical_crossentropy',
                     pooling='avg',
                     l1=None, l2=None,
                     activation='relu',
                     downsample=4,
                     n_classes=10,
                     skips=5,
                     **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        dense_layers = [int(i) for i in attention_heads.strip('[]').split(', ')]

    inputs = Input(image_size)

    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }
    expand_dims = Lambda(lambda z: tf.expand_dims(z, -1))
    squeeze = Lambda(lambda z: tf.squeeze(z, -1))

    x = inputs

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        thrifty_conv = Conv2D(filters=conv_filters[block], kernel_size=(conv_size[block], conv_size[block]),
                              activation=activation, **conv_params)

        x = thrifty_conv(x)
        prev_layers = [x]
        for i in range(iterations - 1):
            x = thrifty_conv(x)
            prev_layers.append(x)
            x = Dense(1, activation=None)(Concatenate(axis=-1)([expand_dims(l) for l in prev_layers[-skips:]]))
            x = squeeze(x)
            x = BatchNormalization()(x)

            if (i % math.ceil(iterations / downsample)) == 0:
                x = MaxPooling2D((2, 2))(x)
                prev_layers = [MaxPooling2D((2, 2))(squeeze(Dense(1, activation=None)(Concatenate(axis=-1)([expand_dims(l) for l in prev_layers[-skips:]]))))]

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)

    for neurons in dense_layers:
        x = Dense(neurons, activation=activation)(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_kMobileNet(**kwargs):
    return build_keras_kapplication(cai.mobilenet.kMobileNet, channels=32, **kwargs)


def build_kEfficientNetB0(**kwargs):
    return build_keras_kapplication(cai.efficientnet.kEfficientNetB0, **kwargs)


def build_transformer_4(conv_filters,
                        conv_size,
                        attention_heads,
                        image_size=(28, 28, 1),
                        learning_rate=1e-3,
                        n_classes=10,
                        activation='selu',
                        l1=None,
                        l2=None,
                        dropout=0,
                        loss='sparse_categorical_crossentropy',
                        pad=2, overlap=4, **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        attention_heads = [int(i) for i in attention_heads.strip('[]').split(', ')]

    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
    }
    # other knobs to tune : pad, overlap, squeeze_dim
    # define the input layer (required)
    inputs = Input(image_size)
    x = inputs
    # set reference x separately to keep track of the input layer
    # 1, create a list of lists of croppings

    crops = []
    row_breaks = range(0, x.shape[1] + (x.shape[1] % (conv_size[0] + pad)), conv_size[0] + pad)
    col_breaks = range(0, x.shape[2] + (x.shape[2] % (conv_size[0] + pad)), conv_size[0] + pad)

    for i, j in enumerate(row_breaks[:-1]):
        new_crops = [((max(j - overlap, 0), max((x.shape[1] - row_breaks[i + 1]) - overlap, 0)),
                      (max(l - overlap, 0), max((x.shape[2] - col_breaks[k + 1]) - overlap, 0)))
                     for k, l in enumerate(col_breaks[:-1])]
        for ci, ((top, bottom), (left, right)) in enumerate(new_crops):
            height = image_size[0] - (left + right)
            width = image_size[1] - (top + bottom)

            desired_height = conv_size[0] + pad + 2 * overlap
            desired_width = conv_size[0] + pad + 2 * overlap

            if width < desired_width:
                top -= (desired_width - width) if top > 0 else top
                bottom -= (desired_width - width) if bottom > 0 else bottom
            if height < desired_height:
                left -= (desired_height - height) if left > 0 else left
                right -= (desired_height - height) if right > 0 else right

            new_crops[ci] = ((top, bottom), (left, right))

        crops.append(new_crops)

    crop_layers = []

    for crop_list in crops:
        crop_layers.append([])
        for cropping in crop_list:
            # 2, perform convolutions on each cropping
            x_1 = Cropping2D(cropping)(x)
            x_1 = Conv2D(filters=conv_filters[0], kernel_size=(conv_size[0], conv_size[0]),
                         activation=activation, **conv_params)(x_1)
            for (filters, kernel) in zip(conv_filters[1:], conv_size[1:]):
                x_1 = SpatialDropout2D(dropout)(x_1)
                if kernel > 1:
                    x_1 = MaxPooling2D(strides=2, pool_size=2)(x_1)
                x_1 = Conv2D(filters=filters, kernel_size=(kernel, kernel),
                             activation=activation, **conv_params)(x_1)

            crop_layers[-1].append(x_1)
        # 3, concatenate along appropriate axes to create the desired volume
        crop_layers[-1] = Concatenate(axis=2)(crop_layers[-1])
    x = Concatenate(axis=1)(crop_layers)

    print(attention_heads)
    for i, heads in enumerate(attention_heads):
        skip = x
        # for all layers except the last one, we return sequences
        key_dim = value_dim = max(x.shape[1], x.shape[-1] // 2)
        if i == len(attention_heads) - 1:
            # at the last layer of attention set the output to be a vector instead of a matrix
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout, output_shape=(1,))(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=2, kernel_size=(1, 1), **conv_params, activation=activation)(x)
        else:
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=max(x.shape[1], x.shape[-1] // 2), kernel_size=(1, 1), **conv_params,
                       activation=activation)(x)

    x = Flatten()(x)

    for i in range(n_classes, x.shape[1], (x.shape[1] - n_classes) // 2)[::-1]:
        x = Dropout(dropout)(x)
        x = Dense(min(i, 96), activation='selu', **conv_params)(x)

    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'vit_model_{"%02d" % time()}')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_transformer_3(conv_filters,
                        conv_size,
                        attention_heads,
                        image_size=(28, 28, 1),
                        learning_rate=1e-3,
                        n_classes=10,
                        activation='selu',
                        l1=None,
                        l2=None,
                        dropout=0,
                        loss='sparse_categorical_crossentropy',
                        pad=2, overlap=4, **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        attention_heads = [int(i) for i in attention_heads.strip('[]').split(', ')]

    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
    }
    # other knobs to tune : pad, overlap, squeeze_dim
    # define the input layer (required)
    inputs = Input(image_size)
    x = inputs
    # set reference x separately to keep track of the input layer
    # 1, create a list of lists of croppings

    crops = []
    row_breaks = range(0, x.shape[1] + (x.shape[1] % (conv_size[0] + pad)), conv_size[0] + pad)
    col_breaks = range(0, x.shape[2] + (x.shape[2] % (conv_size[0] + pad)), conv_size[0] + pad)

    for i, j in enumerate(row_breaks[:-1]):
        new_crops = [((max(j - overlap, 0), max((x.shape[1] - row_breaks[i + 1]) - overlap, 0)),
                      (max(l - overlap, 0), max((x.shape[2] - col_breaks[k + 1]) - overlap, 0)))
                     for k, l in enumerate(col_breaks[:-1])]
        for ci, ((top, bottom), (left, right)) in enumerate(new_crops):
            height = image_size[0] - (left + right)
            width = image_size[1] - (top + bottom)

            desired_height = conv_size[0] + pad + 2 * overlap
            desired_width = conv_size[0] + pad + 2 * overlap

            if width < desired_width:
                top -= (desired_width - width) if top > 0 else top
                bottom -= (desired_width - width) if bottom > 0 else bottom
            if height < desired_height:
                left -= (desired_height - height) if left > 0 else left
                right -= (desired_height - height) if right > 0 else right

            new_crops[ci] = ((top, bottom), (left, right))

        crops.append(new_crops)

    crop_layers = []

    for crop_list in crops:
        crop_layers.append([])
        for cropping in crop_list:
            # 2, perform convolutions on each cropping
            x_1 = Cropping2D(cropping)(x)
            x_1 = Conv2D(filters=conv_filters[0], kernel_size=(conv_size[0], conv_size[0]),
                         activation=activation, **conv_params)(x_1)
            for (filters, kernel) in zip(conv_filters[1:], conv_size[1:]):
                x_1 = SpatialDropout2D(dropout)(x_1)
                x_1 = MaxPooling2D(strides=2, pool_size=2)(x_1)
                x_1 = Conv2D(filters=filters, kernel_size=(kernel, kernel),
                             activation=activation, **conv_params)(x_1)

            crop_layers[-1].append(x_1)
        # 3, concatenate along appropriate axes to create the desired volume
        crop_layers[-1] = Concatenate(axis=2)(crop_layers[-1])
    x = Concatenate(axis=1)(crop_layers)

    print(attention_heads)
    for i, heads in enumerate(attention_heads):
        skip = x
        # for all layers except the last one, we return sequences
        key_dim = value_dim = max(x.shape[1], x.shape[-1] // 2)
        if i == len(attention_heads) - 1:
            # at the last layer of attention set the output to be a vector instead of a matrix
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout, output_shape=(1,))(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=2, kernel_size=(1, 1), **conv_params, activation=activation)(x)
        else:
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=max(x.shape[1], x.shape[-1] // 2), kernel_size=(1, 1), **conv_params,
                       activation=activation)(x)
    # squeeze block
    squeeze_dim = 32
    y = x

    x = Conv2D(squeeze_dim, (x.shape[1], 1), activation=activation, **conv_params)(x)
    x = Reshape((squeeze_dim, x.shape[2], 1))(x)
    x = Conv2D(squeeze_dim, (1, x.shape[2]), activation=activation, **conv_params)(x)

    y = Conv2D(squeeze_dim, (1, y.shape[2]), activation=activation, **conv_params)(y)
    y = Reshape((y.shape[1], squeeze_dim, 1))(y)
    y = Conv2D(squeeze_dim, (y.shape[1], 1), activation=activation, **conv_params)(y)

    y = Reshape((squeeze_dim, squeeze_dim, 1))(y)
    x = Reshape((squeeze_dim, squeeze_dim, 1))(x)

    x = Concatenate()([x, y])
    x = Conv2D(1, (1, 1), activation=activation, **conv_params)(x)
    # end squeeze block
    x = Flatten()(x)

    for i in range(n_classes, x.shape[1], (x.shape[1] - n_classes) // 2)[::-1]:
        x = Dropout(dropout)(x)
        x = Dense(min(i, 96), activation='selu', **conv_params)(x)

    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'vit_model_{"%02d" % time()}')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_transformer_2(conv_filters,
                        conv_size,
                        attention_heads,
                        image_size=(28, 28, 1),
                        learning_rate=1e-3,
                        n_classes=10,
                        activation='selu',
                        l1=None,
                        l2=None,
                        dropout=0,
                        loss='sparse_categorical_crossentropy',
                        pad=2, overlap=4, **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        attention_heads = [int(i) for i in attention_heads.strip('[]').split(', ')]

    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
    }
    # other knobs to tune : pad, overlap, squeeze_dim
    # define the input layer (required)
    inputs = Input(image_size)
    x = inputs
    # set reference x separately to keep track of the input layer
    # 1, create a list of lists of croppings

    crops = []
    row_breaks = range(0, x.shape[1], conv_size[0] + pad)
    col_breaks = range(0, x.shape[2], conv_size[0] + pad)
    for i, j in enumerate(row_breaks[:-1]):
        crops.append([((max(j - overlap, 0), min((row_breaks[-1] - row_breaks[i + 1]) + overlap, x.shape[1])),
                       (max(l - overlap, 0), min((col_breaks[-1] - col_breaks[k + 1]) + overlap, x.shape[2])))
                      for k, l in enumerate(col_breaks[:-1])])

    crop_layers = []

    for crop_list in crops:
        crop_layers.append([])
        for cropping in crop_list:
            # 2, perform convolutions on each cropping
            x_1 = Cropping2D(cropping)(x)
            for (filters, kernel) in zip(conv_filters, conv_size):
                x_1 = Conv2D(filters=filters, kernel_size=(kernel, kernel),
                             activation=activation, **conv_params)(x_1)
            crop_layers[-1].append(x_1)
        # 3, concatenate along appropriate axes to create the desired volume
        crop_layers[-1] = Concatenate(axis=2)(crop_layers[-1])
    x = Concatenate(axis=1)(crop_layers)

    print(attention_heads)
    for i, heads in enumerate(attention_heads):
        skip = x
        # for all layers except the last one, we return sequences
        key_dim = value_dim = max(x.shape[1], x.shape[-1] // 2)
        if i == len(attention_heads) - 1:
            # at the last layer of attention set the output to be a vector instead of a matrix
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout, output_shape=(1,))(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=2, kernel_size=(1, 1), **conv_params, activation=activation)(x)
        else:
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=max(x.shape[1], x.shape[-1] // 2), kernel_size=(1, 1), **conv_params,
                       activation=activation)(x)
    # squeeze block
    squeeze_dim = 32
    y = x

    x = Conv2D(squeeze_dim, (x.shape[1], 1), activation=activation, **conv_params)(x)
    x = Reshape((squeeze_dim, x.shape[2], 1))(x)
    x = Conv2D(squeeze_dim, (1, x.shape[2]), activation=activation, **conv_params)(x)

    y = Conv2D(squeeze_dim, (1, y.shape[2]), activation=activation, **conv_params)(y)
    y = Reshape((y.shape[1], squeeze_dim, 1))(y)
    y = Conv2D(squeeze_dim, (y.shape[1], 1), activation=activation, **conv_params)(y)

    y = Reshape((squeeze_dim, squeeze_dim, 1))(y)
    x = Reshape((squeeze_dim, squeeze_dim, 1))(x)

    x = Concatenate()([x, y])
    x = Conv2D(1, (1, 1), activation=activation, **conv_params)(x)
    # end squeeze block
    x = Flatten()(x)

    for i in range(n_classes, x.shape[1], (x.shape[1] - n_classes) // 2)[::-1]:
        x = Dropout(dropout)(x)
        x = Dense(min(i, 96), activation='selu', **conv_params)(x)

    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'vit_model_{"%02d" % time()}')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_transformer_1(conv_filters,
                        conv_size,
                        attention_heads,
                        image_size=(28, 28, 1),
                        learning_rate=1e-3,
                        n_classes=10,
                        activation='selu',
                        l1=None,
                        l2=None,
                        dropout=0,
                        loss='sparse_categorical_crossentropy', **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        attention_heads = [int(i) for i in attention_heads.strip('[]').split(', ')]

    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
    }
    # other knobs to tune : pad, overlap, squeeze_dim
    # define the input layer (required)
    inputs = Input(image_size)
    x = inputs
    # set reference x separately to keep track of the input layer

    import math
    pad = math.ceil(max(x.shape[1], x.shape[2]) // 16)
    overlap = 4
    # 1, create a list of lists of croppings
    for (filters, kernel) in zip(conv_filters, conv_size):
        crops = []
        row_breaks = range(0, x.shape[1], kernel + pad)
        col_breaks = range(0, x.shape[2], kernel + pad)
        for i, j in enumerate(row_breaks[:-1]):
            crops.append([((max(j - overlap, 0), min((row_breaks[-1] - row_breaks[i + 1]) + overlap, x.shape[1])),
                           (max(l - overlap, 0), min((col_breaks[-1] - col_breaks[k + 1]) + overlap, x.shape[2])))
                          for k, l in enumerate(col_breaks[:-1])])

        crop_layers = []

        for crop_list in crops:
            crop_layers.append([])
            for cropping in crop_list:
                # 2, perform convolutions on each cropping
                x_1 = Cropping2D(cropping)(x)
                x_1 = Conv2D(filters=filters, kernel_size=(kernel, kernel),
                             activation=activation, **conv_params)(x_1)
                crop_layers[-1].append(x_1)
            # 3, concatenate along appropriate axes to create the desired volume
            crop_layers[-1] = Concatenate(axis=2)(crop_layers[-1])
        x = Concatenate(axis=1)(crop_layers)

    print(attention_heads)
    for i, heads in enumerate(attention_heads):
        skip = x
        # for all layers except the last one, we return sequences
        key_dim = value_dim = max(x.shape[1], x.shape[-1] // 2)
        if i == len(attention_heads) - 1:
            # at the last layer of attention set the output to be a vector instead of a matrix
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout, output_shape=(1,))(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=2, kernel_size=(1, 1), **conv_params, activation=activation)(x)
        else:
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=max(x.shape[1], x.shape[-1] // 2), kernel_size=(1, 1), **conv_params,
                       activation=activation)(x)
    # squeeze block
    squeeze_dim = 32
    y = x

    x = Conv2D(squeeze_dim, (x.shape[1], 1), activation=activation, **conv_params)(x)
    x = Reshape((squeeze_dim, x.shape[2], 1))(x)
    x = Conv2D(squeeze_dim, (1, x.shape[2]), activation=activation, **conv_params)(x)

    y = Conv2D(squeeze_dim, (1, y.shape[2]), activation=activation, **conv_params)(y)
    y = Reshape((y.shape[1], squeeze_dim, 1))(y)
    y = Conv2D(squeeze_dim, (y.shape[1], 1), activation=activation, **conv_params)(y)

    y = Reshape((squeeze_dim, squeeze_dim, 1))(y)
    x = Reshape((squeeze_dim, squeeze_dim, 1))(x)

    x = Concatenate()([x, y])
    x = Conv2D(1, (1, 1), activation=activation, **conv_params)(x)
    # end squeeze block
    x = Flatten()(x)

    for i in range(n_classes, x.shape[1], (x.shape[1] - n_classes) // 2)[::-1]:
        x = Dropout(dropout)(x)
        x = Dense(min(i, 96), activation='selu', **conv_params)(x)

    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'vit_model_{"%02d" % time()}')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_transformer_0(conv_filters,
                        conv_size,
                        attention_heads,
                        image_size=(28, 28, 1),
                        learning_rate=1e-3,
                        n_classes=10,
                        activation='selu',
                        l1=None,
                        l2=None,
                        dropout=0,
                        loss='sparse_categorical_crossentropy', **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        attention_heads = [int(i) for i in attention_heads.strip('[]').split(', ')]

    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
    }

    # define the input layer (required)
    inputs = Input(image_size)
    x = inputs
    # set reference x separately to keep track of the input layer
    import math
    layers = max(int(math.log(image_size[0], 3)), int(math.log(image_size[1], 3)))
    lense_filters = 4 * x.shape[-1] * layers
    for layer in range(layers):
        # here we keep track of the input of each block
        x = Conv2D(filters=lense_filters, kernel_size=(3, 3), **conv_params, padding='valid')(x)

    # construct the convolutional block
    for (filters, kernel) in zip(conv_filters, conv_size):
        # here we keep track of the input of each block
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), strides=(kernel, kernel), activation=activation,
                   **conv_params, padding='same')(x)

    print(attention_heads)
    for i, heads in enumerate(attention_heads):
        skip = x
        # for all layers except the last one, we return sequences
        key_dim = value_dim = max(x.shape[1], x.shape[-1] // 2)
        if i == len(attention_heads) - 1:
            # at the last layer of attention set the output to be a vector instead of a matrix
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout, output_shape=(1,))(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=2, kernel_size=(1, 1), **conv_params, activation=activation)(x)
        else:
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=max(x.shape[1], x.shape[-1] // 2), kernel_size=(1, 1), **conv_params,
                       activation=activation)(x)
    # squeeze block
    squeeze_dim = 32
    y = x

    x = Conv2D(squeeze_dim, (x.shape[1], 1), activation=activation, **conv_params)(x)
    x = Reshape((squeeze_dim, x.shape[2], 1))(x)
    x = Conv2D(squeeze_dim, (1, x.shape[2]), activation=activation, **conv_params)(x)

    y = Conv2D(squeeze_dim, (1, y.shape[2]), activation=activation, **conv_params)(y)
    y = Reshape((y.shape[1], squeeze_dim, 1))(y)
    y = Conv2D(squeeze_dim, (y.shape[1], 1), activation=activation, **conv_params)(y)

    y = Reshape((squeeze_dim, squeeze_dim, 1))(y)
    x = Reshape((squeeze_dim, squeeze_dim, 1))(x)

    x = Concatenate()([x, y])
    x = Conv2D(1, (1, 1), activation=activation, **conv_params)(x)
    # end squeeze block
    x = Flatten()(x)

    for i in range(n_classes, x.shape[1], (x.shape[1] - n_classes) // 2)[::-1]:
        x = Dropout(dropout)(x)
        x = Dense(min(i, 96), activation='selu', **conv_params)(x)

    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'vit_model_{"%02d" % time()}')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_axial_transformer(conv_filters,
                            conv_size,
                            attention_heads,
                            image_size=(28, 28, 1),
                            learning_rate=1e-3,
                            n_classes=10,
                            activation='selu',
                            l1=None,
                            l2=None,
                            dropout=0,
                            loss='sparse_categorical_crossentropy', **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        attention_heads = [int(i) for i in attention_heads.strip('[]').split(', ')]

    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
    }

    # define the input layer (required)
    inputs = Input(image_size)
    x = inputs
    # set reference x separately to keep track of the input layer
    import math
    layers = max(int(math.log(image_size[0], 4)), int(math.log(image_size[1], 4)))
    lense_filters = 4 * x.shape[-1] * layers
    for layer in range(layers):
        # here we keep track of the input of each block
        x = Conv2D(filters=lense_filters, kernel_size=(4, 4), **conv_params, padding='same')(x)

    # construct the convolutional block
    for (filters, kernel) in zip(conv_filters, conv_size):
        # here we keep track of the input of each block
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), strides=(kernel, kernel), activation=activation,
                   **conv_params, padding='same')(x)

    print(attention_heads)
    for i, heads in enumerate(attention_heads):
        skip = x
        # for all layers except the last one, we return sequences
        key_dim = value_dim = max(x.shape[1], x.shape[-1] // 2)
        if i == len(attention_heads) - 1:
            # at the last layer of attention set the output to be a vector instead of a matrix
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout, output_shape=(1,))(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=2, kernel_size=(1, 1), **conv_params, activation=activation)(x)
        else:
            x = LayerNormalization()(x)
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(x, x)
            x = Concatenate()([skip, x])
            x = Conv2D(filters=max(x.shape[1], x.shape[-1] // 2), kernel_size=(1, 1), **conv_params,
                       activation=activation)(x)

    x = Flatten()(x)

    for i in range(n_classes, x.shape[1], (x.shape[1] - n_classes) // 2)[::-1]:
        x = Dense(min(i, 96), activation='selu', **conv_params)(x)

    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'vit_model_{"%02d" % time()}')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_vision_transformer(conv_filters,
                             conv_size,
                             attention_heads,
                             image_size=(28, 28, 1),
                             learning_rate=1e-3,
                             n_classes=10,
                             activation='selu',
                             l1=None,
                             l2=None,
                             dropout=0,
                             loss='sparse_categorical_crossentropy', **kwargs):
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
    lense_filters = 3 * x.shape[-1] * layers
    for layer in range(layers + 3):
        # here we keep track of the input of each block
        x = Conv2D(filters=lense_filters, kernel_size=(4, 4), **conv_params, padding='same')(x)

    # construct the convolutional block
    for (filters, kernel) in zip(conv_filters, conv_size):
        # here we keep track of the input of each block
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), strides=(kernel, kernel), activation=activation,
                   **conv_params)(x)

    print(attention_heads)
    for i, heads in enumerate(attention_heads):
        # for all layers except the last one, we return sequences
        key_dim = value_dim = x.shape[1]
        if i == len(attention_heads) - 1:
            # at the last layer of attention set the output to be a vector instead of a matrix
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout, output_shape=(1,))(x, x)
        else:
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(x, x)

    x = Flatten()(x)

    for i in range(128, 256, 64)[::-1]:
        x = Dense(i, activation='selu')(x)

    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'vit_model_{"%02d" % time()}')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_patchwise_vision_transformer(conv_filters,
                                       conv_size,
                                       attention_heads,
                                       image_size=(28, 28, 1),
                                       learning_rate=1e-3,
                                       n_classes=10,
                                       activation='selu',
                                       l1=None,
                                       l2=None,
                                       dropout=0,
                                       loss='sparse_categorical_crossentropy', **kwargs):
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
    lense_filters = 3 * x.shape[-1] * layers
    for layer in range(layers + 3):
        # here we keep track of the input of each block
        x = Conv2D(filters=lense_filters, kernel_size=(4, 4), **conv_params, padding='same')(x)

    # construct the convolutional block
    for (filters, kernel) in zip(conv_filters, conv_size):
        # here we keep track of the input of each block
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), strides=(kernel, kernel), activation=activation,
                   **conv_params)(x)

    print(attention_heads)
    for i, heads in enumerate(attention_heads):
        # for all layers except the last one, we return sequences
        key_dim = value_dim = x.shape[1]
        if i == len(attention_heads) - 1:
            # at the last layer of attention set the output to be a vector instead of a matrix
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout, output_shape=(1,))(x, x)
        else:
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   attention_axes=(1, 2),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(x, x)

    x = Flatten()(x)

    for i in range(16, 64, 16)[::-1]:
        x = Dense(i, activation='selu')(x)

    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'vit_model_{"%02d" % time()}')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_sequential_model(conv_filters,
                           conv_size,
                           dense_layers,
                           image_size=(28, 28, 1),
                           learning_rate=1e-3,
                           n_classes=10,
                           activation='selu', **kwargs):
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
                           activation='selu', **kwargs):
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
                                    dropout=None, **kwargs):
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
                          dropout=0, **kwargs):
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
