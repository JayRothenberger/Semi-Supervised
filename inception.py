"""
Inception Class

Jay Rothenberger (jay.c.rothenberger@ou.edu)

"""

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D
from time import time


class Inception(tf.keras.layers.Layer):
    def __init__(self, filters=32, kernels=((3, 3), (5, 5)), stride=(1, 1), activation=None, use_bias=False,
                 kernel_initializer=tf.keras.initializers.LecunNormal(), bias_initializer='zeros', pool_size=(3, 3),
                 kernel_regularizer=None, bias_regularizer=None, padding='same'):
        """
        Returns an inception block as a keras Model

        :param filters: the number of filters to use in each convolutional block
        :param stride: the stride of the block at the end of each pathway in the inception module
        :param kernels: an iterator of tuples of ints, the kernel sizes to use to build the pathways
        :param activation: the activation function to apply in all convolutional blocks
        :param usebias: if True convolutional layers use bias terms
        :param init: initializer for bias an kernels within the layer
        :param pool_size: size of pool to use in the maxpooling layer
        :param index: a value that is used to name the layer.  Name is f'inception_block_{index}'
        :param reg: kernel regularizer object
        :@ return: a keras Model
        """
        super(Inception, self).__init__()
        # inialize the list with the components that will always be in the inception layers (maxpool, 1x1)

        conv_params = {
            'use_bias': use_bias,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'bias_initializer': bias_initializer,
            'kernel_initializer': kernel_initializer,
            'padding': padding,
            'activation': activation
        }

        self.components = [Conv2D(filters, (1, 1), stride, **conv_params),
                           MaxPooling2D(pool_size, stride, padding='same')] + \
                          [Conv2D(filters, kernel, stride, **conv_params) for kernel in kernels]

        self.downscaling = [Conv2D(filters, (1, 1), (1, 1), **conv_params) for x in self.components[1:]]

    def call(self, inputs):
        return Concatenate()([self.components[0](inputs)] + [component(downscale(inputs)) for component, downscale in
                                                             zip(self.components[1:], self.downscaling)])
