import pickle

import numpy as np
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D


@dataclass
class ModelData:
    weights: np.ndarray
    network_params: dict
    network_fn: callable
    history: dict

    def get_history(self):
        # return the history object for the model
        return self.history

    def get_model(self):
        # return the keras model
        model = self.network_fn(**self.network_params)
        model.set_weights(self.weights)

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

    # Generate an ASCII representation of the architecture
    print(model.summary())
    # generate a graphical plot
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

    return model


if __name__ == '__main__':
    # network parameters dictionary
    network_params = {'learning_rate': args.lrate,
                      'conv_filters': args.filters,
                      'conv_size': args.kernels,
                      'dense_layers': args.hidden,
                      'image_size': (image_size[0], image_size[1], 3),
                      'n_classes': 3,
                      'l1': args.l1,
                      'l2': args.l2,
                      'dropout': args.dropout}

    # Build network: you must provide your own implementation (returns keras Model instance)
    network_fn = build_functional_model
    # use the network building function
    model = network_fn(**network_params)
    # get the history
    history = model.fit()

    # store in new results data structure
    model_data = ModelData(weights=model.get_weights(),
                           network_params=network_params,
                           network_fn=network_fn,
                           history=history.history)
    # pickle it
    with open('results/model.pkl', 'wb') as fp:
        pickle.dump(model_data, fp)
    # load it
    with open('results/model.pkl', 'rb') as fp:
        model_data = pickle.load(fp)
    # restore your keras model
    model = model_data.get_model()
