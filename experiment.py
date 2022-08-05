"""
experiment code
Author: Jay C. Rothenberger (jay.c.rothenberger@ou.edu)
Contributor: Vincent A. Ferrera (vaflyers@gmail.com)
"""
import os
# model-related code I have written (supplied locally)
from cnn_network import *
from data_structures import ModelData
from main import generate_fname


def execute_exp(args, args_str, model, train_dset, val_dset, network_fn, network_params, train_iteration=0,
                train_steps=None, val_steps=None, callbacks=None,
                evaluate_on=None):
    """
    Perform the training and evaluation for a single model

    :param args: Argparse argument
    :param args_str:
    :param model: keras model
    :param train_dset:
    :param val_dset:
    :param train_iteration: training iteration
    :param train_steps:
    :param val_steps:
    :param augment_fn:
    :param image_size:
    :param callbacks:
    :param evaluate_on:
    :return: trained keras model encoded as a ModelData instance
    """

    print(args.exp)

    fbase = generate_fname(args)

    print(fbase)
    print(model.summary())
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True,
                              to_file=os.curdir + f'/../visualizations/models/model_{str(time())[:6]}.png')
    # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        return

    # Learn
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #  validation_steps=None
    #  means that ALL validation samples will be used (of the selected subset)
    history = model.fit(train_dset,
                        epochs=args.epochs,
                        verbose=True,
                        validation_data=val_dset,
                        callbacks=callbacks,
                        steps_per_epoch=train_steps,
                        validation_steps=val_steps,
                        shuffle=False)

    evaluate_on = dict() if evaluate_on is None else evaluate_on

    evaluations = {k: model.evaluate(evaluate_on[k]) for k in evaluate_on}

    # populate results data structure
    model_data = ModelData(weights=model.get_weights(),
                           network_params=network_params,
                           network_fn=network_fn,
                           evaluations=evaluations,
                           classes=network_params['n_classes'],
                           history=history.history,
                           train_fraction=args.train_fraction,
                           train_iteration=train_iteration,
                           args=args)
    return model_data
