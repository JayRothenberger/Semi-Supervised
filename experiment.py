"""
experiment code
Author: Jay C. Rothenberger (jay.c.rothenberger@ou.edu)
Contributor: Vincent A. Ferrera (vaflyers@gmail.com)
"""
# model-related code I have written (supplied locally)
from cnn_network import *
from data_structures import ModelData


def generate_fname(args, params_str):
    """
    Generate the base file name for output files/directories.

    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    :return: a string (file name prefix)
    """
    # network parameters
    hidden_str = '_'.join([str(i) for i in args.hidden])
    filters_str = '_'.join([str(i) for i in args.filters])
    kernels_str = '_'.join([str(i) for i in args.kernels])

    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_" % args.label

    # Experiment type
    if args.exp_type is None:
        experiment_type_str = ""
    else:
        experiment_type_str = "%s_" % args.exp_type

    # experiment index
    num_str = str(args.exp)

    # learning rate
    lrate_str = "LR_%0.6f_" % args.lrate

    return "%s/%s_%s_filt_%s_ker_%s_hidden_%s_l1_%s_l2_%s_drop_%s_frac_%s" % (
        args.results_path,
        experiment_type_str,
        num_str,
        filters_str,
        kernels_str,
        hidden_str,
        str(args.l1),
        str(args.l2),
        str(args.dropout),
        str(args.train_fraction))


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

    # use the available cpus to set the parallelism level
    if args.cpus_per_task is not None:
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)

    print(args.exp)

    fbase = generate_fname(args, args_str)

    print(fbase)
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
                        validation_steps=val_steps)

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
