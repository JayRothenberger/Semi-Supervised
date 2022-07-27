"""
experiment code
Author: Jay C. Rothenberger (jay.c.rothenberger@ou.edu)
Contributor: Vincent A. Ferrera (vaflyers@gmail.com)
"""
# standard python libraries
import os
import sys
from time import time
import PIL
from time import perf_counter as perf_time
import gc

# pip libraries
import argparse
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# code supplied locally
from job_control import *
from cnn_network import *
from transforms import rand_augment_object

# model-related code I have written (supplied locally)
from data_structures import ModelData, ModelEvaluator
from data_generator import prepare_dset, to_flow
from distances import *

CURRDIR = os.path.dirname(__file__)


def augment_with_neighbors(args, model, image_size, image_gen, train_process, distance_fn, labeled_data, unlabeled_data,
                           hard_labels=True, pseudolabeled_data=None, sample=None, pseudo_relabel=True):
    """
    augments the pseudolabeled_data dataframe with a batch of k nearest neighbors from the unlabeled_data dataframe

    dataframes to supply image filepaths expect a 'filepath' column

    :param args: CLA from the argument parser for the experiment
    :param model: model to generate pseudo-labels
    :param image_size: size to reshape image height and width dimensions
    :param image_gen: image generator used to compute withheld flow from dataframe
    :param train_process: image generator used to compute training flow
    :param distance_fn: function to compute distance for computing nearest neighbors
    :param labeled_data: training paths dataframe does not include the pseudolabeled points.
    :param unlabeled_data: unlabeled data paths dataframe
    :param hard_labels: if True returns the hard labels instead of the soft probability vector output
    :param pseudolabeled_data: df or None the points to compute the nearest neighbors of in the unlabeled set.
                         If None it takes the whole training set.
    :param sample: float or None fraction of available labeled data to use to calculate the nearest neighbors.
                   If None, uses all of the data.
    :param pseudo_relabel: the teacher assign new pseudo-labels to each unlabeled point at each iteration
    :return: train_gen, withheld_gen, train, withheld
    """
    unlabeled_data_gen = to_flow(unlabeled_data, shuffle=False, batch_size=args.batch, image_size=image_size,
                                 image_gen=image_gen)
    if args.augment_batch > len(unlabeled_data):
        raise ValueError('Not enough unlabeled data to augment with')

    if pseudolabeled_data is None:
        pseudolabeled_data = labeled_data

    if sample is None:
        sample = 1

    # compute the distance between every image pair
    distances = []
    # pre-loading examples from disk
    print('starting the timer')
    start = perf_time()

    # retrieve top-1 confidence for each prediction on the unlabeled data
    top_1 = np.max(model.predict(unlabeled_data_gen), axis=1)
    # this list holds (enumerate_index, (df_index, unlabeled_image_tensor)) elements (unlabeled images)
    ulab_img_array = [(i,
                       tf.keras.preprocessing.image.img_to_array(
                           tf.keras.preprocessing.image.load_img(unlabeled['filepath'], target_size=image_size)))
                      for i, (j, unlabeled) in enumerate(unlabeled_data.iterrows())]

    print(f'loaded unlabeled images ({image_size}): ', perf_time() - start)
    start = perf_time()

    print(f'calculated top-1 ({image_size}): ', perf_time() - start)
    start = perf_time()
    # this is the array over which the distance to the unlabeled points is computed (either labeled or pseudolabeled)
    # it has the same structure as the previous list, but this one has labeled or pseudolabeled images
    lab_img_array = [(i,
                      tf.keras.preprocessing.image.img_to_array(
                          tf.keras.preprocessing.image.load_img(labeled['filepath'], target_size=image_size)))
                     for i, (j, labeled) in enumerate(
            labeled_data.iloc[
                np.random.choice(range(len(labeled_data)),
                                 int(sample * len(labeled_data)), replace=False)].iterrows())] \
        if args.closest_to_labeled else [(i,
                                          tf.keras.preprocessing.image.img_to_array(
                                              tf.keras.preprocessing.image.load_img(labeled['filepath'],
                                                                                    target_size=image_size)))
                                         for i, (j, labeled) in enumerate(
            pseudolabeled_data.iloc[
                np.random.choice(range(len(labeled_data)),
                                 int(sample * len(labeled_data)), replace=False)].iterrows())]

    print(f'loaded labeled images ({image_size}): ', perf_time() - start)
    # now we have loaded two arrays of image tensors
    start = perf_time()
    count = 0
    # for each unlabeled image
    for i, img0 in ulab_img_array:
        count += 1
        print(f'{count} / {len(ulab_img_array)} ({perf_time() - start}s)                                              ',
              end='\r')
        # record the minimum distance between this unlabeled image and all labeled images in 'distances'
        distances.append(
            min([(i, distance_fn(img0, img1, top_1[i])) for j, img1 in lab_img_array], key=lambda x: x[-1]))

    print()
    print('finished computing distance:', perf_time() - start)
    start = perf_time()
    # sort the unlabeled points by their distance from any point we have the label for
    distances = sorted(distances, key=lambda x: x[-1])
    print(f'{len(distances)} distances')
    print('sorting took: ', perf_time() - start)
    # take the k images with the smallest distance values
    k_nearest_indices = [index for index, distance in distances[:args.augment_batch]]

    # get the image tensors corresponding to the appropriate indices from the previous line
    pseudo_labeled_batch = unlabeled_data.iloc[k_nearest_indices]
    # we will drop these indices from the unlabeled set
    unlabeled_to_drop = pseudo_labeled_batch.index

    def df_difference(df1, df2, column='filepath'):
        # computes the set difference col1 - col2
        d1 = {x: i for i, x in enumerate(df1[column])}
        d2 = {x: i for i, x in enumerate(df2[column])}

        return df1.iloc[[d1[i] for i in set(d1.keys()) - set(d2.keys())]]

    # if we are to re-label our pseudo-labeled data, we will need to find
    if pseudo_relabel:
        pseudo_labeled_batch = pd.concat([df_difference(pseudolabeled_data, labeled_data), pseudo_labeled_batch],
                                         ignore_index=True)

    pseudo_labeled_batch_gen = to_flow(pseudo_labeled_batch, shuffle=False, batch_size=args.batch,
                                       image_size=image_size,
                                       image_gen=image_gen)

    # set the classes as the pseudo-labels
    if hard_labels:
        pseudo_labels = np.argmax(model.predict(pseudo_labeled_batch_gen), axis=1)
    else:
        pseudo_labels = model.predict(pseudo_labeled_batch_gen)
    # set the assign the pseudo-labels
    pseudo_labeled_batch['class'] = pseudo_labels
    pseudo_labeled_batch['class'] = pseudo_labeled_batch['class'].astype(str)
    # remove those examples from the withheld set
    unlabeled_data = unlabeled_data.drop(index=unlabeled_to_drop)
    # add them to training set
    if pseudo_relabel:
        pseudolabeled_data = pd.concat((labeled_data, pseudo_labeled_batch), ignore_index=True)
    else:
        pseudolabeled_data = pd.concat((pseudolabeled_data, pseudo_labeled_batch), ignore_index=True)
    # df - > data flow
    train_gen = to_flow(pseudolabeled_data, shuffle=False, batch_size=args.batch, image_size=image_size,
                        image_gen=train_process)
    unlabeled_gen = to_flow(unlabeled_data, shuffle=False, batch_size=args.batch, image_size=image_size,
                            image_gen=image_gen)
    # return new flows and frames
    return train_gen, unlabeled_gen, pseudolabeled_data, unlabeled_data, labeled_data


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--cpus_per_task', type=int, default=None, help='number of cpus per task available to the job')

    # High-level experiment configuration
    parser.add_argument('--label', type=str, default="", help="Experiment label")
    parser.add_argument('--exp_type', type=str, default='test', help="Experiment type")
    parser.add_argument('--results_path', type=str, default=CURRDIR + '/../results', help='Results directory')

    # Semi-supervised parameters
    parser.add_argument('--train_fraction', type=float, default=0.05, help="fraction of available training data to use")
    parser.add_argument('--train_iterations', type=int, default=1,
                        help='number of iterations of training (after 1 is retraining)')
    parser.add_argument('--augment_batch', type=int, default=256,
                        help='number of examples added in each iteration of retraining')
    parser.add_argument('--retrain_fully', type=bool, default=False,
                        help='retrain the model from initialization entirely instead of tuning')
    parser.add_argument('--sample', type=float, default=None, help="fraction of available training data to use for knn "
                                                                   "calculation")
    parser.add_argument('--pseudo_relabel', type=bool, default=True, help='Teacher assigns a label to every unlabeled'
                                                                          'point at every iteration -- alternative is '
                                                                          'only the new batch of labels is determined '
                                                                          'by the teacher')
    parser.add_argument('--distance_function', type=str, default='euclidean', help='Determines k-nearest neighbors '
                                                                                   'for pseudo-label production')
    parser.add_argument('--closest_to_labeled', action='store_true',
                        help='compute nearest neighbors only to the labeled set')
    # Data augmentation parameters
    parser.add_argument('--rand_M', type=float, default=0, help='magnitude parameter for rand augment')
    parser.add_argument('--rand_N', type=int, default=0, help='iterations parameter for rand augment')

    # Specific experiment configuration
    parser.add_argument('--exp', type=int, default=0, help='Experiment index')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=1e-4, help="Learning rate")

    # convolutional unit parameters
    parser.add_argument('--filters', nargs='+', type=int, default=[],
                        help='Number of filters per layer (sequence of ints)')
    parser.add_argument('--kernels', nargs='+', type=int, default=[],
                        help='kernel sizes for each layer layer (sequence of ints)')

    # Hidden unit parameters
    parser.add_argument('--hidden', nargs='+', type=int, default=[],
                        help='Number of hidden units per layer (sequence of ints)')
    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=6, help="Patience for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=16, help="Training set batch size")
    parser.add_argument('--steps_per_epoch', type=int, default=None, help="Training steps per epoch")

    # Regularization parameters
    parser.add_argument('--l1', type=float, default=None, help="L1 regularization term weight")
    parser.add_argument('--l2', type=float, default=None, help="L2 regularization term weight")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate")

    return parser


def exp_type_to_hyperparameters(args):
    """
    convert the type of experiment to the appropriate hyperparameter grids

    :param args: the CLA for the experiment.

    @return a dictionary of hyperparameters
    """
    switch = {
        'full': {
            'filters': [[24, 32, 64, 128, 256, 256]],
            'kernels': [[3, 5, 3, 3, 3, 3]],
            'hidden': [[10, ]],
            'l1': [1e-4],
            'l2': [0],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [(i + 1) * .05 for i in range(20)],  # .05, .1, .15, ..., 1.0
            'epochs': [150],
            'augment_batch': [284],
            'sample': [None],
            'retrain_fully': [False],  # True, False
            'distance_function': ['euclidean'],  # 'euclidean', 'confidence'
            'rand_M': [0.1],
            'rand_N': [2],
            # pseudo_relabel: ['False']
        },
        'noreg': {
            'filters': [[24, 32, 64, 128, 256, 256]],
            'kernels': [[3, 5, 3, 3, 3, 3]],
            'hidden': [[10, ]],
            'l1': [None],
            'l2': [None],
            'dropout': [0],
            'train_iterations': [5],
            'train_fraction': [(i + 1) * .05 for i in range(20)]  # .05, .1, .15, ..., 1.0
        },
        'test': {
            'filters': [[24, 32, 64, 128, 256, 256]],
            'kernels': [[3, 5, 3, 3, 3, 3]],
            'hidden': [[10, ]],
            'l1': [None],
            'l2': [None],
            'dropout': [0],
            'train_iterations': [20],
            'train_fraction': [(i + 1) * .05 for i in range(20)],  # .05, .1, .15, ..., 1.0
            'epochs': [2],
            'augment_batch': [284],
            'sample': [.1],
            'distance_function': ['confidence'],
            'rand_M': [0.1],
            'rand_N': [2],
            'steps_per_epoch': [None],
            'patience': [2]
        },
        'no_self_train': {
            'filters': [[24, 32, 64, 128, 256, 256]],
            'kernels': [[3, 5, 3, 3, 3, 3]],
            'hidden': [[10, ]],
            'l1': [1e-4],
            'l2': [None],
            'dropout': [0.1],
            'train_iterations': [1],
            'train_fraction': [(i + 1) * .05 for i in range(20)],  # .05, .1, .15, ..., 1.0
        },
        'basic_SL': {
            'filters': [[24, 32, 64, 128, 256, 256]],
            'kernels': [[3, 5, 3, 3, 3, 3]],
            'hidden': [[10, ]],
            'l1': [0, 1e-4],
            'l2': [0, 1e-4],
            'dropout': [0, 0.05, 0.1, 0.15],
            'train_iterations': [1],
            'train_fraction': [1.0],
        }
    }

    rax = switch.get(args.exp_type)

    if rax is not None:
        return rax
    else:
        raise ValueError('unrecognized experiment type')


def augment_args(args):
    """
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    """

    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Check index number
    index = args.exp
    if index is None:
        return ""

    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())

    # Check bounds
    assert (0 <= args.exp < ji.get_njobs()), "exp out of range"

    # Print the parameters specific to this exp
    print(ji.get_index(args.exp))

    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp, args)


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


def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model

    :param args: Argparse arguments
    '''
    print(sys.platform)
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])

    # use the available cpus to set the parallelism level
    if args.cpus_per_task is not None:
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)

    print(args.exp)

    # Override arguments if we are using exp_index
    args_str = augment_args(args)
    image_size = (256, 256)
    # Split metadata into individual data sets
    # the relative data paths within the data directory
    data_paths = ['bronx_allsites/wet',
                  'bronx_allsites/dry',
                  'bronx_allsites/snow',
                  'ontario_allsites/wet',
                  'ontario_allsites/dry',
                  'ontario_allsites/snow',
                  'rochester_allsites/wet',
                  'rochester_allsites/dry',
                  'rochester_allsites/snow']
    # the path string to the data directory relative to this file
    data_paths = [CURRDIR + '/../data/' + f for f in data_paths]

    train, withheld, val, test, image_gen = prepare_dset(data_paths,
                                                         image_size=image_size,
                                                         batch_size=args.batch,
                                                         train_fraction=args.train_fraction)

    train_process = ImageDataGenerator(rescale=1. / 255,
                                       preprocessing_function=rand_augment_object(args.rand_M, args.rand_N, leq_M=True))
    # convert dataframes to flow
    train_gen, withheld_gen, val_gen, test_gen = to_flow(train,
                                                         train_process,
                                                         shuffle=True, image_size=image_size, batch_size=args.batch), \
                                                 to_flow(withheld,
                                                         image_gen,
                                                         shuffle=False, image_size=image_size, batch_size=args.batch), \
                                                 to_flow(val,
                                                         image_gen,
                                                         shuffle=False, image_size=image_size, batch_size=args.batch), \
                                                 to_flow(test,
                                                         image_gen,
                                                         shuffle=False, image_size=image_size, batch_size=args.batch)

    train_old = None
    model = None
    for train_iteration in range(args.train_iterations):
        # convert flows to datasets... (CANNOT have the same name as the flow)
        train_dset = tf.data.Dataset.from_generator(lambda: train_gen, output_types=(tf.float32, tf.int32),
                                                    output_shapes=([None, 256, 256, 3], [None, ])).prefetch(3)

        val_dset = tf.data.Dataset.from_generator(lambda: val_gen, output_types=(tf.float32, tf.int32),
                                                  output_shapes=([None, 256, 256, 3], [None, ])).prefetch(3)

        # arguments that are passed to each of the network returning functions
        network_params = {'learning_rate': args.lrate,
                          'conv_filters': args.filters,
                          'conv_size': args.kernels,
                          'dense_layers': args.hidden,
                          'image_size': (image_size[0], image_size[1], 3),
                          'n_classes': 3,
                          'l1': args.l1,
                          'l2': args.l2,
                          'dropout': args.dropout}

        # Build network: you must provide your own implementation
        network_fn = build_parallel_functional_model
        if args.retrain_fully or not train_iteration:
            model = network_fn(**network_params)

            # Output file base and pkl file
            fbase = generate_fname(args, args_str)
            fname_out = "%s_results.pkl" % fbase

        # Perform the experiment?
        if args.nogo:
            # No!
            print("NO GO")
            print(fbase)
            return

        # Callbacks
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=args.patience,
                                                             restore_best_weights=True,
                                                             min_delta=args.min_delta)

        # Learn
        #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
        #  validation_steps=None
        #  means that ALL validation samples will be used (of the selected subset)
        steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else 2 * len(train_gen)

        history = model.fit(train_dset,
                            epochs=args.epochs,
                            verbose=True,
                            validation_data=val_dset,
                            callbacks=[early_stopping_cb],
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=len(val) // args.batch)

        if args.train_fraction < 1:
            withheld_metrics = model.evaluate(withheld_gen, return_dict=True)
            withheld_predict = list(np.argmax(model.predict(withheld_gen, batch_size=args.batch), axis=1))
            withheld_true = withheld_gen.classes
        else:
            withheld_metrics = dict()
            withheld_predict = list()
            withheld_true = list()

        # if we will perform retraining (self-training)

        # new results data structure
        model_data = ModelData(weights=model.get_weights(),
                               network_params=network_params,
                               network_fn=network_fn,
                               val_metrics=model.evaluate(val_gen, return_dict=True),
                               train_metrics=model.evaluate(train_gen, return_dict=True),
                               test_metrics=model.evaluate(test_gen, return_dict=True),
                               withheld_metrics=withheld_metrics,
                               classes=list(np.unique(train_gen.classes)),
                               history=history.history,
                               withheld_true=withheld_true,
                               withheld_predict=withheld_predict,
                               val_predict=list(np.argmax(model.predict(val_gen, batch_size=args.batch), axis=1)),
                               val_true=val_gen.classes,
                               test_predict=list(np.argmax(model.predict(test_gen, batch_size=args.batch), axis=1)),
                               test_true=test_gen.classes,
                               train_fraction=args.train_fraction,
                               train_iteration=train_iteration,
                               args=args)

        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True,
                                  to_file=CURRDIR + '/../visualizations/model.png')

        with open('%s/model_%s_%s_%s' % (args.results_path,
                                         str(args.train_fraction),
                                         str(train_iteration),
                                         str(time())), 'wb') as fp:
            pickle.dump(model_data, fp)
        # de-reference model_data so it can be garbage collected

        dist_func_selector = {
            'euclidean': euclidean,
            'euclidean_with_confidence': euclidean_with_confidence,
            'confidence': confidence
        }

        distance_function = dist_func_selector.get(args.distance_function)

        if distance_function is None:
            raise ValueError('unrecognized experiment type')

        if withheld is not None and (len(withheld) - args.augment_batch) > 0:
            print(len(withheld))
            tf.keras.backend.clear_session()
            gc.collect()
            tf.keras.backend.clear_session()
            # replace the generators and dataframes with the updated ones from augment_with_neighbors
            train_gen, withheld_gen, train_old, withheld, train = augment_with_neighbors(args,
                                                                                         model,
                                                                                         image_size,
                                                                                         distance_fn=distance_function,
                                                                                         image_gen=image_gen,
                                                                                         train_process=train_process,
                                                                                         pseudolabeled_data=train_old,
                                                                                         unlabeled_data=withheld,
                                                                                         labeled_data=train,
                                                                                         sample=args.sample)
        else:
            print('not enough data for an additional iteration of training!')
            print(f'stopped after {train_iteration} iterations!')
            break
        print('retraining: ', train_iteration)


if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    # Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # just use one GPU

    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU')
    n_physical_devices = len(physical_devices)

    if n_physical_devices > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        print('We have %d GPUs\n' % n_physical_devices)
    else:
        print('NO GPU')

    execute_exp(args)
