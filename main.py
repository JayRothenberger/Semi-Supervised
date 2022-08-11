"""
Experiment Management Code by Jay Rothenberger (jay.c.rothenberger@ou.edu)
"""

# code supplied by pip / conda
import os
import argparse

import matplotlib.pyplot as plt
import gc
from time import time
import numpy as np

# code supplied locally
import experiment
from job_control import JobIterator
from make_figure import *
from data_structures import *
from cnn_network import build_patchwise_vision_transformer
from transforms import custom_rand_augment_object
from data_generator import get_dataframes_self_train, to_dataset, get_cv_rotation, load_unlabeled
from experiment import start_training, self_train


def create_parser():
    """
    Create argument parser
    """
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--peek', action='store_true', help='Display images from dataset instead of training')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--cpus_per_task', type=int, default=None, help='number of cpus per task available to the job')
    parser.add_argument('--gpu_type', type=str, default="a100", help="a100 or v100 type of gpu training on")

    # High-level experiment configuration
    parser.add_argument('--label', type=str, default="", help="Experiment label")
    parser.add_argument('--exp_type', type=str, default='test', help="Experiment type")
    parser.add_argument('--results_path', type=str, default=os.curdir + '/../results', help='Results directory')
    parser.add_argument('--cv_rotation', type=int, default=0, help='positive int in [0, k],'
                                                                   ' rotation of cross-validation to execute')
    parser.add_argument('--cv_k', type=int, default=4, help='positive int - number of rotations of cross-validation')
    parser.add_argument('--cross_validate', action='store_true', help='Perform k-fold cross-validation')

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
    parser.add_argument('--randAugment', action='store_true', help='Use rand augment data augmentation')
    parser.add_argument('--convexAugment', action='store_true', help='Use convex data augmentation')

    parser.add_argument('--rand_M', type=float, default=0, help='magnitude parameter for rand augment')
    parser.add_argument('--rand_N', type=int, default=0, help='iterations parameter for rand augment')

    parser.add_argument('--convex_dim', type=int, default=1, help='number of examples to combine for the convex batch')
    parser.add_argument('--convex_prob', type=float, default=0, help='probability of encountering a convex augment'
                                                                     ' batch during a train step')

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
        'control': {
            'filters': [[36, 64]],
            'kernels': [[4, 1]],
            'hidden': [[3, 3]],
            'l1': [None],
            'l2': [None],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [1],
            'epochs': [512],
            'convex_dim': [1],
            'convex_prob': [0],
            'steps_per_epoch': [512],
            'patience': [32],
            'batch': [48],
            'lrate': [1e-4],
            'randAugment': [False],
            'peek': [True],
            'convexAugment': [False]
        },
        'test': {
            'filters': [[36, 64]],
            'kernels': [[4, 1]],
            'hidden': [[3, 3]],
            'l1': [None],
            'l2': [None],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [1],
            'epochs': [512],
            'convex_dim': [2, 3],
            'convex_prob': [1.0, .4],
            'steps_per_epoch': [512],
            'patience': [32],
            'batch': [48],
            'lrate': [1e-4],
            'randAugment': [False],
            'peek': [False],
            'convexAugment': [True],
            'cross_validate': [False],
            'rand_M': [.1],
            'rand_N': [2],
        },
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


def prep_gpu(gpu=False, style='a100'):
    """prepare the GPU for tensorflow computation"""
    # tell tensorflow to be quiet
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # use the available cpus to set the parallelism level
    if args.cpus_per_task is not None:
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)

    # Turn off GPU?
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif style == 'a100':
        pass
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # just use one GPU
    else:
        pass  # do nothing (v100)

    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU')
    n_physical_devices = len(physical_devices)

    if n_physical_devices > 0:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)
        print('We have %d GPUs\n' % n_physical_devices)
    else:
        print('NO GPU')


def blended_dset(train_df, batch_size=16, n_blended=2, image_size=(256, 256, 3), prefetch=4, prob=None, std=.1):
    """
    :param train_df: dataframe of training image paths
    :param batch_size: size of batches to return from the generator
    :param n_blended: number of examples to blend together
    :param image_size: shape of the input image tensor
    :param prefetch: number of examples to pre fetch from disk
    :param prob: probability of repacing a training batch with a convex combination of n_blended
    :param std: standard deviation of (mean 0) gaussian noise to add to images before blending
                (0.0 or equivalently None for no noise)
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    prob = prob if prob is not None else 1.0
    std = float(std) if std is not None else 0.0

    def add_gaussian_noise(x, y, std=1.0):
        return x + tf.random.normal(shape=x.shape, mean=0.0, stddev=std, dtype=tf.float32), y

    # create a dataset from which to get batches to blend
    dataset = to_dataset(train_df, shuffle=True, batch_size=batch_size, seed=42, prefetch=n_blended,
                         class_mode='categorical').map(
        lambda x, y: tf.py_function(add_gaussian_noise, inp=[x, y, std], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE).batch(n_blended)

    def random_weighting(n):
        # get a random weighting chosen uniformly from the convex hull of the unit vectors.
        samp = -1 * np.log(np.random.uniform(0, 1, n))
        samp /= np.sum(samp)
        return np.array(samp)

    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n-1 simplex
          (convex hull of unit vectors)
        """
        if np.random.uniform(0, 1, 1) > prob:
            return np.array([ex for ex in x][0]), np.array([ex for ex in y][0]).astype(np.float32)
        # compute the weights for the combination
        weights = random_weighting(n_blended)
        # yield the convex combination
        try:
            return tf.reduce_sum(np.array([weight * ex for ex, weight in zip(x, weights)]), axis=0), \
                   tf.reduce_sum(np.array([weight * tf.cast(ex, tf.float32) for ex, weight in zip(y, weights)]),
                                 axis=0)
        except ValueError as e:
            # sometimes (infrequently) our batches differ in size and cannot be stacked or meaned, we just need
            # to retry
            print(e)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE).prefetch(prefetch)

    return dataset


def mixup_dset(train_ds, prefetch=4, alpha=None):
    """
    :param train_ds: dataset of batches to train on
    :param prefetch: number of examples to pre fetch from disk
    :param alpha: Dirichlet parameter.  Weights are drawn from Dirichlet(alpha, ..., alpha) for combining two examples.
                    Empirically choose a value in [.1, .4]
    :return: a dataset
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    alpha = alpha if alpha is not None else 1.0

    rng = np.random.default_rng()

    # create a dataset from which to get batches to blend
    dataset = train_ds.batch(2)

    def random_weighting(n):
        return rng.dirichlet([alpha for i in range(n + 1)], 1)

    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n-1 simplex
          (convex hull of unit vectors)
        """
        # compute the weights for the combination
        weights = random_weighting(2)
        # return the convex combination
        return tf.reduce_sum(np.array([weight * ex for ex, weight in zip(x, weights)]), axis=0), \
               tf.reduce_sum(np.array([weight * tf.cast(ex, tf.float32) for ex, weight in zip(y, weights)]),
                             axis=0)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE).prefetch(prefetch)

    return dataset


def bc_plus(train_ds, prefetch=4):
    """
    :param train_ds: dataset of batches to train on
    :param prefetch: number of examples to pre fetch from disk
    :param alpha: Dirichlet parameter.  Weights are drawn from Dirichlet(alpha, ..., alpha) for combining two examples.
                    Empirically choose a value in [.1, .4]
    :return: a dataset
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    rng = np.random.default_rng()

    # create a dataset from which to get batches to blend
    dataset = train_ds.batch(2)

    def random_weighting(n):
        return rng.dirichlet([1 for i in range(n + 1)], 1)

    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
          (convex hull of unit vectors)
        """
        # compute the weights for the combination
        weights = random_weighting(2)
        weights *= float(1 / np.linalg.norm(weights))
        weights = np.array(weights, dtype=np.double).reshape(-1, 1)
        x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
        y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))[0]
        x = x - tf.reduce_mean(x)
        # return the convex combination
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE).prefetch(prefetch)

    return dataset


def my_mixup_dset(train_ds, n_blended=2, prefetch=4, alpha=.25):
    """
    :param train_ds: dataset of batches to train on
    :param n_blended: number of examples to mix
    :param prefetch: number of examples to pre fetch from disk
    :param alpha: Dirichlet parameter.  Weights are drawn from Dirichlet(alpha, ..., alpha) for combining two examples.
                    Empirically choose a value in [.1, .4]
    :return: a dataset
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    alpha = alpha if alpha is not None else 1.0

    rng = np.random.default_rng()

    def add_gaussian_noise(x, y, std=0.01):
        return x + tf.random.normal(shape=x.shape, mean=0.0, stddev=std, dtype=tf.float32), y

    # create a dataset from which to get batches to blend
    dataset = train_ds.map(
        lambda x, y: tf.py_function(add_gaussian_noise, inp=[x, y, .01], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE).batch(n_blended)

    def random_weighting(n):
        return rng.dirichlet(tuple([alpha for i in range(n)]), 1)

    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
          (convex hull of unit vectors)
        """
        # compute the weights for the combination
        weights = random_weighting(n_blended)
        weights *= float(1 / np.linalg.norm(weights))
        weights = np.array(weights, dtype=np.double).reshape(-1, 1)
        x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
        y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))[0]
        x = x - tf.reduce_mean(x)
        # return the convex combination
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE).prefetch(prefetch)

    return dataset


def fix_image_df(df):
    """
    Take in a dataframe of absolute image paths.  PIL opens and then saves them again, this can fix image corruption
    errors in tensorflow
    """
    from PIL import Image

    def verify_jpeg_image(file_path):
        try:
            img = Image.open(file_path)
            img0 = img.getdata()[0]
            img.save(file_path)
            return bool(img0) or True
        except OSError:
            return False

    bads = 0
    goods = 0

    for index, row in df():
        img = row['filepath']
        if verify_jpeg_image(img):
            goods += 1
        else:
            bads += 1
    # print the number of images that couldn't be saved, and the number fixed
    print('irreparable:', bads, 'fixed', goods)


def fix_image_dir(directory):
    """
    Take in the path of a directory.  PIL opens images in subdir structure and then saves them again, this can fix image
    corruption errors in tensorflow.
    """
    from PIL import Image

    file_list = []

    for path, directories, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(path, file))

    def verify_jpeg_image(file_path):
        try:
            img = Image.open(file_path)
            img0 = img.getdata()[0]
            img.save(file_path)
            return bool(img0) and True
        except OSError:
            return False

    bads = 0
    goods = 0

    for path in file_list:
        if verify_jpeg_image(path):
            goods += 1
        else:
            bads += 1
            print(path)

    # print the number of images that couldn't be saved, and the number fixed
    print('irreparable:', bads, 'fixed', goods)


def generate_fname(args):
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

    return "%s/%s_%s_filt_%s_ker_%s_hidden_%s_l1_%s_l2_%s_drop_%s_frac_%s_lrate_%s_%s" % (
        args.results_path,
        experiment_type_str,
        num_str,
        filters_str,
        kernels_str,
        hidden_str,
        str(args.l1),
        str(args.l2),
        str(args.dropout),
        str(args.train_fraction),
        lrate_str,
        str(time()).replace('.', '')[-6:])


if __name__ == '__main__':
    from data_generator import get_image_dsets

    image_size = (256, 256)
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    args, args_str = augment_args(args)

    prep_gpu(args.gpu, style=args.gpu_type)

    paths = ['bronx_allsites/wet', 'bronx_allsites/dry', 'bronx_allsites/snow',
             'ontario_allsites/wet', 'ontario_allsites/dry', 'ontario_allsites/snow',
             'rochester_allsites/wet', 'rochester_allsites/dry', 'rochester_allsites/snow']

    unlabeled_paths = [os.curdir + '/../unlabeled/']

    # unlabeled_df = load_unlabeled(unlabeled_paths)

    print('getting dsets...')
    if args.cross_validate:
        print(f'({args.cv_k}-fold) cross-validation rotation: {args.cv_rotation}')
        train_df, withheld_df, val_df, test_df = get_cv_rotation(
            [os.curdir + '/../data/' + path for path in paths],
            train_fraction=args.train_fraction, k=args.cv_k, rotation=args.cv_rotation)
    else:
        train_df, withheld_df, val_df, test_df, ig = get_dataframes_self_train(
            [os.curdir + '/../data/' + path for path in paths],
            train_fraction=args.train_fraction)

    network_params = {'learning_rate': args.lrate,
                      'conv_filters': args.filters,
                      'conv_size': args.kernels,
                      'attention_heads': args.hidden,
                      'image_size': (image_size[0], image_size[1], 3),
                      'n_classes': 3,
                      'l1': args.l1,
                      'l2': args.l2,
                      'dropout': args.dropout,
                      'loss': 'categorical_crossentropy'}

    print('hidden', args.hidden)
    network_fn = build_patchwise_vision_transformer
    """
    self_train(args, network_fn, network_params, train_df, val_df, withheld_df,
               augment_fn=custom_rand_augment_object(.1, 2, leq_M=False),
               evaluate_on=None)
    """
    class_mode = network_params['loss'].split('_')[0]
    val_dset = to_dataset(val_df, shuffle=True, batch_size=args.batch, class_mode=class_mode)
    train_dset = to_dataset(train_df, shuffle=True, batch_size=args.batch, seed=42, prefetch=8,
                            class_mode='categorical', center=True)
    # define our randAugment object
    rand_aug = custom_rand_augment_object(args.rand_M, args.rand_N, True)
    # data augmentation strategy selection
    if args.convexAugment and args.randAugment:
        train_dset = blended_dset(train_df, args.batch, args.convex_dim, prob=args.convex_prob, std=.05) \
            .map(lambda x, y: (tf.py_function(rand_aug, [x], [tf.float32])[0], y),
                 num_parallel_calls=tf.data.AUTOTUNE, )
    elif args.convexAugment:
        train_dset = my_mixup_dset(train_dset, args.convex_dim, 8, args.convex_prob)
    elif args.randAugment:
        train_dset = train_dset.map(lambda x, y: (tf.py_function(rand_aug, [x], [tf.float32])[0], y),
                                    num_parallel_calls=tf.data.AUTOTUNE, )

    # peek at the dataset instead of training
    if args.peek:
        explore_image_dataset(train_dset, 8)
        exit(-1)
    # create the scope
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    # build the model (in the scope)
    model = network_fn(**network_params)
    # train the model
    model_data = start_training(args, model, train_dset, val_dset, network_fn, network_params,
                                train_steps=args.steps_per_epoch, val_steps=len(val_df) // args.batch)
    # save the model
    try:
        with open(f'{os.curdir}/../results/{generate_fname(args)}', 'wb') as fp:
            pickle.dump(model_data, fp)
    except Exception as e:
        print(e)
        with open(f'./{generate_fname(args)}', 'wb') as fp:
            pickle.dump(model_data, fp)
