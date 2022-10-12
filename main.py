"""
Experiment Management Code by Jay Rothenberger (jay.c.rothenberger@ou.edu)
"""

# code supplied by pip / conda
import argparse
from tensorflow import keras
import keras_tuner as kt
from copy import deepcopy as copy
import time
import datetime

TIMESTRING = datetime.datetime.fromtimestamp(time.time()).isoformat(sep='T', timespec='auto').replace(':', '')
# code supplied locally
from job_control import JobIterator
from cnn_network import *
from data_structures import *
from experiment import cifar10, cifar100, DOT_CV, DOT_CV_self_train, get_dsets, cd, dw, cl


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
    parser.add_argument('--scratch', type=str, default=None, help="Local scratch partition")

    # CPU/GPU
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--cpus_per_task', type=int, default=None, help='number of cpus per task available to the job')
    parser.add_argument('--gpu_type', type=str, default="None", help="a100 or v100 type of gpu training on")
    parser.add_argument('--distributed', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--prefetch', type=int, default=12, help='number of dataset batches to prefetch')

    # High-level experiment configuration
    parser.add_argument('--label', type=str, default="", help="Experiment label")
    parser.add_argument('--exp_type', type=str, default='test', help="Experiment type")
    parser.add_argument('--results_path', type=str, default=os.curdir + '/../results', help='Results directory')
    parser.add_argument('--cv_rotation', type=int, default=0, help='positive int in [0, k],'
                                                                   ' rotation of cross-validation to execute')
    parser.add_argument('--cv_k', type=int, default=4, help='positive int - number of rotations of cross-validation')
    parser.add_argument('--cross_validate', action='store_true', help='Perform k-fold cross-validation')
    parser.add_argument('--cache', action='store_true', help='cache the dataset in memory')
    parser.add_argument('--hyperband', action='store_true', help='Perform hyperband hyperparameter search')

    # Semi-supervised parameters
    parser.add_argument('--train_fraction', type=float, default=0.05, help="fraction of available training data to use")
    parser.add_argument('--train_iterations', type=int, default=1,
                        help='number of iterations of training (after 1 is retraining)')
    parser.add_argument('--augment_batch', type=int, default=256,
                        help='number of examples added in each iteration of retraining')
    parser.add_argument('--retrain_fully', type=bool, default=False,
                        help='retrain the model from initialization entirely instead of tuning')
    parser.add_argument('--sample', type=float, default=None, help="fraction of available unlabeled data to use for "
                                                                   "knn "
                                                                   "calculation")
    parser.add_argument('--pseudo_relabel', type=bool, default=True, help='Teacher assigns a label to every unlabeled'
                                                                          'point at every iteration -- alternative is '
                                                                          'only the new batch of labels is determined '
                                                                          'by the teacher')
    parser.add_argument('--distance_function', type=str, default='confidence', help='Determines k-nearest neighbors '
                                                                                    'for pseudo-label production')
    parser.add_argument('--closest_to_labeled', action='store_true',
                        help='compute nearest neighbors only to the labeled set')
    # Data augmentation parameters
    parser.add_argument('--randAugment', action='store_true', help='Use rand augment data augmentation')
    parser.add_argument('--convexAugment', type=str, default=None, help='type of MSDA to use.')

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
            'convex_dim': [2],
            'convex_prob': [1.0],
            'steps_per_epoch': [512],
            'patience': [32],
            'batch': [48],
            'lrate': [1e-4],
            'randAugment': [False],
            'peek': [False],
            'convexAugment': [None],
            'cross_validate': [False],
            'rand_M': [.1],
            'rand_N': [2],
            'network_fn': [build_transformer_4]
        },
        'self_train': {
            'filters': [[36, 64]],
            'kernels': [[4, 1]],
            'hidden': [[3, 3]],
            'l1': [None],
            'l2': [None],
            'dropout': [0.1],
            'train_iterations': [2],
            'train_fraction': [1],
            'epochs': [512],
            'convex_dim': [2],
            'convex_prob': [.5],
            'steps_per_epoch': [512],
            'patience': [32],
            'batch': [48],
            'augment_batch': [65536],
            'lrate': [1e-4],
            'randAugment': [True],
            'peek': [False],
            'convexAugment': ['mixup'],
            'cross_validate': [False],
            'rand_M': [.1],
            'rand_N': [2],
            'sample': [1.0],
            'network_fn': [build_transformer_4]
        },
        'cifar100': {
            'filters': [[36, 64]],
            'kernels': [[2, 1]],
            'hidden': [[3, 3]],
            'l1': [None],
            'l2': [None],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [1],
            'epochs': [512],
            'convex_dim': [2],
            'convex_prob': [.25],
            'steps_per_epoch': [1024],
            'patience': [32],
            'batch': [128],
            'lrate': [1e-3],
            'randAugment': [False],
            'peek': [False],
            'convexAugment': [None],
            'cross_validate': [False],
            'rand_M': [.1],
            'rand_N': [2],
            'network_fn': [build_kDensenetBCL40, build_kMobileNetV3, build_kEfficientNetB0],
            'iterations': [24],
            'downsample': [4],
        },
        'cifar10': {
            'filters': ['[56]'],
            'kernels': ['[3]'],
            'hidden': ['[10]'],
            'l1': [None],
            'l2': [None],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [1],
            'epochs': [512],
            'convex_dim': [2],
            'convex_prob': [.5],
            'steps_per_epoch': [1024],
            'patience': [32],
            'batch': [64],
            'lrate': [1e-3],
            'randAugment': [True],
            'peek': [False],
            'convexAugment': ['mixup'],  # ],  # 'mixup', 'blended', 'fff', 'fmix'],  #'foff'
            'cross_validate': [False],
            'rand_M': [.1],
            'rand_N': [1],
            # 'blocks': [i for i in range(1, 9)],
            # 'iterations': list(range(12, 40, 2)),
            # 'activation': ['relu', 'selu', 'elu'],
            # 'downsample': list(range(1, 5)),
            # 'search_space': {
            #    'downsample': True,
            #    'activation': False,
            #    'iterations': True,
            # },
            'min_delta': [.001],
            'network_fn': [build_hallucinetv4, build_hallucinetv3],
            'iterations': [20],
            'downsample': [5],
            # , build_kDensenetBCL40, build_kMobileNetV3, build_kEfficientNetB0, build_thriftynet_sep, build_thriftynet
        },
        'da': {
            'filters': ['[32]'],
            'kernels': ['[3]'],
            'hidden': ['[10]'],
            'l1': [None],
            'l2': [1e-5],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [1],
            'epochs': [512],
            'convex_dim': [2],
            'convex_prob': [.5],
            'steps_per_epoch': [512],
            'patience': [12],
            'batch': [3],
            'lrate': [1e-3],
            'randAugment': [True],
            'peek': [False],
            'convexAugment': [None],
            'cross_validate': [False],
            'rand_M': [.3],
            'rand_N': [1],
            # 'search_space': {
            #    'dropout': True,
            #    'l1': True,
            #    'l2': True,
            #    'hidden': False,
            # },
            'min_delta': [.001],
            'network_fn': [build_hallucinetv4_upcycle_plus_plus],
            'iterations': [8],
            'downsample': [7],
        },
        'cd': {
            'filters': ['[48]'],
            'kernels': ['[3]'],
            'hidden': ['[10]'],
            'l1': [None],
            'l2': [None],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [1],
            'epochs': [512],
            'convex_dim': [2],
            'convex_prob': [.5],
            'steps_per_epoch': [512],
            'patience': [32],
            'batch': [8],
            'lrate': [5e-4],
            'randAugment': [True],
            'peek': [False],
            'convexAugment': [None],
            'cross_validate': [False],
            'rand_M': [.3],
            'rand_N': [1],
            # 'search_space': {
            #    'dropout': True,
            #    'l1': True,
            #    'l2': True,
            #    'hidden': False,
            # },
            'min_delta': [.001],
            'network_fn': [build_hallucinetv4_upcycle_plus_plus, build_hallucinet_upcycle],
            'iterations': [8],
            'downsample': [7],
        },
        'dw': {
            'filters': ['[32]'],
            'kernels': ['[3]'],
            'hidden': ['[10]'],
            'l1': [None],
            'l2': [None],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [1],
            'epochs': [512],
            'convex_dim': [2],
            'convex_prob': [.5],
            'steps_per_epoch': [512],
            'patience': [12],
            'batch': [5],
            'lrate': [1e-3],
            'randAugment': [True],
            'peek': [False],
            'convexAugment': [None],
            'cross_validate': [False],
            'rand_M': [.3],
            'rand_N': [1],
            # 'search_space': {
            #    'dropout': True,
            #    'l1': True,
            #    'l2': True,
            #    'hidden': False,
            # },
            'min_delta': [.001],
            'network_fn': [build_hallucinetv4_upcycle_plus_plus, build_hallucinet_upcycle],
            'iterations': [12],
            'downsample': [7],
        },
        'cl': {
            'filters': ['[32]'],
            'kernels': ['[3]'],
            'hidden': ['[10]'],
            'l1': [None],
            'l2': [None],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [1],
            'epochs': [512],
            'convex_dim': [2],
            'convex_prob': [.5],
            'steps_per_epoch': [512],
            'patience': [4],
            'batch': [8],
            'lrate': [1e-3],
            'randAugment': [True],
            'peek': [False],
            'convexAugment': [None],
            'cross_validate': [False],
            'rand_M': [.3],
            'rand_N': [1],
            # 'search_space': {
            #    'dropout': True,
            #    'l1': True,
            #    'l2': True,
            #    'hidden': False,
            # },
            'min_delta': [.001],
            'network_fn': [build_hallucinetv4_upcycle_plus_plus, build_hallucinetv4_upcycle_plus],
            'iterations': [3],
            'downsample': [7],
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
    ji = JobIterator({key: p[key] for key in set(p) - set(p['search_space']) - {'search_space'}}) \
        if args.hyperband else JobIterator({key: p[key] for key in set(p) - {'search_space'}})

    print("Total jobs:", ji.get_njobs())

    # Check bounds
    assert (0 <= args.exp < ji.get_njobs()), "exp out of range"

    # Print the parameters specific to this exp
    print(ji.get_index(args.exp))

    # Push the attributes to the args object and return a string that describes these structures
    augmented, arg_str = ji.set_attributes_by_index(args.exp, args)
    if args.hyperband:
        vars(augmented).update({key: p[key] for key in p['search_space']})
        vars(augmented).update({'search_space': p['search_space']})

    return augmented, arg_str


def prep_gpu(index, gpu=False, style='a100'):
    """prepare the GPU for tensorflow computation"""
    # tell tensorflow to be quiet
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Turn off GPU?
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif style == 'a100':
        # GPU check
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[index % len(tf.config.list_physical_devices('GPU'))], 'GPU')
    else:
        pass  # do nothing (v100, distributed)
    physical_devices = tf.config.get_visible_devices('GPU')
    n_physical_devices = len(physical_devices)
    print(physical_devices)

    # use the available cpus to set the parallelism level
    if args.cpus_per_task is not None:
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)

    if n_physical_devices > 1:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)
        print('We have %d GPUs\n' % n_physical_devices)
    elif n_physical_devices:
        print('We have %d GPUs\n' % n_physical_devices)
    else:
        print('NO GPU')


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
    hidden_str = '_'.join([str(i) for i in args.hidden]) if not isinstance(args.hidden, str) else args.hidden
    filters_str = '_'.join([str(i) for i in args.filters]) if not isinstance(args.filters, str) else args.filters
    kernels_str = '_'.join([str(i) for i in args.kernels]) if not isinstance(args.kernels, str) else args.kernels
    try:
        # Experiment type
        if args.exp_type is None:
            experiment_type_str = ""
        else:
            experiment_type_str = "%s_" % args.exp_type
    except:
        experiment_type_str = "%s_" % str(args.exp_type)

    # experiment index
    num_str = str(args.exp)

    # learning rate
    lrate_str = f"LR_{str(args.lrate)[:4]}_"

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


def network_switch(args, key, default):
    switch = {
        'self_train': {
            'params': {'learning_rate': args.lrate,
                       'conv_filters': args.filters,
                       'conv_size': args.kernels,
                       'attention_heads': args.hidden,
                       'image_size': (256, 256, 3),
                       'n_classes': 3,
                       'l1': args.l1,
                       'l2': args.l2,
                       'dropout': args.dropout,
                       'loss': 'categorical_crossentropy'},
            'network_fn': args.network_fn},

        'da': {
            'params':
                {'learning_rate': args.lrate,
                 'conv_filters': args.filters,
                 'conv_size': args.kernels,
                 'attention_heads': args.hidden,
                 'image_size': (256, 256, 3),
                 'n_classes': 3,
                 'l1': args.l1,
                 'l2': args.l2,
                 'dropout': args.dropout,
                 'loss': 'categorical_crossentropy',
                 'iterations': args.iterations,
                 'downsample': args.downsample,
                 'pad': 24,
                 'overlap': 8},
            'network_fn': args.network_fn},

        'cd': {
            'params':
                {'learning_rate': args.lrate,
                 'conv_filters': args.filters,
                 'conv_size': args.kernels,
                 'attention_heads': args.hidden,
                 'image_size': (128, 128, 3),
                 'n_classes': 2,
                 'l1': args.l1,
                 'l2': args.l2,
                 'dropout': args.dropout,
                 'loss': 'categorical_crossentropy',
                 'iterations': args.iterations,
                 'downsample': args.downsample,
                 'pad': 24,
                 'overlap': 8},
            'network_fn': args.network_fn},

        'dw': {
            'params':
                {'learning_rate': args.lrate,
                 'conv_filters': args.filters,
                 'conv_size': args.kernels,
                 'attention_heads': args.hidden,
                 'image_size': (128, 128, 3),
                 'n_classes': 9,
                 'l1': args.l1,
                 'l2': args.l2,
                 'dropout': args.dropout,
                 'loss': 'categorical_crossentropy',
                 'iterations': args.iterations,
                 'downsample': args.downsample,
                 'pad': 24,
                 'overlap': 8},

            'network_fn': args.network_fn},

        'cl': {
            'params':
                {'learning_rate': args.lrate,
                 'conv_filters': args.filters,
                 'conv_size': args.kernels,
                 'attention_heads': args.hidden,
                 'image_size': (128, 128, 3),
                 'n_classes': 4,
                 'l1': args.l1,
                 'l2': args.l2,
                 'dropout': args.dropout,
                 'loss': 'categorical_crossentropy',
                 'iterations': args.iterations,
                 'downsample': args.downsample,
                 'pad': 24,
                 'overlap': 8},

            'network_fn': args.network_fn},

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
                       'pad': 1,
                       'overlap': 4,
                       'iterations': args.iterations,
                       'downsample': args.downsample,
                       'skip_stride_cnt': 3},
            'network_fn': args.network_fn},

        'cifar100': {
            'params': {'learning_rate': args.lrate,
                       'conv_filters': args.filters,
                       'conv_size': args.kernels,
                       'attention_heads': args.hidden,
                       'image_size': (32, 32, 3),
                       'n_classes': 100,
                       'l1': args.l1,
                       'l2': args.l2,
                       'dropout': args.dropout,
                       'loss': 'categorical_crossentropy',
                       'iterations': args.iterations,
                       'downsample': args.downsample,
                       'pad': 4,
                       'overlap': 4,
                       'skip_stride_cnt': 3},
            'network_fn': args.network_fn},

        'control': {
            'params':
                {'learning_rate': args.lrate,
                 'conv_filters': args.filters,
                 'conv_size': args.kernels,
                 'attention_heads': args.hidden,
                 'image_size': (256, 256, 3),
                 'n_classes': 3,
                 'l1': args.l1,
                 'l2': args.l2,
                 'dropout': args.dropout,
                 'loss': 'categorical_crossentropy',
                 'pad': 24,
                 'overlap': 8},
            'network_fn': args.network_fn},
    }

    return switch.get(key, default)


if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    args, args_str = augment_args(args)

    prep_gpu(args.exp, args.gpu, style=args.gpu_type)

    switch = {
        'self_train': DOT_CV_self_train,
        'da': DOT_CV,
        'cifar10': cifar10,
        'cifar100': cifar100,
        'control': DOT_CV,
        'cd': cd,
        'dw': dw,
        'cl': cl,
    }

    from data_generator import blended_dset, mixup_dset, bc_plus, generalized_bc_plus, fast_fourier_fuckup, fmix_dset, \
        foff_dset

    da_switch = {
        'blended': blended_dset,
        'mixup': mixup_dset,
        'bc+': bc_plus,
        'bc++': generalized_bc_plus,
        'fff': fast_fourier_fuckup,
        'fmix': fmix_dset,
        'foff': foff_dset
    }

    da_args = {
        'n_blended': args.convex_dim,
        'prefetch': args.prefetch,
        'prob': args.convex_prob,
        'std': .05,
        'alpha': args.convex_prob,
        'cache': args.cache,
        'M': args.rand_M,
        'N': args.rand_N
    }


    def default(dset, **kwargs):
        return dset


    # takes: train_ds, n_blended, prefetch, prob, std, alpha
    da_fn = da_switch.get(args.convexAugment, default)

    exp_fn = switch.get(args.exp_type)

    network = network_switch(args, args.exp_type, None)

    network_fn, network_params = network['network_fn'], network['params']

    if args.hyperband:
        vars(args)['util'] = True
        train_dset, val_dset, callbacks = get_dsets(exp_fn, args, da_fn, da_args)
        vars(args)['util'] = False


        def hyperband_wrapper(args, da_args):
            # convert network params and search space to an acceptable function format
            def hyperband_fn(hp):

                special_args = copy(args)
                for key in special_args.search_space:
                    vars(special_args)[key] = hp.Choice(name=key, values=vars(special_args)[key],
                                                        ordered=special_args.search_space[key])

                print(special_args.dropout)

                network = network_switch(special_args, args.exp_type, None)

                network_fn, network_params = network['network_fn'], network['params']

                if args.distributed:
                    # create the scope
                    strategy = tf.distribute.MirroredStrategy()
                    with strategy.scope():
                        # build the model (in the scope)
                        model = network_fn(**network_params)
                else:
                    model = network_fn(**network_params)

                return model

            return hyperband_fn


        hypermodel = hyperband_wrapper(args, da_args)
        print(args.search_space)
        tuner = kt.Hyperband(hypermodel=hypermodel,
                             objective=kt.Objective("val_categorical_accuracy", direction="max"),
                             max_epochs=100,
                             project_name=f'./../hyperband_tuner/{TIMESTRING}')

        tuner.search(train_dset,
                     steps_per_epoch=args.steps_per_epoch,
                     validation_data=val_dset,
                     validation_steps=int(val_dset.cardinality()),
                     epochs=args.epochs,
                     shuffle=False,
                     callbacks=callbacks,
                     initial_epoch=0,
                     use_multiprocessing=True,
                     workers=tf.config.threading.get_inter_op_parallelism_threads())

        print(tuner.results_summary())
    else:
        exp_fn(args, da_fn, da_args, network_fn, network_params)
