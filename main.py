"""
Experiment Management Code by Jay Rothenberger (jay.c.rothenberger@ou.edu)
"""

# code supplied by pip / conda
import argparse

# code supplied locally
from job_control import JobIterator
from make_figure import *
from data_structures import *
from experiment import cifar10, cifar100, DOT_CV, DOT_CV_self_train


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
    parser.add_argument('--distributed', action='store_true', help='Use multiple GPUs')

    # High-level experiment configuration
    parser.add_argument('--label', type=str, default="", help="Experiment label")
    parser.add_argument('--exp_type', type=str, default='test', help="Experiment type")
    parser.add_argument('--results_path', type=str, default=os.curdir + '/../results', help='Results directory')
    parser.add_argument('--cv_rotation', type=int, default=0, help='positive int in [0, k],'
                                                                   ' rotation of cross-validation to execute')
    parser.add_argument('--cv_k', type=int, default=4, help='positive int - number of rotations of cross-validation')
    parser.add_argument('--cross_validate', action='store_true', help='Perform k-fold cross-validation')
    parser.add_argument('--cache', action='store_true', help='cache the dataset in memory')

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
    parser.add_argument('--distance_function', type=str, default='confidence', help='Determines k-nearest neighbors '
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
            'convex_dim': [2],
            'convex_prob': [1.0],
            'steps_per_epoch': [512],
            'patience': [32],
            'batch': [48],
            'lrate': [1e-4],
            'randAugment': [False],
            'peek': [False],
            'convexAugment': [False],
            'cross_validate': [False],
            'rand_M': [.1],
            'rand_N': [2],
        },
        'self_train': {
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
            'convex_prob': [.5],
            'steps_per_epoch': [512],
            'patience': [32],
            'batch': [48],
            'augment_batch': [2048],
            'lrate': [1e-4],
            'randAugment': [True],
            'peek': [False],
            'convexAugment': [True],
            'cross_validate': [False],
            'rand_M': [.1],
            'rand_N': [2],
            'sample': [.1],
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
            'convexAugment': [False],
            'cross_validate': [False],
            'rand_M': [.1],
            'rand_N': [2],
            'cfar': [True]
        },
        'cifar10': {
            'filters': [[96, 96]],
            'kernels': [[2, 1]],
            'hidden': [[8, 8, 8]],
            'l1': [None],
            'l2': [None],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [1],
            'epochs': [512],
            'convex_dim': [2],
            'convex_prob': [.5],
            'steps_per_epoch': [256],
            'patience': [32],
            'batch': [128],
            'lrate': [2e-4],
            'randAugment': [True],
            'peek': [False],
            'convexAugment': [True],
            'cross_validate': [False],
            'rand_M': [.1],
            'rand_N': [2],
            'cfar': [True]
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
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    args, args_str = augment_args(args)

    prep_gpu(args.gpu, style=args.gpu_type)

    switch = {
        'self_train': DOT_CV_self_train,
        'test': DOT_CV,
        'cifar10': cifar10,
        'cifar100': cifar100,
        'control': DOT_CV
    }

    exp_fn = switch.get(args.exp_type)
    exp_fn(args)
