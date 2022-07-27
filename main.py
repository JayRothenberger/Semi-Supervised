# code supplied by pip / conda
import matplotlib.pyplot as plt
import gc
from time import time
# code supplied locally
import experiment
from job_control import JobIterator
from make_figure import *
from data_structures import *
from cnn_network import build_parallel_functional_model
from transforms import rand_augment_object
from data_generator import augment_with_neighbors, get_dataframes_self_train, to_flow, to_dataset

CURRDIR = os.path.dirname(__file__)


def create_parser():
    """
    Create argument parser
    """
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


def start_training(args, model, train_dset, val_dset, evaluate_on=None, train_steps=None, val_steps=None):
    # Override arguments if we are using exp_index
    args, args_str = augment_args(args)

    train_steps = train_steps if train_steps is not None else train_dset.__len__
    val_steps = val_steps if val_steps is not None else val_dset.__len__

    print(train_steps, val_steps)

    evaluate_on = dict() if evaluate_on is None else evaluate_on

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=args.patience,
                                                  restore_best_weights=True,
                                                  min_delta=args.min_delta)]

    return experiment.execute_exp(args, args_str, model, train_dset, val_dset, network_fn, network_params,
                                  0, train_steps, val_steps, callbacks=callbacks, evaluate_on=evaluate_on)


def prep_gpu(gpu=False, style='a100'):
    """prepare the GPU for tensorflow computation"""
    # Turn off GPU?
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif style == 'a100':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # just use one GPU
    else:
        pass  # do nothing (v100)

    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU')
    n_physical_devices = len(physical_devices)

    if n_physical_devices > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('We have %d GPUs\n' % n_physical_devices)
    else:
        print('NO GPU')


def self_train(args, network_fn, network_params, train_df, val_df, withheld_df, augment_fn=None, evaluate_on=None):
    args, args_str = augment_args(args)

    import distances
    dist_func_selector = {
        'euclidean': distances.euclidean,
        'euclidean_with_confidence': distances.euclidean_with_confidence,
        'confidence': distances.confidence
    }

    train_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                      preprocessing_function=augment_fn)

    default_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    model = None
    train_old = None
    labeled = train_df
    for train_iteration in range(args.train_iterations):
        if not train_iteration or args.retrain_fully:
            model = network_fn(**network_params)

        model_data = start_training(args,
                                    model,
                                    to_dataset(train_df, train_image_gen),
                                    to_dataset(val_df, default_image_gen),
                                    train_steps=args.steps_per_epoch,
                                    evaluate_on={'val': to_flow(val_df, default_image_gen),
                                                 'withheld': to_flow(withheld_df, default_image_gen),
                                                 'test': to_flow(test_df, default_image_gen)}
                                    )

        with open('%s/model_%s_%s_%s' % (args.results_path,
                                         str(args.train_fraction),
                                         str(train_iteration),
                                         str(time())), 'wb') as fp:
            pickle.dump(model_data, fp)

        distance_function = dist_func_selector.get(args.distance_function)

        if distance_function is None:
            raise ValueError('unrecognized experiment type')

        if withheld_df is not None and (len(withheld_df) - args.augment_batch) > 0:
            print(len(withheld_df))
            tf.keras.backend.clear_session()
            gc.collect()

            # replace the generators and dataframes with the updated ones from augment_with_neighbors
            train_df, withheld_df, labeled = augment_with_neighbors(args,
                                                                    model,
                                                                    image_size,
                                                                    distance_fn=distance_function,
                                                                    image_gen=default_image_gen,
                                                                    pseudolabeled_data=train_df,
                                                                    unlabeled_data=withheld_df,
                                                                    labeled_data=labeled,
                                                                    sample=args.sample)
        else:
            print('not enough data for an additional iteration of training!')
            print(f'stopped after {train_iteration} iterations!')
            break
        print('retraining: ', train_iteration)


if __name__ == '__main__':
    from data_generator import get_image_dsets

    image_size = (256, 256)
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    prep_gpu(args.gpu)

    paths = ['bronx_allsites/wet', 'bronx_allsites/dry', 'bronx_allsites/snow',
             'ontario_allsites/wet', 'ontario_allsites/dry', 'ontario_allsites/snow',
             'rochester_allsites/wet', 'rochester_allsites/dry', 'rochester_allsites/snow']
    print('getting dsets...')

    train_df, withheld_df, val_df, test_df, ig = get_dataframes_self_train(
        [CURRDIR + '/../data/' + path for path in paths],
        train_fraction=args.train_fraction)

    utility = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                              preprocessing_function=rand_augment_object(.5, 2,
                                                                                                         leq_M=False))
    network_params = {'learning_rate': args.lrate,
                      'conv_filters': args.filters,
                      'conv_size': args.kernels,
                      'dense_layers': args.hidden,
                      'image_size': (image_size[0], image_size[1], 3),
                      'n_classes': 3,
                      'l1': args.l1,
                      'l2': args.l2,
                      'dropout': args.dropout}

    network_fn = build_parallel_functional_model

    # self_train(args, network_fn, network_params, train_df, val_df, withheld_df, augment_fn=None, evaluate_on=None)

    val_dset = to_dataset(val_df, utility)
    explore_image_dataset(val_dset, 10)

    # tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True,
    #                          to_file=CURRDIR + '/../visualizations/model.png')
