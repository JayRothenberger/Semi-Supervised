# code supplied by pip / conda
import matplotlib.pyplot as plt
import gc
from time import time
# code supplied locally
import numpy as np

import experiment
from job_control import JobIterator
from make_figure import *
from data_structures import *
from cnn_network import build_parallel_functional_model, build_patchwise_vision_transformer
from transforms import rand_augment_object, custom_rand_augment_object
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
            'filters': [[32, 64, 128, 128]],
            'kernels': [[4, 2, 1, 1]],
            'hidden': [[4, 4, 4, 4]],
            'l1': [None],
            'l2': [1e-4],
            'dropout': [0],
            'train_iterations': [20],
            'train_fraction': [(i + 1) * .05 for i in range(20)],  # .05, .1, .15, ..., 1.0
            'epochs': [100],
            'augment_batch': [284],
            'sample': [.1],
            'distance_function': ['confidence'],
            'rand_M': [0.1],
            'rand_N': [2],
            'steps_per_epoch': [None],
            'patience': [16],
            'batch': [32],
            'lrate': [1e-4]
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
            'filters': [[32, 64, 128]],
            'kernels': [[4, 1, 1]],
            'hidden': [[3, 3, 3]],
            'l1': [None],
            'l2': [1e-4],
            'dropout': [0.1],
            'train_iterations': [20],
            'train_fraction': [(i + 1) * .05 for i in range(20)],  # .05, .1, .15, ..., 1.0
            'epochs': [100],
            'augment_batch': [284],
            'sample': [.1],
            'distance_function': ['confidence'],
            'rand_M': [0.1],
            'rand_N': [3],
            'steps_per_epoch': [None],
            'patience': [16],
            'batch': [32],
            'lrate': [1e-4]
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

    train_steps = train_steps if train_steps is not None else 5 * train_dset.__len__
    val_steps = val_steps if val_steps is not None else 5 * val_dset.__len__

    print(train_steps, val_steps)

    evaluate_on = dict() if evaluate_on is None else evaluate_on

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=args.patience,
                                                  restore_best_weights=True,
                                                  min_delta=args.min_delta)]

    return experiment.execute_exp(args, args_str, model, train_dset, val_dset, network_fn, network_params,
                                  0, train_steps, val_steps, callbacks=callbacks, evaluate_on=evaluate_on)


def prep_gpu(gpu=False, style='a100'):
    """prepare the GPU for tensorflow computation"""
    # use the available cpus to set the parallelism level
    if args.cpus_per_task is not None:
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)

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
    print('args', args)
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
                                    to_dataset(train_df, train_image_gen, shuffle=True),
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


def explore_lense_channels(dset, num_images, model_path='/../results/test.network', display=False):
    """display num_images from dset"""
    from PIL import Image
    # for each image
    for i in range(num_images):
        imgs = None
        # take candidate image
        ex = dset.take(1)
        # separate image and label
        for x, y in ex:
            imgs = [i for i in x]

        to_show = Image.fromarray(np.uint8(imgs[0] * 255))
        with open(CURRDIR + f'/../visualizations/input{i}.png', 'wb') as fp:
            # save
            to_show.save(fp)
        if display:
            # display
            to_show.show()
        # transform candidate using only the lense block layers
        # get the model data
        with open(CURRDIR + model_path, 'rb') as fp:
            model = pickle.load(fp)
        # get the model
        model = model.get_model()

        new_output = None
        for layer in model.layers:
            try:
                if layer.strides != 1 and layer.strides != (1, 1):
                    break
            except:
                pass
            new_output = layer
        # create a new model with the lense outputs as the outputs
        model = tf.keras.models.Model(inputs=[model.layers[0].input], outputs=[new_output.output])
        opt = tf.keras.optimizers.Nadam(learning_rate=1e-4,
                                        beta_1=0.9, beta_2=0.999,
                                        epsilon=None, decay=0.99)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=opt,
                      metrics=['sparse_categorical_accuracy'])
        # get the lense layer outputs for the first image
        volume = model.predict(ex)
        volume = volume[0, ::]
        import math
        # construct the compound image
        per_row = math.ceil(math.sqrt(volume.shape[-1] + imgs[0].shape[-1]))
        square_size = volume.shape[0]
        arr = np.zeros((per_row * square_size, (((volume.shape[-1] + imgs[0].shape[-1]) // per_row) + 1) * square_size),
                       np.uint8)
        try:
            for j in range(imgs[0].shape[-1]):
                arr[(j % per_row) * square_size:(j + 1 % per_row) * square_size,
                (j // per_row) * square_size:((j // per_row) + 1) * square_size] = np.uint8(
                    (imgs[0][:, :, j] / np.max(imgs[0][:, :, j])) * 255)
            for j in range(volume.shape[-1]):
                volume[:, :, j] += -1 * min(0, np.min(volume[:, :, j]))
            for j in range(imgs[0].shape[-1], volume.shape[-1] + imgs[0].shape[-1]):
                x = (j % per_row) * square_size
                y = (j // per_row) * square_size
                arr[x:x + square_size, y:y + square_size] = np.uint8(
                    (volume[:, :, j - imgs[0].shape[-1]] / np.max(volume[:, :, j - imgs[0].shape[-1]])) * 255)
        except IndexError as v:
            to_show = Image.fromarray(arr)
            if display:
                to_show.show()
            print(j, per_row)
            raise v
        to_show = Image.fromarray(arr)
        if display:
            # display the compound image
            to_show.show()
        with open(CURRDIR + f'/../visualizations/lense_output{i}.png', 'wb') as fp:
            # save it
            to_show.save(fp)


def mult_along_axis(A, B, axis=0):
    """
    multiply 1D array B along an axis of A

    :param A: array along which B will be multiplied
    :param B: 1D array of weights
    :param axis: axis along which B will be multiplied
    :return: the tensor A weighted along some axis by the vector B
    """
    A = np.array(A)
    B = np.array(B)

    # shape check
    if axis >= A.ndim:
        raise ValueError("Bad Shape")
    if A.shape[axis] != B.size:
        raise ValueError("Length of 'A' along the given axis must be the same as B.size")
    # convert the arrays to be broadcastable to the desired shape
    shape = np.swapaxes(A, A.ndim-1, axis).shape
    B_brc = np.broadcast_to(B, shape)
    B_brc = np.swapaxes(B_brc, A.ndim-1, axis)

    return A * B_brc


def blended_dset(train_df, image_gen, batch_size=16, n_blended=2, image_size=(256, 256, 3), prefetch=4):
    # what if we take elements in our dataset and blend them together and predict the mean label?
    # need two train sets to blend together

    datasets = []  # this array will hold all of the datasets we spawn to take batches from to mix together
    for i in range(n_blended):
        datasets.append(to_dataset(train_df, image_gen, True, batch_size=batch_size, seed=i))

    # create a generator which will be turned into the new Dataset object
    def arg_free_gen():
        def random_weighting(n):
            # get a random weighting chosen uniformly from the convex hull of the unit vectors.
            samp = -1 * np.log(np.random.uniform(0, 1, n))
            samp /= np.sum(samp)
            return np.array(samp)
        # the generator yields batches blended together with this weighting
        while True:
            # get a batch from each generator
            batches = [dataset.take(1) for dataset in datasets]
            # split the batches into x and y
            imgs = [[np.stack([i for i in x]) for x, y in batch] for batch in batches]
            labs = [[np.stack([i for i in y]) for x, y in batch] for batch in batches]
            # generate the random weighting
            weights = random_weighting(len(labs))
            m = len(weights)
            imgs, labs = m * mult_along_axis(imgs, weights, 0), m * mult_along_axis(labs, weights, 0)
            # return the convex combination point
            yield np.squeeze(np.mean(np.stack(imgs), axis=0), axis=0), np.squeeze(np.mean(np.stack(labs), axis=0),
                                                                                  axis=0)
    # return the dataset object of the generator
    return tf.data.Dataset.from_generator(arg_free_gen,
                                          output_types=(tf.float32, tf.int32),
                                          output_shapes=([None, *image_size], [None, ])).prefetch(prefetch)


if __name__ == '__main__':
    from data_generator import get_image_dsets

    image_size = (256, 256)
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    args, args_str = augment_args(args)

    prep_gpu(args.gpu)

    paths = ['bronx_allsites/wet', 'bronx_allsites/dry', 'bronx_allsites/snow',
             'ontario_allsites/wet', 'ontario_allsites/dry', 'ontario_allsites/snow',
             'rochester_allsites/wet', 'rochester_allsites/dry', 'rochester_allsites/snow']
    print('getting dsets...')

    train_df, withheld_df, val_df, test_df, ig = get_dataframes_self_train(
        [CURRDIR + '/../data/' + path for path in paths],
        train_fraction=args.train_fraction)

    utility = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                              preprocessing_function=custom_rand_augment_object(.1, 2,
                                                                                                                leq_M=False))
    network_params = {'learning_rate': args.lrate,
                      'conv_filters': args.filters,
                      'conv_size': args.kernels,
                      'attention_heads': args.hidden,
                      'image_size': (image_size[0], image_size[1], 3),
                      'n_classes': 3,
                      'l1': args.l1,
                      'l2': args.l2,
                      'dropout': args.dropout}

    print('hidden', args.hidden)
    network_fn = build_patchwise_vision_transformer
    """
    self_train(args, network_fn, network_params, train_df, val_df, withheld_df,
               augment_fn=custom_rand_augment_object(.1, 2, leq_M=False),
               evaluate_on=None)
    """
    val_dset = to_dataset(val_df, ig)
    train_dset = blended_dset(train_df, ig, args.batch, 2)

    explore_image_dataset(train_dset, 10)

    model = network_fn(**network_params)
    # start_training(args, model, train_dset, val_dset)

    # explore_image_dataset(val_dset, 10)

    # explore_lense_channels(val_dset, 16, display=False)
