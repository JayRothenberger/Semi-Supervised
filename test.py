import matplotlib.pyplot as plt
from make_figure import get_train_fractions, explore_image_dataset
from data_structures import ModelData, ModelEvaluator, update_evaluator
import os
import numpy as np
import tensorflow as tf

from data_generator import get_dataframes_self_train, to_dataset, mixup_dset, fast_fourier_fuckup


def varying_args(evaluator):
    args_objects = [vars(model.args) for model in evaluator.models]
    # want to calculate set operations, but what if the objects are unhashable?
    # Only the keys are guaranteed to be hashable. - the lists are not
    keys = set()
    for i, args_object in enumerate(args_objects):
        for arg in args_object:
            try:
                if args_object[arg] != args_objects[0][arg]:
                    keys.add(arg)
            except:
                pass

    return keys


def figure_metric_epoch(evaluator, title, fname, metric):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes

    legend = []
    to_plot = []
    to_print = []
    varying = varying_args(evaluator)
    varying -= {'cpus_per_task', 'exp_type', 'exp', 'rand_M', 'rand_N', 'gpu_type', }

    for model in evaluator.models:
        def plot(metric, series, c):
            to_plot.append((series, {'linestyle': '-', 'c': c}))
            vals = vars(model.args)
            prefix = ' & ' + metric + f'({len(model.history["loss"]) - model.args.patience}) & '
            legend.append(prefix + ' & '.join([f'{val} {vals[val]}' for val in varying]) + '\\' + '\\')
            if 'loss' in metric:
                to_print.append((prefix + ' & '.join([f'{val} {vals[val]}' for val in varying]) + '\\' + '\\', min(series)))
            else:
                to_print.append((prefix + ' & '.join([f'{val} {vals[val]}' for val in varying]) + '\\' + '\\',
                                 max(series)))

        # plot the metric v.s. epochs for each model
        r = float(np.random.uniform(0, 1, 1))
        g = float(np.random.uniform(.25, .75, 1))
        b = float(np.random.uniform(.25, .75, 1))
        try:
            # plot(metric, model.history[metric], (r, .25, .25))
            plot('val_' + metric, model.history['val_' + metric], (r, g, b))
        except:
            # plot('sparse_' + metric, model.history['sparse_' + metric], (r, .75, .25))
            plot('val_sparse_' + metric, model.history['val_sparse_' + metric], (r, .75, .75))
    to_plot = sorted(zip(legend, to_plot), key=lambda x: x[0])
    legend = [x for x, y in to_plot]
    for x, y in to_plot:
        plt.plot(y[0], **y[1])
    for string, val in sorted(to_print, key=lambda x: x[1]):
        print(val, string)
    # add the plot readability information
    plt.title(title)
    plt.legend(legend)
    plt.xlabel('epoch')
    plt.ylabel(metric)

    # save the figure
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    plt.savefig(fname)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    paths = ['bronx_allsites/wet', 'bronx_allsites/dry', 'bronx_allsites/snow',
             'ontario_allsites/wet', 'ontario_allsites/dry', 'ontario_allsites/snow',
             'rochester_allsites/wet', 'rochester_allsites/dry', 'rochester_allsites/snow']

    print('getting dsets...')

    _, _, val_df, _, _ = get_dataframes_self_train(
        [os.curdir + '/../data/' + path for path in paths],
        train_fraction=1)

    from data_structures import ModelEvaluator
    from robustness import pgd_evaluation


    def prep_gpu(gpu=False, style='a100'):
        """prepare the GPU for tensorflow computation"""
        # tell tensorflow to be quiet
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    prep_gpu(True)

    dset = to_dataset(val_df, center=True)
    dset = fast_fourier_fuckup(dset, alpha=.4)
    explore_image_dataset(dset, 10)
    exit()


    evaluator = update_evaluator(ModelEvaluator([]), os.curdir + '/../results/cifar10/', fbase='')
    from data_generator import cifar10_dset

    train, val, test = cifar10_dset(batch_size=64)

    for model in evaluator.models:
        print(model.args)

        def get_model(model):
            # return the keras model
            model.network_params['loss'] = 'categorical_crossentropy'
            k_model = model.network_fn(**model.network_params)
            k_model.set_weights(model.weights)
            return k_model

        print(model.get_model().evaluate(val))
        print(model.get_model().evaluate(test))
        continue

        pgd_evaluation(
                       get_model(model),
                       to_dataset(val_df, batch_size=24, shuffle=True, class_mode='categorical'),
                       [1e-4, 1e-3, 1e-2, 1e-1], steps=12
                       )

    for metric, name in [('loss', 'Validation Loss'), ('categorical_accuracy', 'Validation Accuracy')]:
        figure_metric_epoch(evaluator, f'{name}',
                            os.curdir + '/../visualizations/' + f'{metric}_test.png', metric)
