"""
Example code for creating a figure using matplotlib to display tensorflow model performance

Jay Rothenberger (jay.c.rothenberger@ou.edu)
"""

import os
import re

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import *
from data_structures import *


def read_all_pkl(dirname, filebase):
    """
    Read results from dirname from files matching filebase

    :param dirname: directory to read .pkl files from
    :param filebase: prefix that all files to read start with
    :@ return: a list of pickle loaded objects
    """

    # The set of files in the directory
    files = [f for f in os.listdir(dirname) if re.match(r'%s.+.pkl' % filebase, f)]
    files.sort()
    results = []

    # Loop over matching files
    for f in files:
        fp = open("%s/%s" % (dirname, f), "rb")
        r = pickle.load(fp)
        fp.close()
        results.append(r)

    return results


def figure_metric_epoch(evaluator, metric, title, fname, train_fraction=None):
    if isinstance(train_fraction, float):
        train_fraction = [train_fraction]
    elif train_fraction is None:
        train_fraction = get_train_fractions(evaluator)

    legend = []

    for model in [model for model in evaluator.models if model.train_fraction in train_fraction]:
        # plot the metric v.s. epochs for each model
        series = model.history[metric]
        plt.plot(range(len(series)), series, linestyle='-')
        
        # Legend format depends on the experiment type. The experiment type is hinted via 'fname'
        # legend.append(str(model.network_params))
        if "basic" in fname:
            legend.append(f"l1={model.network_params['l1']}, l2={model.network_params['l2']}, "
                          f"dropout={model.network_params['dropout']}")
        elif "train_fraction" in fname:
            legend.append(f"train_fraction={str(model.train_fraction)[0:4]}")

    # add the plot readability information
    plt.title(title)
    plt.legend(legend)
    plt.xlabel('epoch')
    plt.ylabel(f'{metric}')

    # save the figure
    fig = plt.gcf()
    fig.set_size_inches(12, 7.5)
    plt.savefig(fname)
    plt.clf()
    plt.close()


def save_confusion_matrix_fig(matrix, labels, title, fname):
    display = sklearn.metrics.ConfusionMatrixDisplay(matrix, display_labels=labels)
    display.plot()
    plt.title(title)
    plt.savefig(fname)
    plt.clf()


def get_train_fractions(evaluator):
    # returns a set of the available training fractions represented in the evaluator
    rax = set()
    for model in evaluator.models:
        rax.add(model.train_fraction)
    return rax


def val_metric_vs_train_iterations(abbrv, dir='results/exp_1/'):
    """
    Retrieve the ModelData files and plot the validation metric as a function of the training iterations

    :param abbrv: 'acc' or 'loss' the abbreviation for the metric to evaluate
    :param dir: directory from which to load ModelData instances
    :return: a list of (train_fraction,
    """

    names = {
        'acc': ('Accuracy', 'sparse_categorical_accuracy'),
        'loss': ('(Sparse) Categorical Crossentropy', 'loss')
    }

    name, metric = names[abbrv]

    evaluator = ModelEvaluator([])
    evaluator = update_evaluator(evaluator, dir)
    s = {model.train_fraction: [] for model in evaluator.models}

    colors = []
    for model in sorted(evaluator.models, key=lambda model: model.train_iteration):
        s[model.train_fraction].append(model.val_metrics[metric])
        colors.append((model.train_fraction, 0, 0))

    for key in s:
        plt.plot(range(len(s[key])), s[key], color=(key, 0, 0))
    plt.legend([f'labeled fraction {str(key)[:4]}' for key in sorted(s.keys())])
    plt.xlabel('training iterations')
    plt.ylabel(metric)
    plt.xticks(range(19))

    plt.title(f'Validation {name} as a Function of Training Iteration')
    plt.show()

    s1 = {model.train_fraction: [] for model in evaluator.models}

    for model in sorted(evaluator.models, key=lambda model: model.train_iteration):
        s1[model.train_fraction].append(model.val_metrics)

    rax = []
    for key in s1:
        rax.append(
            (
                key,
                s1[key][0][metric],
                max([0] + [i[metric] for i in s1[key][1:]]),
                max([0] + [i[metric] for i in s1[key][1:]]) - s1[key][0][metric]
            )
        )

    return rax


def print_metric_fn_training_fraction(metric='sparse_categorical_accuracy', fp="model_evaluations.txt"):
    """
    print and write to a file a series of the metric as a function of the training fraction

    :param metric: metric to display over training fractions
    :param fp: name of the file to save the printed information to
    """
    evaluator = ModelEvaluator([])
    evaluator = update_evaluator(evaluator)
    train, withheld, val, test = [], [], [], []
    analysis = open(fp, "w")

    print(len(evaluator.models))
    for model in evaluator.models:
        x = model.train_fraction
        if model.withheld_true:
            withheld.append((x, model.withheld_metrics[metric]))
        else:
            withheld.append((x, 'n/a'))
        train.append((x, model.train_metrics[metric]))
        val.append((x, model.val_metrics[metric]))
        test.append((x, model.test_metrics[metric]))

    for series, name in zip([train, withheld, val, test], ['train', 'withheld', 'val', 'test']):
        print(sorted(series, key=lambda a: a[0]))
        analysis.write(f'{name}: {sorted(series, key=lambda a: a[0])}\n\n')

    analysis.close()


def basic_supervised(dir='results/'):
    """
    If we're displaying results for the basic SL experiment, we run this code

    :param dir: directory from which to load ModelData instances
    """
    evaluator = ModelEvaluator([])
    evaluator = update_evaluator(evaluator, dir)
    rows = []

    for model in evaluator.models:
        save_confusion_matrix_fig(model.val_confusion_matrix(),
                                  labels=model.classes,
                                  title=(
                                      f"Validation Confusion Matrix with l1={model.network_params['l1']}, "
                                      f"l2={model.network_params['l2']}, dropout={model.network_params['dropout']}"
                                  ),
                                  fname=(
                                      f"val_matrix_l1_{model.network_params['l1']}__l2_{model.network_params['l2']}__"
                                      f"drop_{model.network_params['dropout']}.png"
                                  )
                                  )
        rows.append((model.train_metrics['sparse_categorical_accuracy'],
                     model.val_metrics['sparse_categorical_accuracy'], model.network_params['l1'],
                     model.network_params['l2'], model.network_params['dropout']))

    df = pd.DataFrame(rows, columns=["Training Accuracy", "Validation Accuracy", "L1", "L2", "Dropout"])
    df.to_csv('accuracy_evaluations.csv')

    # Creating the plots of training accuracy (for basic Supervised Learning models)
    figure_metric_epoch(evaluator, metric='sparse_categorical_accuracy',
                        title="Training Accuracy as a Function of Epochs",
                        fname="train_accuracy_plots_basic.png"
                        )
    # Creating the plots of validation accuracy (for basic Supervised Learning models)
    figure_metric_epoch(evaluator, metric='val_sparse_categorical_accuracy',
                        title="Validation Accuracy as a Function of Epochs",
                        fname="valid_accuracy_plots_basic.png"
                        )


# make figure
def train_fraction_split(dir='results/'):
    """
    If we're displaying results for the train-fraction-split SL experiment, we run this code

    :param dir: directory from which to load ModelData instances
    """
    evaluator = ModelEvaluator([])
    evaluator = update_evaluator(evaluator, dir)
    rows = []

    for value in get_train_fractions(evaluator):
        best = evaluator.best_hyperparams(value)
        save_confusion_matrix_fig(best.val_confusion_matrix(),
                                  labels=best.classes,
                                  title=f"Validation Confusion Matrix with Training Fraction={str(value)[0:4]}",
                                  fname=f"val_matrix_train_fraction_{str(value)[0:4]}.png")
        rows.append((best.train_metrics['sparse_categorical_accuracy'],
                     best.val_metrics['sparse_categorical_accuracy'],
                     str(value)[0:4]))
    df = pd.DataFrame(rows, columns=["Training Accuracy", "Validation Accuracy", "Training Fraction"])
    df.to_csv('accuracy_evaluations_train_fraction.csv')

    # Creating the plots for the train-fraction-split Supervised Learning models
    figure_metric_epoch(evaluator, metric='sparse_categorical_accuracy',
                        title="Training Accuracy as a Function of Epochs",
                        fname="train_accuracy_plots_train_fraction.png"
                        )
    # Creating the plots of validation accuracy (for basic Supervised Learning models)
    figure_metric_epoch(evaluator, metric='val_sparse_categorical_accuracy',
                        title="Validation Accuracy as a Function of Epochs",
                        fname="valid_accuracy_plots_train_fraction.png"
                        )


def explore_image_dataset(dset, num_images):
    """display num_images from dset"""
    from PIL import Image

    for i in range(num_images):
        imgs = None
        for x, y in dset.take(1):
            imgs = [i for i in x]

        img = Image.fromarray(np.uint8(imgs[0] * 255))
        img.show()
