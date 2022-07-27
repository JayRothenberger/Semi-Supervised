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
