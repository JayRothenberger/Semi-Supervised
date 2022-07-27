import argparse
import pickle
import os
import re

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from functools import singledispatchmethod

from sklearn.metrics import confusion_matrix


@dataclass
class ModelData:
    weights: np.ndarray
    network_params: dict
    network_fn: callable
    val_metrics: dict
    train_metrics: dict
    test_metrics: dict
    withheld_metrics: dict
    classes: list
    history: dict
    withheld_predict: object
    withheld_true: object
    val_predict: list
    val_true: list
    test_predict: list
    test_true: list
    train_fraction: float
    train_iteration: int
    args: object

    def report(self):
        # give a performance report for an individual model
        print('train:')
        print(self.train_metrics)
        print('withheld:')
        print(self.withheld_metrics)
        print('val:')
        print(self.val_metrics)
        print('test:')
        print(self.test_metrics)

    def withheld_confusion_matrix(self):
        # generate an ndarray confusion matrix
        return confusion_matrix(self.withheld_true, self.withheld_predict, labels=self.classes)

    def val_confusion_matrix(self):
        # generate an ndarray confusion matrix
        return confusion_matrix(self.val_true, self.val_predict, labels=self.classes)

    def test_confusion_matrix(self):
        # generate an ndarray confusion matrix
        return confusion_matrix(self.test_true, self.test_predict, labels=self.classes)

    def get_history(self):
        # return the history object for the model
        return self.history

    def get_model(self):
        # return the keras model
        model = self.network_fn(**self.network_params)
        model.set_weights(self.weights)

        return model


@dataclass
class ModelEvaluator:
    models: list

    def best_hyperparams(self, train_fraction=None, metric='loss', mode=min):
        """
        finds and returns a dictionary of the best hyperparameters over the model evaluations in the models list, and
        the performance metrics for the corresponding model

        :param train_fraction: fraction of training data to scan over (float, iterable of floats, or None) if None looks over all training fractions
        :param metric: metric over which to optimize
        :param mode: function to define what 'best' means, reasonable choices include: min, max
        :return: a dict of the best hyperparameter name: value pairs
        """

        if isinstance(train_fraction, float):
            train_fraction = [train_fraction]
        elif isinstance(train_fraction, int):
            train_fraction = [train_fraction]
        elif train_fraction is None:
            train_fraction = self.unique_fractions()
        # validation loss minimizing parameters
        # this comprehension returns a ModelData instance
        # it contains the ModelData instance of the stored model and the metrics on which it was evaluated

        best = min([model
                    for model in self.models if model.train_fraction in train_fraction],
                   key=lambda x: x.val_metrics[metric])

        return best

    def to_pickle(self, filename):
        # pickle the object
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)

    def unique_fractions(self):
        # find the unique training fraction amounts
        rax = set()
        for model in self.models:
            rax.add(model.train_fraction)

        return rax

    def performance_fn_data(self, metric='loss', mode=min):
        """
        generates series (best model) performance (according to metric) as a function of the size of the data

        :param metric: the metric to choose the best model over and to plot
        :param mode: function that decides what 'best' means, reasonable choices include: min, max
        :return: four series: train v.s. data, withheld " , val ", test "
        """
        # returns a series of performance v.s. train_fraction
        domain = sorted(list(self.unique_fractions()))

        # series over the domain for respective performances
        train, withheld, val, test = [], [], [], []

        for x in domain:
            try:
                best_model = self.best_hyperparams(x, metric=metric, mode=mode)
                if not best_model.withheld_true:
                    best_model.withheld_metrics[metric] = 'n/a'
                withheld.append((x, best_model.withheld_metrics[metric]))
                train.append((x, best_model.train_metrics[metric]))
                val.append((x, best_model.val_metrics[metric]))
                test.append((x, best_model.test_metrics[metric]))
            except Exception as e:
                print('error: ', e)

        return train, withheld, val, test

    def append(self, model: ModelData):
        self.models.append(model)

    def confusion_matrix(self, metric='loss', train_fraction=None, subset=None):
        best_model = self.best_hyperparams(train_fraction, metric)

        return confusion_matrix(best_model.val_true, best_model.val_predict)


def read_all_pkl(dirname, filebase):
    """
    Read results from dirname from files matching filebase

    :param dirname: directory to read .pkl files from
    :param filebase: prefix that all files to read start with
    :@ return: a list of pickle loaded objects
    """

    # The set of files in the directory
    files = [f for f in os.listdir(dirname) if re.match(r'%s.+' % filebase, f)]
    files.sort()
    results = []

    # Loop over matching files
    for f in files:
        fp = open("%s/%s" % (dirname, f), "rb")
        r = pickle.load(fp)
        fp.close()
        results.append(r)

    return results


def update_evaluator(evaluator, dir='results/'):
    pkl = read_all_pkl(dir, 'model')
    for model in pkl:
        evaluator.append(model)

    return evaluator
