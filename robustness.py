import tensorflow as tf
import numpy as np
import sys
import os

import pickle

from data_generator import to_flow, prepare_dset
from keras.preprocessing.image import ImageDataGenerator
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent


def get_val_data(train_fraction, batch_size, examples=None, image_size=(256, 256)):
    """
    retrieve the validation data for the given training fraction at a given batch size

    :param train_fraction: fraction of the available training data to use
    :param batch_size: size of batches to return from generator
    :param examples: maximum number of examples in the returned dataset
    :param image_size: size to reshaped loaded images to
    :return: a tensorflow Dataset of validation data, and the number of batches in the dataset
    """

    # retrieve the validation filepaths and the image data generator
    _, _, val, _, image_gen = prepare_dset(['bronx_allsites/wet',
                                            'bronx_allsites/dry',
                                            'bronx_allsites/snow',
                                            'ontario_allsites/wet',
                                            'ontario_allsites/dry',
                                            'ontario_allsites/snow',
                                            'rochester_allsites/wet',
                                            'rochester_allsites/dry',
                                            'rochester_allsites/snow'],
                                           image_size=image_size,
                                           batch_size=batch_size,
                                           train_fraction=train_fraction)
    examples = examples if len(val) is None else min(examples, len(val))
    # convert dataframe to flow
    val_gen = to_flow(val.sample(examples, random_state=42),
                      image_gen, shuffle=False, image_size=image_size, batch_size=batch_size)
    # convert flow to dataset
    val_dset = tf.data.Dataset.from_generator(lambda: val_gen, output_types=(tf.float32, tf.int32),
                                              output_shapes=([None, 256, 256, 3], [None, ])).prefetch(3)
    # return the dataset and its length
    return val_dset, len(val_gen)


def pgd_attack(model, dataset, epsilons, budget=1, steps=0):
    """
    performs a projected gradient descent attack on the model for each example in the dataset for each epsilon in epsilons

    :param model: ModelData instance model to attack
    :param dataset: data to attack the model on
    :param epsilons: iterable of non-negative floats - max attack norm for PGD
    :param budget: number of iterations is calculated as budget * (eps / delta) where delta is 1e-4 the step size
    :param steps: number of batches in the dataset
    :return: a list of accuracies - the model accuracy for each epsilon over all the attacked examples
    """
    # this will store the perturbed x and the true y for each adversarial example
    table = {eps: {'x': [], 'y': []} for eps in epsilons}
    # for each choice of epsilon
    for i, eps in enumerate(epsilons):
        print(eps)
        # for each batch in the dataset
        for j in range(steps):
            print(j, end='\r')
            # retrieve one batch
            z = dataset.take(1).as_numpy_iterator()
            x, y = None, None
            for zx, zy in z:
                x, y = zx, zy

            delta = 1e-4
            # perform the attack
            x_pgd = projected_gradient_descent(model, x, eps, delta, budget * (eps / delta), np.inf, y=y)
            # clip the result to the allowed range
            x_pgd = tf.clip_by_value(x_pgd, 0.0, 1.0)
            table[eps]['x'].append(x_pgd)
            table[eps]['y'].append(y)
    print()
    for key in table:
        # evaluate the model on the attack points
        table[key] = model.evaluate(np.concatenate(table[key]['x']),
                                    np.concatenate(table[key]['y']),
                                    return_dict=True, steps=steps)['sparse_categorical_accuracy']
    # return the accuracies
    return [table[key] for key in sorted(table.keys())]


def pgd_evaluation(path, epsilons, examples=256, batch_size=64):
    """
    evaluate a model on several epsilons in the PDG adversarial attack regime

    :param path: path to load the ModelData instance from
    :param epsilons: list of PGD norm budgets to try
    :param examples: maximum number of examples to use
    :param batch_size: batch size for tensorflow computation
    """
    # instantiate a model (could also be a TensorFlow or JAX model)
    model = None
    # load the ModelData instance
    with open(path, 'rb') as fp:
        model = pickle.load(fp)
    # silence the prints
    fp = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    # read in the validation data to use for the attack
    val_dset, batches = get_val_data(model.train_fraction, batch_size, examples)
    # compute the clean model accuracy (not under attack)
    clean_acc = model.get_model().evaluate(val_dset, return_dict=True, steps=batches)['sparse_categorical_accuracy']
    sys.stdout = fp
    # stdout is restored
    print(batches)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    # silence stdout
    fp = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    # calculate the accuracy for each pgd epsilon
    robust_accuracy = pgd_attack(model.get_model(), val_dset, epsilons=epsilons, steps=batches)
    sys.stdout = fp
    # stdout is restored
    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    print(f"robust accuracy for perturbations with {model.train_fraction}%, {model.train_iteration} iter")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  inf norm ≤ {eps:<6}: {acc * 100:4.1f} %")


def main() -> None:
    # model_0.30000000000000004_0_1658803078.33387
    mypath = 'results/exp_2/'
    paths = [os.path.join(mypath, f) for f in os.listdir(mypath) if '_10_' in f or '_0_' in f]

    epsilons = [
                   0.0002,
                   0.0005,
                   0.0008,
                   0.001,
                   0.0015,
                   0.002,
                   0.003,
                   0.01,
                   0.1,
                   0.3,
                   0.5,
                   1.0,
               ][:-4]

    for path in paths:
        pgd_evaluation(path, epsilons)


if __name__ == "__main__":
    main()
