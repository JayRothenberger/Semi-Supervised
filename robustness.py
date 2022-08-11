import tensorflow as tf
import numpy as np
import sys
import os
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

from make_figure import explore_image_dataset


def pgd_attack(model, dataset, epsilons, budget=4, steps=0, norm=np.inf):
    """
    performs a projected gradient descent attack on the model for each example in the dataset for each epsilon in epsilons

    :param model: ModelData instance model to attack
    :param dataset: data to attack the model on
    :param epsilons: iterable of non-negative floats - max attack norm for PGD
    :param budget: number of iterations is calculated as budget * 4, and the step size is epsilon / 4
    :param steps: number of batches in the dataset
    :param norm: norm by which to bound the pgd attack influence
    :return: a list of accuracies - the model accuracy for each epsilon over all the attacked examples
             a list of datasets containing the adversarial examples
    """
    # this will store the perturbed x and the true y for each adversarial example
    table = {eps: {'x': [], 'y': []} for eps in epsilons}
    # for each choice of epsilon
    exes = []
    for i, eps in enumerate(epsilons):
        print(eps)
        # for each batch in the dataset
        counter = 0
        for x, y in dataset:
            counter += 1
            delta = eps / 4
            # perform the attack
            x_pgd = projected_gradient_descent(model, x, eps, delta, budget * (eps / delta), norm,
                                               y=np.array(np.argmax(y, axis=-1)))
            # clip the result to the allowed range
            x_pgd = tf.clip_by_value(x_pgd, 0.0, 1.0)
            table[eps]['x'].append(x_pgd)
            table[eps]['y'].append(y)
            if steps < counter:
                break

        X = tf.data.Dataset.from_tensor_slices((np.concatenate(table[eps]['x']), np.concatenate(table[eps]['y']))).batch(16)
        # evaluate the model on the attack points
        table[eps] = model.evaluate(X,
                                    return_dict=True, steps=steps)['categorical_accuracy']

        exes.append(X)
    # return the accuracies
    return [table[key] for key in sorted(table.keys())], exes


def pgd_evaluation(model, dset, epsilons, steps=0, norm=np.inf):
    """
    evaluate a model on several epsilons in the PDG adversarial attack regime

    :param model: keras model instance
    :param dset: tensorflow dataset that returns batches of examples to perturb and evaluate
    :param epsilons: list of PGD norm budgets to try
    :param steps: number of batches to take from the dataset for evaluation
    :param norm: norm by which to bound the pgd attack influence
    """
    # compute the clean model accuracy (not under attack)
    clean_acc = model.evaluate(dset, return_dict=True, steps=steps)['categorical_accuracy']
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    # calculate the accuracy for each pgd epsilon
    fp = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    robust_accuracy, exes = pgd_attack(model, dset, epsilons=epsilons, steps=steps, norm=norm)
    for x, eps in zip(exes, epsilons):
        explore_image_dataset(x, 1, fname=f'{eps}')
    sys.stdout = fp
    # stdout is restored
    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    print(f"robust accuracy:")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  inf norm ≤ {eps:<6}: {acc * 100:4.1f} %")
