import tensorflow as tf
import numpy as np
import sys
import os
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent


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
        counter = 0
        for x, y in dataset:
            counter += 1
            delta = 1e-4
            # perform the attack
            x_pgd = projected_gradient_descent(model, x, eps, delta, budget * (eps / delta), np.inf, y=np.array([np.argmax(y)]))
            # clip the result to the allowed range
            x_pgd = tf.clip_by_value(x_pgd, 0.0, 1.0)
            table[eps]['x'].append(x_pgd)
            table[eps]['y'].append(y)
            if steps < counter:
                break

    for key in table:
        print(np.concatenate(table[key]['y']).shape)
        X = tf.data.Dataset.from_tensor_slices((np.concatenate(table[key]['x']), np.concatenate(table[key]['y']))).batch(1)
        # evaluate the model on the attack points
        table[key] = model.evaluate(X,
                                    return_dict=True, steps=steps)['categorical_accuracy']
    # return the accuracies
    return [table[key] for key in sorted(table.keys())]


def pgd_evaluation(model, dset, epsilons, steps=0):
    """
    evaluate a model on several epsilons in the PDG adversarial attack regime

    :param model: keras model instance
    :param epsilons: list of PGD norm budgets to try
    :param examples: maximum number of examples to use
    :param batch_size: batch size for tensorflow computation
    """
    # silence the prints
    fp = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    # compute the clean model accuracy (not under attack)
    clean_acc = model.evaluate(dset, return_dict=True, steps=steps)['sparse_categorical_accuracy']
    sys.stdout = fp
    # stdout is restored
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    # silence stdout
    fp = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    # calculate the accuracy for each pgd epsilon
    robust_accuracy = pgd_attack(model, dset, epsilons=epsilons, steps=steps)
    sys.stdout = fp
    # stdout is restored
    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    print(f"robust accuracy:")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  inf norm ≤ {eps:<6}: {acc * 100:4.1f} %")
