"""
Example code for creating a figure using matplotlib to display tensorflow model performance

Jay Rothenberger (jay.c.rothenberger@ou.edu)
"""
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from data_structures import *
import tensorflow as tf
from time import time


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
    """

    """
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
    """
    takes in a square numpy matrix, a list of labels, a title for the figure, and a filename

    saves a confusion matrix figure as fname
    """
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


def explore_image_dataset(dset, num_images, fname=''):
    """save num_images from dset for visualization"""
    from PIL import Image

    imgs = []

    for x, y in iter(dset):
        imgs.append(x)
        num_images -= 1
        if num_images <= 0:
            break

    for i, img in enumerate(imgs):
        print(img.shape)
        img = Image.fromarray(np.uint8(img[0] * 255))
        with open(os.curdir + f'/../visualizations/perturbations/{i}_{fname}.jpg', 'wb') as fp:
            img.save(fp)


def explore_lense_channels(dset, num_images, model_path='/../results/vit_model_95', display=False):
    """display an num_images from dset and the output channels of the lense convolution block"""
    from PIL import Image
    # for each image
    for i in range(num_images):
        imgs = None
        # take candidate image
        # separate image and label
        for x, y in dset.take(1):
            imgs = [x]

        to_show = Image.fromarray(np.uint8(imgs[0][0] * 255))
        with open(os.curdir + f'/../visualizations/input{i}.png', 'wb') as fp:
            # save
            to_show.save(fp)
        if display:
            # display
            to_show.show()
        # transform candidate using only the lense block layers
        # get the model data
        with open(os.curdir + model_path, 'rb') as fp:
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
        volume = model.predict(np.array([imgs[0][0][:, :, :3], ]))[0]

        import math
        # construct the compound image
        per_row = math.ceil(math.sqrt(volume.shape[-1] + imgs[0].shape[-1]))
        square_size = imgs[0][0].shape[0]
        arr = np.zeros((per_row * square_size, (((volume.shape[-1] + imgs[0].shape[-1]) // per_row) + 1) * square_size),
                       np.uint8)
        try:
            for j in range(imgs[0].shape[-1]):
                arr[(j % per_row) * square_size:(j + 1 % per_row) * square_size,
                (j // per_row) * square_size:((j // per_row) + 1) * square_size] = np.uint8(
                    (imgs[0][0][:, :, j] / np.max(imgs[0][0][:, :, j])) * 255)
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
        with open(os.curdir + f'/../visualizations/lense_output{i}.png', 'wb') as fp:
            # save it
            to_show.save(fp)
