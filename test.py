import matplotlib.pyplot as plt
from make_figure import get_train_fractions, explore_image_dataset
from data_structures import ModelData, ModelEvaluator, update_evaluator
import os
import numpy as np
import tensorflow as tf
import gc

from lime import lime_image
import shap

from skimage.io import imread
from skimage.segmentation import mark_boundaries

from data_generator import get_dataframes_self_train, to_dataset, mixup_dset, fast_fourier_fuckup, cifar10_dset, \
    fmix_dset, foff_dset, cats_dogs, deep_weeds, citrus_leaves

from data_structures import ModelEvaluator
from robustness import pgd_evaluation


def explain_image_classifier_with_lime(model, instance, n_classes):
    """
    show a visual explanation using LIME for an image classification keras Model and a image instance with matplotlib
    """
    instance = np.array(instance)
    explainer = lime_image.LimeImageExplainer(kernel_width=.125)
    explanation = explainer.explain_instance(instance.astype(np.double), model.predict, top_labels=n_classes,
                                             hide_color=0, num_samples=2048, batch_size=16)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=True)
    plt.imshow(mark_boundaries(temp / 2 + .5, mask))
    plt.show()


def explain_image_classifier_with_shap(model, instance, class_names):
    instance = np.array(instance).astype(np.double)
    print(instance.shape)
    # define a masker that is used to mask out partitions of the input image, this one uses a blurred background
    masker = shap.maskers.Image("blur(128,128)", instance[0].shape)

    # By default the Partition explainer is used for all  partition explainer
    explainer = shap.Explainer(model.predict, masker, output_names=class_names)

    # here we use 500 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(np.expand_dims(instance[0], 0), max_evals=4096, batch_size=64, outputs=shap.Explanation.argsort.flip)
    shap.image_plot(shap_values)


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

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes

    legend = []
    to_plot = []
    to_print = []
    varying = sorted(
        list(varying_args(evaluator) - {'cpus_per_task', 'exp_type', 'exp', 'rand_M', 'rand_N', 'gpu_type'}))

    rows = dict()

    for model in evaluator.models:
        def plot(metric, series, c):
            to_plot.append((series, {'linestyle': '-', 'c': c}))
            vals = {str(i): str(vars(model.args)[i]) for i in vars(model.args)}
            prefix = f'({np.sum([np.prod(v.shape) for v in model.weights])}) & ' + f'({len(model.history["loss"]) - model.args.patience}) & '
            legend.append(
                f'{np.sum([np.prod(v.shape) for v in model.weights])}'
            )
            # to_print.append((' & '.join([f'{val}' for val in varying]), min(series)))
            if 'loss' in metric:
                to_print.append(
                    (prefix + ' & '.join([f'{vals[val]}' for val in varying]) + '\\' + '\\', min(series)))
                rows[tuple([vals[val] for val in varying])] = (len(model.history["loss"]) - model.args.patience,
                                                               min(series))
            else:
                to_print.append((prefix + ' & '.join([f'{vals[val]}' for val in varying]) + '\\' + '\\',
                                 max(series)))
                rows[(*[vals[val] for val in varying],)] = (len(model.history["loss"]) - model.args.patience,
                                                            max(series))

        # plot the metric v.s. epochs for each model
        r = float(np.random.uniform(0, 1, 1))
        g = float(np.random.uniform(.25, .75, 1))
        b = float(np.random.uniform(.25, .75, 1))
        try:
            # plot(metric, model.history[metric], (r, .25, .25))
            plot('val_' + metric, model.history['val_' + metric], (r, g, b))
        except:
            # plot('sparse_' + metric, model.history['sparse_' + metric], (r, .75, .25))
            plot('val_' + metric, model.history['val_' + metric], (r, .75, .75))
            pass
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

    return rows


def pgd_eval(model, val, test):
    print(model.args)

    def get_model(model):
        # return the keras model
        model.network_params['loss'] = 'categorical_crossentropy'
        k_model = model.network_fn(**model.network_params)
        k_model.set_weights(model.weights)
        return k_model

    try:
        pgd_evaluation(
            get_model(model),
            test,
            [1e-4, 1e-3, 1e-2, 1e-1], steps=12
        )
    except Exception as e:
        print(e)


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


def to_shape(a, shape):
    if len(shape) == 3:
        y_, x_, _ = shape
        y, x, _ = a.shape
    else:
        _, y_, x_, _ = shape
        _, y, x, _ = a.shape
    y_pad = (y_ - y)
    x_pad = (x_ - x)

    a, _ = tf.linalg.normalize(a, 1, axis=-1)

    if len(shape) == 3:
        return np.pad(a, (
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (x_pad // 2, x_pad // 2 + x_pad % 2),
            (0, 0)
        ),
                      mode='constant')
    return np.pad(a, (
        (0, 0),
        (y_pad // 2, y_pad // 2 + y_pad % 2),
        (x_pad // 2, x_pad // 2 + x_pad % 2),
        (0, 0)
    ),
                  mode='constant')


def get_mask(model, image):
    model = model.get_model()
    new_outputs = []
    d = -1
    for i, layer in enumerate(model.layers[::-1]):
        if 'dense' in layer.name:
            d = -i - 1
            new_outputs.append(layer.output)
            break
    else:
        raise ValueError('dense layer not found')

    for i, layer in enumerate(model.layers):
        try:
            if 'chkpt' in layer.name:
                new_outputs.append(model.layers[d](layer.output))
        except Exception as e:
            print(e)
    model = tf.keras.models.Model(inputs=[model.input], outputs=new_outputs)
    model.compile()
    pred = model.predict(image)
    return np.concatenate([to_shape(z, max([p.shape for p in pred], key=lambda k: k[1])) for z in pred], 2), \
           [tf.reduce_sum(img[0], axis=(0, 1)) for img in pred]


def get_mask_simple(model, image):
    model = model.get_model()
    new_outputs = model.layers[-3].output
    model = tf.keras.models.Model(inputs=[model.input], outputs=[new_outputs])
    model.compile()
    return model.predict(image)


def color_squish(x):
    # a couple candidate colors
    colors = [(240, 163, 255), (0, 117, 220), (153, 63, 0), (76, 0, 92), (25, 25, 25), (0, 92, 49),
              (43, 206, 72), (255, 204, 153), (128, 128, 128), (148, 255, 181), (143, 124, 0), (157, 204, 0),
              (194, 0, 136), (0, 51, 128), (255, 164, 5), (255, 168, 187), (66, 102, 0), (255, 0, 16),
              (94, 241, 242), (0, 153, 143), (224, 255, 102), (116, 10, 255), (153, 0, 0), (255, 255, 128),
              (255, 255, 0), (255, 80, 5)][:x.shape[-1]]
    colors = np.array(colors, np.float32)
    x = tf.cast(x, tf.float32)

    return np.array(tf.einsum('ijk,kl->ijl', x, colors)).astype(np.uint8), colors


def show_mask_simple(dset, num_images, model, fname=''):
    from PIL import Image

    masks = []
    none = []
    imgs = []

    for x, y in iter(dset):
        imgs.append(x)
        output = get_mask_simple(model, x)
        print(output.shape)
        masks.append(output)
        none.append(output)
        num_images -= 1
        if num_images <= 0:
            break

    for i, img in enumerate(imgs):
        img = img[0] + np.max(np.min(img[0]), 0)
        img = img - np.max(np.min(img), 0)
        img = Image.fromarray(np.uint8((img / np.max(img)) * 255))
        with open(os.curdir + f'/../visualizations/pictures/{fname}_{i}.jpg', 'wb') as fp:
            img.save(fp)

    for i, img in enumerate(masks):
        img = np.uint8((img[0] / np.max(img[0])) * 255)
        img = Image.fromarray(img)

        with open(os.curdir + f'/../visualizations/pictures/{fname}_{i}_mask.jpg', 'wb') as fp:
            img.save(fp)

    for i, img in enumerate(none):
        ima = imgs[i]
        ima = ima[0] + np.max(np.min(ima[0]), 0)
        ima = ima - np.max(np.min(ima), 0)
        ima = (ima / np.max(ima)) * 255

        im = masks[i][0]
        im = np.int32((im / np.max(im)) * 255)

        img = np.int32((img[0] / np.max(img[0])) * 255)
        # img = np.stack(img, -1)
        # img = np.swapaxes(img, 1, -1)
        print(img.shape, im.shape)
        # lmao
        im = tf.nn.relu(im - (img / 3))

        img = .5 * (tf.cast(im, tf.float32) + ima)

        img = Image.fromarray(np.uint8(img))
        im = Image.fromarray(np.uint8(im))

        with open(os.curdir + f'/../visualizations/pictures/{fname}_{i}_min.jpg', 'wb') as fp:
            img.save(fp)

        with open(os.curdir + f'/../visualizations/pictures/{fname}_{i}_none.jpg', 'wb') as fp:
            im.save(fp)


def labeled_multi_image(rows, n_cols, row_labs=None, col_labs=None, colors=None, class_names=None):
    """
    row_labs: a list of row labels
    rows: a list of tensors with matching dimension 2
    col_labs: column labels
    n_cols: number of columns
    """
    colors = [tuple([c / 255 for c in color]) for color in colors] if colors is not None else colors
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(n_cols*2, 2*len(rows)))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(len(rows), n_cols),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    col_width = (rows[0].shape[1] // n_cols)
    rows = [[row[:, j * col_width:(j + 1) * col_width, :] for j in range(n_cols)] for i, row in enumerate(rows)]

    rows = [im for sublist in rows for im in sublist]
    print(grid.get_geometry())
    for i, (ax, im) in enumerate(zip(grid, rows)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(col_labs[i % n_cols])
        ax.set_ylabel(row_labs[i // n_cols])
    fig.legend(handles=[mpatches.Patch(color=color, label=class_name)
                        for color, class_name in zip(colors, class_names)])
    plt.show()


def show_mask(dset, num_images, model, class_names, fname=''):
    from PIL import Image
    values = []
    masks = []
    none = []
    imgs = []

    for x, y in iter(dset):
        imgs.append(x)
        output, probs = get_mask(model, x)
        values.append(probs)
        masks.append(output[:, :, :, :-1])
        none.append(output[:, :, :, -1])
        num_images -= 1
        if num_images <= 0:
            break

    print([mask.shape for mask in masks])

    for i, img in enumerate(imgs):
        img = img[0] + np.max(np.min(img[0]), 0)
        img = img - np.max(np.min(img), 0)
        img = Image.fromarray(np.uint8((img / np.max(img)) * 255))
        with open(os.curdir + f'/../visualizations/pictures/{fname}_{i}.jpg', 'wb') as fp:
            img.save(fp)

    for i, img in enumerate(masks):
        img, colors = color_squish(img[0])
        img = Image.fromarray(img)

        with open(os.curdir + f'/../visualizations/pictures/{fname}_{i}_mask.jpg', 'wb') as fp:
            img.save(fp)

    for i, img in enumerate(none):
        ima = imgs[i]
        ima = ima[0] + np.max(np.min(ima[0]), 0)
        ima = ima - np.max(np.min(ima), 0)
        ima = (ima / np.max(ima)) * 255
        ima = np.concatenate([ima for i in range(len(values[i]))], 1)

        im = masks[i][0]

        img = np.float32(img[0])
        img = np.stack([img for i in range(im.shape[-1])], -1)

        # lmao
        im = tf.nn.relu(im - (img / len(class_names)))
        im, colors = color_squish(im)
        print(im.shape, ima.shape)
        img = (tf.cast(im, tf.float32) + ima) * .5

        img = tf.cast(img, tf.uint8)
        ima = tf.cast(ima, tf.uint8)

        label = values[i]
        normed = [[round(float(ele), 3) for ele in tf.linalg.normalize(lab[:-1], 1)[0]] + ['n/a'] for lab in label]

        def to_label(counts, normed):
            counts = [[l_2 for l_2 in str(np.array(l, int)).replace(']', '').replace('[', '').split(' ') if l_2]
                      for l in counts]

            normed = [[l_2.strip("'") for l_2 in str(np.array(l)).replace(']', '').replace('[', '').split(' ') if l_2]
                      for l in normed]

            labels = ['\n'.join([f'{cl}: {count} p={prob}' for count, prob, cl in zip(c, p, class_names + ['none'])])
                      for c, p in zip(counts, normed)]

            return labels

        labeled_multi_image([img, im, ima], len(to_label(label, normed)), row_labs=['overlay', 'mask', 'image'],
                            col_labs=to_label(label, normed),
                            colors=colors, class_names=class_names)
        continue
        img = Image.fromarray(np.uint8(img))
        im = Image.fromarray(np.uint8(im))

        with open(os.curdir + f'/../visualizations/pictures/{fname}_{i}_{normed}_min.jpg', 'wb') as fp:
            img.save(fp)

        with open(os.curdir + f'/../visualizations/pictures/{fname}_{i}_{label}_.jpg', 'wb') as fp:
            im.save(fp)


if __name__ == "__main__":
    # paths to the dot data
    paths = ['bronx_allsites/wet', 'bronx_allsites/dry', 'bronx_allsites/snow',
             'ontario_allsites/wet', 'ontario_allsites/dry', 'ontario_allsites/snow',
             'rochester_allsites/wet', 'rochester_allsites/dry', 'rochester_allsites/snow']
    # get the dataframes that hold the [image path, class] information
    _, _, val_df, test_df, _ = get_dataframes_self_train(
        [os.curdir + '/../data/' + path for path in paths],
        train_fraction=1)
    # prepare the gpu for computation
    prep_gpu(True)
    # switch that will match loaded models with the appropriate dataset
    exp_type_to_dset = {
        'da': to_dataset(test_df, class_mode='categorical'),
        'cd': cats_dogs(batch_size=16)[-1],
        'dw': deep_weeds(batch_size=16)[-1],
        'cl': citrus_leaves(batch_size=8)[-1],
    }
    # switch that will match loaded models with the appropriate class names
    exp_type_to_classes = {
        'da': ['dry', 'snow', 'wet'],
        'cd': ['cats', 'dogs'],
        'dw': ['chinee', 'lantana', 'parkinsonia', 'parenthenium', 'prickly', 'rubber', 'siam', 'snake', 'none'],
        'cl': ['black spot', 'canker', 'greening', 'healthy'],
    }

    exp_type_to_path = {
        'da': 'dot',
        'cd': 'cats_dogs',
        'dw': 'deep_weeds',
        'cl': 'citrus_leaves',
    }

    evaluator = update_evaluator(ModelEvaluator([]), os.curdir + '/../results/citrus_leaves/', fbase='')

    for i, model in enumerate(evaluator.models[::-1]):
        print(max(model.history['val_categorical_accuracy']))
        test = exp_type_to_dset[model.args.exp_type]
        class_names = exp_type_to_classes[model.args.exp_type]
        show_mask(test, 1, model, class_names, str(i))
        keras_model = model.get_model()
        for x, y in iter(test):
            explain_image_classifier_with_lime(keras_model, x[0], len(class_names))
            explain_image_classifier_with_shap(keras_model, x, class_names)
            break

    exit()
    evaluator = update_evaluator(ModelEvaluator([]), os.curdir + '/../results/params/', fbase='')

    for metric, name in [('loss', 'Validation Loss'), ('categorical_accuracy', 'Validation Accuracy')]:
        rows = figure_metric_epoch(evaluator, f'{name}',
                                   os.curdir + '/../visualizations/' + f'{metric}_test.png', metric)
        print('\n'.join(sorted([str(row) for row in rows])))
    exit()
    train, val, test = cifar10_dset(batch_size=64)
    for model in evaluator.models:
        pgd_eval(model, val, test)
