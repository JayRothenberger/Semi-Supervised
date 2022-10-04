import matplotlib.pyplot as plt
from make_figure import get_train_fractions, explore_image_dataset
from data_structures import ModelData, ModelEvaluator, update_evaluator
import os
import numpy as np
import tensorflow as tf

from data_generator import get_dataframes_self_train, to_dataset, mixup_dset, fast_fourier_fuckup, cifar10_dset, \
    fmix_dset, foff_dset

from data_structures import ModelEvaluator
from robustness import pgd_evaluation


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
    varying = varying_args(evaluator)
    varying -= {'cpus_per_task', 'exp_type', 'exp', 'rand_M', 'rand_N', 'gpu_type'}

    rows = dict()

    for model in evaluator.models:
        def plot(metric, series, c):
            to_plot.append((series, {'linestyle': '-', 'c': c}))
            vals = {str(i): str(vars(model.args)[i]) for i in vars(model.args)}
            prefix = f'({np.sum([np.prod(v.shape) for v in model.weights])}) & ' + metric + f'({len(model.history["loss"]) - model.args.patience}) & '
            legend.append(
                          f'{np.sum([np.prod(v.shape) for v in model.weights])}'
                          )
            to_print.append((' & '.join([f'{val}' for val in varying]), min(series)))
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


def get_mask(model, image):
    model = model.get_model()
    new_output = model.layers[-3].output
    model = tf.keras.models.Model(inputs=[model.input], outputs=[new_output])
    model.compile()
    return model.predict(image)


if __name__ == "__main__":
    paths = ['bronx_allsites/wet', 'bronx_allsites/dry', 'bronx_allsites/snow',
             'ontario_allsites/wet', 'ontario_allsites/dry', 'ontario_allsites/snow',
             'rochester_allsites/wet', 'rochester_allsites/dry', 'rochester_allsites/snow']

    print('getting dsets...')

    _, _, val_df, _, _ = get_dataframes_self_train(
        [os.curdir + '/../data/' + path for path in paths],
        train_fraction=1)

    prep_gpu(True)
    # train, val, test = cifar10_dset(batch_size=64)
    val = to_dataset(val_df, class_mode='categorical')
    # explore_image_dataset(val, 10)
    evaluator = update_evaluator(ModelEvaluator([]), os.curdir + '/../results/forced_segmentation/', fbase='')

    def show_mask(dset, num_images, model, fname=''):
        from PIL import Image

        masks = []
        none = []
        imgs = []

        for x, y in iter(dset):
            imgs.append(x)
            output = get_mask(model, x)
            masks.append(output[:, :, :, :-1])
            none.append(output[:, :, :, -1])
            num_images -= 1
            if num_images <= 0:
                break

        for i, img in enumerate(imgs):
            img = img[0] + np.max(np.min(img[0]), 0)
            img = img - np.max(np.min(img), 0)
            img = Image.fromarray(np.uint8((img / np.max(img)) * 255))
            with open(os.curdir + f'/../visualizations/pictures/{i}_{fname}.jpg', 'wb') as fp:
                img.save(fp)

        for i, img in enumerate(masks):
            img = np.uint8((img[0] / np.max(img[0])) * 255)
            img = Image.fromarray(img)

            with open(os.curdir + f'/../visualizations/pictures/{i}_{fname}_mask.jpg', 'wb') as fp:
                img.save(fp)

        for i, img in enumerate(none):
            ima = imgs[i]
            ima = ima[0] + np.max(np.min(ima[0]), 0)
            ima = ima - np.max(np.min(ima), 0)
            ima = (ima / np.max(ima)) * 255

            im = masks[i][0]
            im = np.int32((im / np.max(im)) * 255)

            img = np.int32((img[0] / np.max(img[0])) * 255)
            img = np.stack([img for i in range(3)], -1)
            print(img.shape)
            # lmao
            im = tf.nn.relu(im - (img / 3))

            img = np.max(tf.cast(im, tf.float32) + ima, 255)

            img = Image.fromarray(np.uint8(img))
            im = Image.fromarray(np.uint8(im))

            with open(os.curdir + f'/../visualizations/pictures/{i}_{fname}_min.jpg', 'wb') as fp:
                img.save(fp)

            with open(os.curdir + f'/../visualizations/pictures/{i}_{fname}_none.jpg', 'wb') as fp:
                im.save(fp)

    show_mask(val, 10, evaluator.models[0])
    exit()
    evaluator = update_evaluator(ModelEvaluator([]), os.curdir + '/../results/params/', fbase='')

    for metric, name in [('loss', 'Validation Loss'), ('categorical_accuracy', 'Validation Accuracy')]:
        rows = figure_metric_epoch(evaluator, f'{name}',
                            os.curdir + '/../visualizations/' + f'{metric}_test.png', metric)
        print('\n'.join(sorted([str(row) for row in rows])))
    # exit
    train, val, test = cifar10_dset(batch_size=64)
    for model in evaluator.models:
        pgd_eval(model, val, test)
