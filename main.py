from make_figure import *
from data_structures import *
import matplotlib.pyplot as plt


def val_metric_vs_train_iterations(abbrv, dir='results/exp_1/'):
    """
    Retrieve the ModelData files and plot the validation metric as a function of the training iterations

    :param abbrv: 'acc' or 'loss' the abbreviation for the metric to evaluate
    :param dir: directory from which to load ModelData instances
    :return: a list of (train_fraction,
    """
    data = ModelData()

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


if __name__ == '__main__':
    from transforms import rand_augment_object
    from data_generator import prepare_dset
    from PIL import Image
    from data_generator import to_flow

    rand_augment = rand_augment_object(.1, 4, leq_M=True)
    image_size = (256, 256)
    # Split metadata into individual data sets
    train, withheld, val, test, image_gen = prepare_dset(['bronx_allsites/wet',
                                                          'bronx_allsites/dry',
                                                          'bronx_allsites/snow',
                                                          'ontario_allsites/wet',
                                                          'ontario_allsites/dry',
                                                          'ontario_allsites/snow',
                                                          'rochester_allsites/wet',
                                                          'rochester_allsites/dry',
                                                          'rochester_allsites/snow'],
                                                         image_size=image_size,
                                                         batch_size=16, train_fraction=1)

    # convert dataframes to flow
    train_gen, withheld_gen, val_gen, test_gen = to_flow(train,
                                                         image_gen,
                                                         shuffle=True, image_size=image_size, batch_size=16), \
                                                 to_flow(withheld,
                                                         image_gen,
                                                         shuffle=False, image_size=image_size, batch_size=16), \
                                                 to_flow(val,
                                                         image_gen,
                                                         shuffle=False, image_size=image_size, batch_size=16), \
                                                 to_flow(test,
                                                         image_gen,
                                                         shuffle=False, image_size=image_size, batch_size=16)
    metrics = val_metric_vs_train_iterations('acc', dir='results/exp_2/')
    print('\n'.join([', '.join([str(i)[:6] for i in m]) for m in sorted(metrics, key=lambda k: k[0])]))
