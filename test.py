import matplotlib.pyplot as plt
from make_figure import get_train_fractions
from data_structures import ModelData, ModelEvaluator, update_evaluator
import os
import pickle


def figure_metric_epoch(evaluator, title, fname):

    legend = []
    for metric in ['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy']:
        for name, model in zip(['control', 'convex augmentation'], [model for model in evaluator.models]):
            # plot the metric v.s. epochs for each model
            series = model.history[metric]
            plt.plot(range(len(series)), series, linestyle='-')

            legend.append(f'{metric} {name}')

    # add the plot readability information
    plt.title(title)
    plt.legend(legend)
    plt.xlabel('epoch')
    plt.ylabel(f'loss / accuracy')

    # save the figure
    fig = plt.gcf()
    fig.set_size_inches(12, 7.5)
    plt.savefig(fname)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    paths = ['vit_model1659692001.5214853', 'vit_model1659692737.1895618']
    paths = [os.curdir + '/../results/' + path for path in paths]
    evaluator = ModelEvaluator([])
    for path in paths:
        with open(path, 'rb') as fp:
            evaluator.append(pickle.load(fp))

    figure_metric_epoch(evaluator, 'Accuracy and Loss With / Without Augmentation', 'test.png')