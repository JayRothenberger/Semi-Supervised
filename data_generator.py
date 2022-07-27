import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from transforms import rand_augment_object

# I included this because when I didn't I got an error.
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_file_name_tuples(dirname, classmap):
    """
    Takes a directory name and a dictionary that serves as a map between the class names and the class integger labels,
    and returns a list of tuples (filename, class)
    
    :param dirname: a directory name local to the running process
    :param classmap: a dictionary of str: int
    :return: a list of tuples (filename, class) parsed from dirname
    """
    return [(f'{dirname}/{f}', classmap[dirname.split('/')[-1]]) for f in os.listdir(dirname)]


def df_from_dirlist(dirlist):
    """
    Creates a pandas dataframe from the files in a list of directories
    
    :param dirlist: a list of directories where the last sub-directory is the class label for the images within
    :return: a pandas dataframe with columns=['filepath', 'class']
    """
    # remove trailing slash in path (it will be taken care of later)
    dirlist = [dirname if dirname[-1] != '/' else dirname[:-2] for dirname in dirlist]

    # determine the list of classes by directory names
    # E.g. "foo/bar/cat - (split) -> ['foo', 'bar', 'cat'][-1] == 'cat'
    classes = sorted(list(set([dirname.split('/')[-1] for dirname in dirlist])))

    # enumerate(['wet', 'dry', 'snow']) = [(0, 'dry'), (1, 'snow'), (2, 'wet')]
    # dict = {key: value}, where key = 'dry', 'snow', 'wet', and value = '0', '1', '2'
    class_map = {c: str(i) for (i, c) in enumerate(classes)}
    print(class_map)
    # find all of the file names
    names = sum([get_file_name_tuples(dirname, class_map) for dirname in dirlist], [])
    return pd.DataFrame(names, columns=['filepath', 'class'])


def train_val_test_split(df, val_fraction=.2, test_fraction=.2, shuffle=True):
    # Read documentation?
    # Random state (pseudorandom)
    """
    Takes a dataframe and returns splits for training, testing, and validation
    
    :param df: dataframe containing all examples
    :param val_fraction: fraction of data to be used for validation (float \in (0, 1))
    :param test_fraction: fraction of data to be used for testing (float \in (0, 1))
    :param shuffle: shuffle before splitting (bool)
    :return: three dataframes (train, val, test)
    """
    train, test = train_test_split(df, test_size=val_fraction + test_fraction, shuffle=shuffle, random_state=42)
    test, val = train_test_split(test, test_size=val_fraction / (val_fraction + test_fraction), random_state=42)

    return train, val, test


def report(df):
    if df is None:
        print('None')
        return None

    # for each site, for each class
    # site is filepath.split('/')[0]
    # class is filepath.split('/')[-1]
    sites = set()
    classes = set()

    for index, row in df.iterrows():
        split_path = row['filepath'].split('/')
        sites.add(split_path[0])
        classes.add(row['class'])
    # map all sites / classes to their row / column indices in the table
    sites = {site: i for i, site in enumerate(sorted(list(sites)))}
    classes = {c: i for i, c in enumerate(sorted(list(classes)))}
    # assign the table
    table = np.zeros((len(sites), len(classes)))
    # iterate through the dataset incrementing the appropriate table values to get a table of counts
    for index, row in df.iterrows():
        split_path = row['filepath'].split('/')
        # table[site, class] += 1
        table[sites[split_path[0]], classes[row['class']]] += 1

    for site in sites:
        for c in classes:
            print(f'{site} ({c}): {table[sites[site], classes[c]]}')

    return table


def data_generators(df, val_dirlist=None, image_size=(128, 128), batch_size=16, train_fraction=.05):
    """
    Return train, val, test generators that generate batches of data to be input for training or evaluation of a
    model

    :param df: dataframe containing all of the data
    :param image_size: shape to resize images to tuple of (int, int)
    :param batch_size: batch size for generated batches
    :param val_dirlist: (optional) list of directories for validation and testing data
    :param train_fraction: fraction of training data to use
    :param rand_augment: tuple of (M, N) parameters for rand augment
    :return: returns four data generators train, withheld, val, test
    """

    if val_dirlist is None:
        train, val, test = train_val_test_split(df)
    else:
        train, (val, test) = df, train_test_split(val_dirlist, test_size=.5, shuffle=True, random_state=42)

    if train_fraction < 1:
        train, train_withheld = train_test_split(train, test_size=(1 - train_fraction), shuffle=False)
    else:
        train, train_withheld = train, None

    print('train: ')
    report(train)
    print('validation: ')
    report(val)
    print('test: ')
    report(test)
    print('training withheld')
    report(train_withheld)

    image_gen = ImageDataGenerator(rescale=1. / 255)

    train_gen = image_gen.flow_from_dataframe(train,
                                              x_col='filepath',
                                              y_col='class',
                                              target_size=image_size,
                                              color_mode='rgb',
                                              class_mode='sparse',
                                              batch_size=batch_size)

    withheld_gen = None

    if train_fraction < 1:
        withheld_gen = image_gen.flow_from_dataframe(train_withheld,
                                                     x_col='filepath',
                                                     y_col='class',
                                                     target_size=image_size,
                                                     color_mode='rgb',
                                                     class_mode='sparse',
                                                     batch_size=batch_size,
                                                     shuffle=False)

    val_gen = image_gen.flow_from_dataframe(val,
                                            x_col='filepath',
                                            y_col='class',
                                            target_size=image_size,
                                            color_mode='rgb',
                                            class_mode='sparse',
                                            batch_size=batch_size,
                                            shuffle=False)

    test_gen = image_gen.flow_from_dataframe(test,
                                             x_col='filepath',
                                             y_col='class',
                                             target_size=image_size,
                                             color_mode='rgb',
                                             class_mode='sparse',
                                             batch_size=batch_size,
                                             shuffle=False)

    return train_gen, withheld_gen, val_gen, test_gen


def split_dataframes(df, val_dirlist=None, batch_size=16, train_fraction=.05):
    """
    Return train, val, test generators that generate batches of data to be input for training or evaluation of a
    model

    :param df: dataframe containing all of the data
    :param batch_size: batch size for generated batches
    :param val_dirlist: (optional) list of directories for validation and testing data
    :param train_fraction: fraction of training data to use
    :param rand_augment: tuple of (M, N) parameters for rand augment
    :return: returns three data generators train, val, test
    """
    if val_dirlist is None:
        train, val, test = train_val_test_split(df)
    else:
        train, (val, test) = df, train_test_split(val_dirlist, test_size=.5, shuffle=True, random_state=42)

    if train_fraction < 1:
        train, train_withheld = train_test_split(train, test_size=(1 - train_fraction), shuffle=False)
    else:
        train, train_withheld = train, None

    print('train: ')
    report(train)
    print('training withheld: ')
    report(train_withheld)
    print('validation: ')
    report(val)
    print('test: ')
    report(test)

    image_gen = ImageDataGenerator(rescale=1. / 255)

    return train, train_withheld, val, test, image_gen


def to_flow(df, image_gen, shuffle=False, image_size=(256, 256), batch_size=16):
    if df is None:
        return None

    return image_gen.flow_from_dataframe(df,
                                         x_col='filepath',
                                         y_col='class',
                                         target_size=image_size,
                                         color_mode='rgb',
                                         class_mode='sparse',
                                         batch_size=batch_size,
                                         shuffle=shuffle)


def prepare_dset(train_dirlist, val_dirlist=None, image_size=(128, 128), batch_size=16, train_fraction=.05):
    """
    returns data generators created as flows from dataframes of all file paths

    :param train_dirlist: list of directories from which to parse example file names for training data
    :param val_dirlist: list of directories for validation examples
    :param image_size: height and width dimensions of the images fed to nn (performs resizing)
    :param batch_size: batch size for example batches produced by generators
    :param train_fraction: fraction of available training data to use
    :param rand_augment: tuple of (M, N) parameters for rand augment
    :return: train_gen, val_gen, test_gen, withheld_gen image data generators for train, val, test, more train
    """
    if val_dirlist is None:
        return split_dataframes(df_from_dirlist(train_dirlist),
                                batch_size=batch_size,
                                train_fraction=train_fraction)
    else:
        return split_dataframes(df_from_dirlist(train_dirlist),
                                df_from_dirlist(val_dirlist),
                                batch_size=batch_size,
                                train_fraction=train_fraction)
