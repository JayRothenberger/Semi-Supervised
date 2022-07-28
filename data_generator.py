import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import perf_counter as perf_time
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


def split_dataframes(df, val_dirlist=None, train_fraction=.05):
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


def to_dataset(df, image_gen, shuffle=False, image_size=(256, 256), batch_size=16, prefetch=4):
    if df is None:
        return None

    gen = image_gen.flow_from_dataframe(df,
                                        x_col='filepath',
                                        y_col='class',
                                        target_size=image_size,
                                        color_mode='rgb',
                                        class_mode='sparse',
                                        batch_size=batch_size,
                                        shuffle=shuffle)

    dset = tf.data.Dataset.from_generator(lambda: gen, output_types=(tf.float32, tf.int32),
                                          output_shapes=([None, 256, 256, 3], [None, ])).prefetch(prefetch)

    dset.__len__ = len(gen)

    return dset


def get_dataframes_self_train(train_dirlist, val_dirlist=None, train_fraction=.05):
    """
    returns data generators created as flows from dataframes of all file paths

    :param train_dirlist: list of directories from which to parse example file names for training data
    :param val_dirlist: list of directories for validation examples
    :param image_size: height and width dimensions of the images fed to nn (performs resizing)
    :param batch_size: batch size for example batches produced by generators
    :param train_fraction: fraction of available training data to use
    :return: train_df, withheld_df, val_df, test_df image path dataframes for train, val, test, more train
    """
    if val_dirlist is None:
        return split_dataframes(df_from_dirlist(train_dirlist),
                                train_fraction=train_fraction)
    else:
        return split_dataframes(df_from_dirlist(train_dirlist),
                                df_from_dirlist(val_dirlist),
                                train_fraction=train_fraction)


def augment_with_neighbors(args, model, image_size, image_gen, distance_fn, labeled_data, unlabeled_data,
                           hard_labels=True, pseudolabeled_data=None, sample=None, pseudo_relabel=True):
    """
    augments the pseudolabeled_data dataframe with a batch of k nearest neighbors from the unlabeled_data dataframe

    dataframes to supply image filepaths expect a 'filepath' column

    :param args: CLA from the argument parser for the experiment
    :param model: model to generate pseudo-labels
    :param image_size: size to reshape image height and width dimensions
    :param image_gen: image generator used to compute withheld flow from dataframe
    :param distance_fn: function to compute distance for computing nearest neighbors
    :param labeled_data: training paths dataframe does not include the pseudolabeled points.
    :param unlabeled_data: unlabeled data paths dataframe
    :param hard_labels: if True returns the hard labels instead of the soft probability vector output
    :param pseudolabeled_data: df or None the points to compute the nearest neighbors of in the unlabeled set.
                         If None it takes the whole training set.
    :param sample: float or None fraction of available labeled data to use to calculate the nearest neighbors.
                   If None, uses all of the data.
    :param pseudo_relabel: the teacher assign new pseudo-labels to each unlabeled point at each iteration
    :return: train_gen, withheld_gen, train, withheld
    """
    unlabeled_data_gen = to_flow(unlabeled_data, shuffle=False, batch_size=args.batch, image_size=image_size,
                                 image_gen=image_gen)
    if args.augment_batch > len(unlabeled_data):
        raise ValueError('Not enough unlabeled data to augment with')

    if pseudolabeled_data is None:
        pseudolabeled_data = labeled_data

    if sample is None:
        sample = 1

    # compute the distance between every image pair
    distances = []
    # pre-loading examples from disk
    print('starting the timer')
    start = perf_time()

    # retrieve top-1 confidence for each prediction on the unlabeled data
    top_1 = np.max(model.predict(unlabeled_data_gen), axis=1)
    # this list holds (enumerate_index, (df_index, unlabeled_image_tensor)) elements (unlabeled images)
    ulab_img_array = [(i,
                       tf.keras.preprocessing.image.img_to_array(
                           tf.keras.preprocessing.image.load_img(unlabeled['filepath'], target_size=image_size)))
                      for i, (j, unlabeled) in enumerate(unlabeled_data.iterrows())]

    print(f'loaded unlabeled images ({image_size}): ', perf_time() - start)
    start = perf_time()

    print(f'calculated top-1 ({image_size}): ', perf_time() - start)
    start = perf_time()
    # this is the array over which the distance to the unlabeled points is computed (either labeled or pseudolabeled)
    # it has the same structure as the previous list, but this one has labeled or pseudolabeled images
    lab_img_array = [(i,
                      tf.keras.preprocessing.image.img_to_array(
                          tf.keras.preprocessing.image.load_img(labeled['filepath'], target_size=image_size)))
                     for i, (j, labeled) in enumerate(
            labeled_data.iloc[
                np.random.choice(range(len(labeled_data)),
                                 int(sample * len(labeled_data)), replace=False)].iterrows())] \
        if args.closest_to_labeled else [(i,
                                          tf.keras.preprocessing.image.img_to_array(
                                              tf.keras.preprocessing.image.load_img(labeled['filepath'],
                                                                                    target_size=image_size)))
                                         for i, (j, labeled) in enumerate(
            pseudolabeled_data.iloc[
                np.random.choice(range(len(labeled_data)),
                                 int(sample * len(labeled_data)), replace=False)].iterrows())]

    print(f'loaded labeled images ({image_size}): ', perf_time() - start)
    # now we have loaded two arrays of image tensors
    start = perf_time()
    count = 0
    # for each unlabeled image
    for i, img0 in ulab_img_array:
        count += 1
        print(f'{count} / {len(ulab_img_array)} ({perf_time() - start}s)                                              ',
              end='\r')
        # record the minimum distance between this unlabeled image and all labeled images in 'distances'
        distances.append(
            min([(i, distance_fn(img0, img1, top_1[i])) for j, img1 in lab_img_array], key=lambda x: x[-1]))

    print()
    print('finished computing distance:', perf_time() - start)
    start = perf_time()
    # sort the unlabeled points by their distance from any point we have the label for
    distances = sorted(distances, key=lambda x: x[-1])
    print(f'{len(distances)} distances')
    print('sorting took: ', perf_time() - start)
    # take the k images with the smallest distance values
    k_nearest_indices = [index for index, distance in distances[:args.augment_batch]]

    # get the image tensors corresponding to the appropriate indices from the previous line
    pseudo_labeled_batch = unlabeled_data.iloc[k_nearest_indices]
    # we will drop these indices from the unlabeled set
    unlabeled_to_drop = pseudo_labeled_batch.index

    def df_difference(df1, df2, column='filepath'):
        # computes the set difference col1 - col2
        d1 = {x: i for i, x in enumerate(df1[column])}
        d2 = {x: i for i, x in enumerate(df2[column])}

        return df1.iloc[[d1[i] for i in set(d1.keys()) - set(d2.keys())]]

    # if we are to re-label our pseudo-labeled data, we will need to find
    if pseudo_relabel:
        pseudo_labeled_batch = pd.concat([df_difference(pseudolabeled_data, labeled_data), pseudo_labeled_batch],
                                         ignore_index=True)

    pseudo_labeled_batch_gen = to_flow(pseudo_labeled_batch, shuffle=False, batch_size=args.batch,
                                       image_size=image_size,
                                       image_gen=image_gen)

    # set the classes as the pseudo-labels
    if hard_labels:
        pseudo_labels = np.argmax(model.predict(pseudo_labeled_batch_gen), axis=1)
    else:
        pseudo_labels = model.predict(pseudo_labeled_batch_gen)
    # set the assign the pseudo-labels
    pseudo_labeled_batch['class'] = pseudo_labels
    pseudo_labeled_batch['class'] = pseudo_labeled_batch['class'].astype(str)
    # remove those examples from the withheld set
    unlabeled_data = unlabeled_data.drop(index=unlabeled_to_drop)
    # add them to training set
    if pseudo_relabel:
        pseudolabeled_data = pd.concat((labeled_data, pseudo_labeled_batch), ignore_index=True)
    else:
        pseudolabeled_data = pd.concat((pseudolabeled_data, pseudo_labeled_batch), ignore_index=True)

    # return new frames
    return pseudolabeled_data, unlabeled_data, labeled_data


def get_image_dsets(data_paths, path_prefix, image_size=(256, 256), batch_size=16, prefetch=3, augment_fn=None,
                    augment_args=None, train_fraction=None):
    """


    """
    # path_prefix = CURRDIR + '/../data/'
    # the path string to the data directory relative to this file
    data_paths = [path_prefix + f for f in data_paths]

    train, withheld, val, test, image_gen = prepare_dset(data_paths,
                                                         image_size=image_size,
                                                         batch_size=batch_size,
                                                         train_fraction=train_fraction)

    augment_args = augment_args if isinstance(augment_args, dict) else dict()

    train_process = ImageDataGenerator(rescale=1. / 255,
                                       preprocessing_function=augment_fn(**augment_args))
    # convert dataframes to flow
    train_gen, withheld_gen, val_gen, test_gen = to_flow(train,
                                                         train_process,
                                                         shuffle=True, image_size=image_size, batch_size=batch_size), \
                                                 to_flow(withheld,
                                                         image_gen,
                                                         shuffle=False, image_size=image_size, batch_size=batch_size), \
                                                 to_flow(val,
                                                         image_gen,
                                                         shuffle=False, image_size=image_size, batch_size=batch_size), \
                                                 to_flow(test,
                                                         image_gen,
                                                         shuffle=False, image_size=image_size, batch_size=batch_size)

    # convert flows to datasets... (CANNOT have the same name as the flow)
    train_dset = tf.data.Dataset.from_generator(lambda: train_gen, output_types=(tf.float32, tf.int32),
                                                output_shapes=([None, 256, 256, 3], [None, ])).prefetch(prefetch)

    val_dset = tf.data.Dataset.from_generator(lambda: val_gen, output_types=(tf.float32, tf.int32),
                                              output_shapes=([None, 256, 256, 3], [None, ])).prefetch(prefetch)

    return train_dset, val_dset
