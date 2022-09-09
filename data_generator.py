import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import perf_counter as perf_time

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


def split_dataframes(df, val_dirlist=None, train_fraction=.05):
    """
    Return train, withheld, val, test dataframes to be turned into datasets or flows to be used as input for training or
    evaluation of a model

    :param df: dataframe containing all of the data
    :param val_dirlist: (optional) list of directories for validation and testing data
    :param train_fraction: fraction of training data to use
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


def get_cv_rotation(dirlist, rotation=0, k=5, train_fraction=1):
    """
    Return train, withheld, val, test dataframes

    :param dirlist: list of directories containing the data
    :param rotation: rotation of cross validation
    :param k: number of folds for cross_validation
    :param train_fraction: fraction of training data to use
    :return: returns three data generators train, val, test
    """
    assert isinstance(k, int), "k should be integer"
    assert isinstance(rotation, int), "rotation should be integer"
    assert rotation < k, "rotation must always be less than k"
    assert rotation >= 0, "rotation must be strictly nonnegative"
    assert k >= 1, "k must be strictly positive"

    # want to split this into k shards and then
    df = df_from_dirlist(dirlist)
    # shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # shard the dataframe
    shards = [df[i * (len(df) // (k + 1)):(i + 1) * (len(df) // (k + 1))] if i < k else df[i * (len(df) // (k + 1)):]
              for i in range(k + 1)]
    # test is always the last shard
    test, shards = shards[-1], shards[:-1]
    val = shards[rotation]
    train = pd.concat([shards[i] for i in range(k) if i != rotation])

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

    return train, train_withheld, val, test


def cifar100_dset(path='./../cifar-100/', batch_size=32, prefetch=1, center=True, cache=True):
    import pickle
    with open(path + 'train', 'rb') as fo:
        train = pickle.load(fo, encoding='bytes')
    with open(path + 'test', 'rb') as fo:
        test = pickle.load(fo, encoding='bytes')

    def re_ravel(x):
        return np.concatenate([np.reshape(x[i * 1024:(i + 1) * 1024], (32, 32, 1), 'C') for i in range(3)], axis=-1)

    print(train.keys(), test.keys())
    x_train = np.stack([re_ravel(i) for i in train[b'data']])
    y_train = np.stack(train[b'fine_labels'])

    x_test = np.stack([np.reshape(i, (32, 32, 3), 'F') for i in test[b'data']])
    y_test = np.stack(test[b'fine_labels'])

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2, shuffle=True, random_state=42)

    train, val, test = tf.data.Dataset.from_tensor_slices((x_train, y_train)), \
                       tf.data.Dataset.from_tensor_slices((x_val, y_val)), \
                       tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def preprocess_image(x, y, center=True):
        """
        Load in image from filename and resize to target shape.
        """

        image = tf.image.convert_image_dtype(x, tf.float32)
        if center:
            image = image - tf.reduce_mean(image)

        label = tf.one_hot(y, 100, dtype=tf.float32)

        return image, label

    train, val, test = train.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, center], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE), \
                       val.map(
                           lambda x, y: tf.py_function(preprocess_image, inp=[x, y, center],
                                                       Tout=(tf.float32, tf.float32)),
                           num_parallel_calls=tf.data.AUTOTUNE), \
                       test.map(
                           lambda x, y: tf.py_function(preprocess_image, inp=[x, y, center],
                                                       Tout=(tf.float32, tf.float32)),
                           num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        train, val, test = train.cache(), val.cache(), test.cache()

    train, val, test = train.repeat().batch(batch_size).prefetch(prefetch), \
                       val.batch(batch_size).prefetch(prefetch), \
                       test.batch(batch_size).prefetch(prefetch)

    return train, val, test


def cifar10_dset(path='./../cifar-10/', batch_size=32, prefetch=1, center=True, cache=True):
    import pickle
    train = []
    with open(path + 'data_batch_1', 'rb') as fo:
        train.append(pickle.load(fo, encoding='bytes'))
    with open(path + 'data_batch_2', 'rb') as fo:
        train.append(pickle.load(fo, encoding='bytes'))
    with open(path + 'data_batch_3', 'rb') as fo:
        train.append(pickle.load(fo, encoding='bytes'))
    with open(path + 'data_batch_4', 'rb') as fo:
        train.append(pickle.load(fo, encoding='bytes'))
    with open(path + 'data_batch_5', 'rb') as fo:
        train.append(pickle.load(fo, encoding='bytes'))
    with open(path + 'test_batch', 'rb') as fo:
        test = pickle.load(fo, encoding='bytes')

    def re_ravel(x):
        return np.concatenate([np.reshape(x[i * 1024:(i + 1) * 1024], (32, 32, 1), 'C') for i in range(3)], axis=-1)

    x_train, y_train = [], []

    for batch in train:
        x_train.append(np.stack([re_ravel(i) for i in batch[b'data']]))
        y_train.append(np.stack(batch[b'labels']))
    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
    print('cifar10:', len(y_train), 'training examples')

    x_test = np.stack([re_ravel(i) for i in test[b'data']])
    y_test = np.stack(test[b'labels'])

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.1, shuffle=False, random_state=42)

    train, val, test = tf.data.Dataset.from_tensor_slices((x_train, y_train)), \
                       tf.data.Dataset.from_tensor_slices((x_val, y_val)), \
                       tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def preprocess_image(x, y, center=True):
        """
        Load in image from filename and resize to target shape.
        """

        image = tf.image.convert_image_dtype(x, tf.float32)
        if center:
            image = image - tf.reduce_mean(image)

        label = tf.one_hot(y, 10, dtype=tf.float32)

        return image, label

    train, val, test = train.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, center], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE), \
                       val.map(
                           lambda x, y: tf.py_function(preprocess_image, inp=[x, y, center],
                                                       Tout=(tf.float32, tf.float32)),
                           num_parallel_calls=tf.data.AUTOTUNE), \
                       test.map(
                           lambda x, y: tf.py_function(preprocess_image, inp=[x, y, center],
                                                       Tout=(tf.float32, tf.float32)),
                           num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        train, val, test = train.cache(), val.cache(), test.cache()

    train, val, test = train.repeat().batch(batch_size).prefetch(prefetch), \
                       val.batch(batch_size).prefetch(prefetch), \
                       test.batch(batch_size).prefetch(prefetch)

    return train, val, test


def load_unlabeled(dirlist):
    file_lists = []
    for directory in dirlist:
        file_list = []
        for path, directories, files in os.walk(directory):
            for file in files:
                file_list.append(os.path.join(path, file))
        file_lists.append(file_list)

    names = sum(file_lists, [])
    print(f'parsed unlabeled data: {len(names)} examples found')
    return pd.DataFrame(names, columns=['filepath'])


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


def to_dataset(df, shuffle=False, image_size=(256, 256), batch_size=16, prefetch=1, seed=42,
               class_mode='sparse', center=False, cache=True, repeat=True, batch=True, **kwargs):
    if df is None:
        return None

    def preprocess_image(item, target_shape, center=True):
        """
        Load in image from filename and resize to target shape.
        """

        filename, label = item[0], item[1]

        image_bytes = tf.io.read_file(filename)
        image = tf.io.decode_image(image_bytes)  # this line does not work kinda
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_shape)
        if center:
            image = image - tf.reduce_mean(image)

        if class_mode == 'categorical':
            label = tf.one_hot(tf.strings.to_number(label, tf.dtypes.int32), 3, dtype=tf.float32)
        elif class_mode == 'sparse':
            label = tf.strings.to_number(label, tf.dtypes.int32)
        else:
            raise ValueError('improper class mode')

        return image, label

    try:
        df['class'] = df['class'].astype(int).astype(str)
    except KeyError as e:
        print('setting class manually to 0...')
        df['class'] = ['0' for i in range(len(df))]
    except ValueError as e:
        print('setting class manually to 0...')
        df['class'] = ['0' for i in range(len(df))]
    slices = df.to_numpy()
    out_type = tf.int32 if class_mode == 'sparse' else tf.float32

    ds = tf.data.Dataset.from_tensor_slices(
        slices
    )

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(len(slices), seed, True)

    ds = ds.map(lambda x:
                tf.py_function(func=preprocess_image,
                               inp=[x, image_size],
                               Tout=(tf.float32, out_type)),
                num_parallel_calls=tf.data.AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    if batch:
        ds = ds.batch(batch_size)

    return ds.prefetch(prefetch)


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


def augment_with_neighbors(args, model, image_size, distance_fn, labeled_data, unlabeled_data,
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
    :return: pseudolabeled_df, labeled_df, withheld_df
    """
    if args.augment_batch > len(unlabeled_data):
        raise ValueError('Not enough unlabeled data to augment with')

    if pseudolabeled_data is None:
        pseudolabeled_data = labeled_data

    if sample is None:
        sample = 1
    print('starting the timer')
    start = perf_time()

    unlabeled_sample = unlabeled_data.iloc[
        np.random.choice(range(len(unlabeled_data)),
                         int(sample * len(unlabeled_data)), replace=False)]

    unlabeled_dataset = to_dataset(unlabeled_sample, shuffle=False, image_size=image_size, batch_size=1,
                                   repeat=False, cache=False)
    labeled_dataset = to_dataset(labeled_data, shuffle=False, image_size=image_size, batch_size=1,
                                 repeat=True, cache=False)
    print(f'sampling {unlabeled_dataset.cardinality()} batches (~{unlabeled_dataset.cardinality()}) '
          f'of {len(unlabeled_data)} available unlabeled examples for NN')
    # compute the distance between every image pair
    distances = []
    # pre-loading examples from disk

    # retrieve top-1 confidence for each prediction on the unlabeled data
    # TODO: make use of this to avoid the second predict step below
    top_1 = np.max(model.predict(unlabeled_dataset), axis=-1)

    start = perf_time()
    count = 0
    # for each unlabeled image
    if args.distance_function != 'confidence':
        for img0, y0 in unlabeled_dataset:
            # TODO: make this faster
            print(f'{count + 1} / {unlabeled_dataset.cardinality()} ({perf_time() - start}s)                          ',
                  end='\r')
            # record the minimum distance between this unlabeled image and all labeled images in 'distances'
            distances.append(
                (
                    count,
                    min([distance_fn(img0[0], img1[0], top_1[count]) for img1, y1 in labeled_dataset])
                )
            )
            count += 1
    else:
        distances = list(enumerate(top_1))
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
    pseudo_labeled_batch_gen = to_dataset(pseudo_labeled_batch, shuffle=False, image_size=image_size,
                                          batch_size=args.batch, repeat=False, cache=False)
    # set the classes as the pseudo-labels
    if hard_labels:
        pseudo_labels = np.argmax(model.predict(pseudo_labeled_batch_gen), axis=1)
    else:
        pseudo_labels = model.predict(pseudo_labeled_batch_gen, steps=len(pseudo_labeled_batch) // args.batch)
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


def blended_dset(train_ds, n_blended=2, prefetch=4, prob=None, std=.1, **kwargs):
    """
    :param train_ds: dataset of training images
    :param batch_size: size of batches to return from the generator
    :param n_blended: number of examples to blend together
    :param image_size: shape of the input image tensor
    :param prefetch: number of examples to pre fetch from disk
    :param prob: probability of repacing a training batch with a convex combination of n_blended
    :param std: standard deviation of (mean 0) gaussian noise to add to images before blending
                (0.0 or equivalently None for no noise)
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    prob = prob if prob is not None else 1.0
    std = float(std) if std is not None else 0.0

    def add_gaussian_noise(x, y, std=1.0):
        return x + tf.random.normal(shape=x.shape, mean=0.0, stddev=std, dtype=tf.float32), y

    # create a dataset from which to get batches to blend
    dataset = train_ds.map(
        lambda x, y: tf.py_function(add_gaussian_noise, inp=[x, y, std], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE).batch(n_blended)

    def random_weighting(n):
        # get a random weighting chosen uniformly from the convex hull of the unit vectors.
        samp = -1 * np.log(np.random.uniform(0, 1, n))
        samp /= np.sum(samp)
        return np.array(samp)

    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
          (convex hull of unit vectors)
        """
        if prob < np.random.uniform(0, 1, 1):
            return x[0], y[0]
        # compute the weights for the combination
        weights = random_weighting(n_blended)
        weights *= float(1 / np.linalg.norm(weights))
        weights = np.array(weights, dtype=np.double).reshape(-1, 1)
        # sum along the 0th dimension weighted by weights
        x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
        y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))[0]
        # return the convex combination
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE).prefetch(prefetch)

    return dataset


def mixup_dset(train_ds, prefetch=4, alpha=None, **kwargs):
    """
    :param train_ds: dataset of batches to train on
    :param prefetch: number of examples to pre fetch from disk
    :param alpha: Dirichlet parameter.  Weights are drawn from Dirichlet(alpha, ..., alpha) for combining two examples.
                    Empirically choose a value in [.1, .4]
    :return: a dataset
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    alpha = alpha if alpha is not None else 1.0

    rng = np.random.default_rng()

    # create a dataset from which to get batches to blend
    dataset = train_ds.batch(2)

    def random_weighting(n):
        return rng.dirichlet([alpha for i in range(n)], 1)

    # the generator yields batches blended together with this weighting
    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
          (convex hull of unit vectors)
        """
        # compute the weights for the combination
        weights = random_weighting(2)
        weights *= float(1 / np.linalg.norm(weights))
        weights = np.array(weights, dtype=np.double).reshape(-1, 1)
        # sum along the 0th dimension weighted by weights
        x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
        y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))[0]
        # return the convex combination
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE).prefetch(prefetch)

    return dataset


def bc_plus(train_ds, prefetch=4, **kwargs):
    """
    :param train_ds: dataset of batches to train on
    :param prefetch: number of examples to pre fetch from disk
    :param alpha: Dirichlet parameter.  Weights are drawn from Dirichlet(alpha, ..., alpha) for combining two examples.
                    Empirically choose a value in [.1, .4]
    :return: a dataset
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    rng = np.random.default_rng()

    # create a dataset from which to get batches to blend
    dataset = train_ds.batch(2)

    def random_weighting(sigma_1, sigma_2):
        p = rng.uniform(0, 1, 1)
        p = 1 / (1 + ((sigma_1 / sigma_2) * ((1 - p) / p)))
        return np.array([p, 1 - p])

    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
          (convex hull of unit vectors)
        """
        # compute the weights for the combination
        weights = random_weighting(tf.sqrt(tf.math.reduce_variance(x[0])), tf.sqrt(tf.math.reduce_variance(x[1])))
        weights *= float(1 / np.linalg.norm(weights))
        weights = np.array(weights, dtype=np.double)
        # sum along the 0th dimension weighted by weights
        x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
        y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))[0]
        # return the convex combination
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE).prefetch(prefetch)

    return dataset


def generalized_bc_plus(train_ds, n_blended=2, prefetch=4, alpha=.25, **kwargs):
    """
    :param train_ds: dataset of batches to train on
    :param n_blended: number of examples to mix
    :param prefetch: number of examples to pre fetch from disk
    :param alpha: Dirichlet parameter.  Weights are drawn from Dirichlet(alpha, ..., alpha) for combining two examples.
                    Empirically choose a value in [.1, .4]
    :return: a dataset
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    alpha = alpha if alpha is not None else 1.0

    rng = np.random.default_rng()

    def add_gaussian_noise(x, y, std=0.01):
        return x + tf.random.normal(shape=x.shape, mean=0.0, stddev=std, dtype=tf.float32), y

    # create a dataset from which to get batches to blend
    # add gaussian noise to each tensor
    dataset = train_ds.map(
        lambda x, y: tf.py_function(add_gaussian_noise, inp=[x, y, .05], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE).batch(n_blended)

    def random_weighting(n):
        return rng.dirichlet(tuple([alpha for i in range(n)]), 1)

    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
          (convex hull of unit vectors)
        """
        # compute the weights for the combination
        weights = random_weighting(n_blended)
        weights *= float(1 / np.linalg.norm(weights))
        weights = np.array(weights, dtype=np.double).reshape(-1, 1)
        # sum along the 0th dimension weighted by weights
        x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
        y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))[0]
        # return the convex combination
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE).prefetch(prefetch)

    return dataset
