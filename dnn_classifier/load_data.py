#!/usr/bin/env python3
"""
Import and prepare data from *.csv files for training.

Created on 08.04.2018
@author: Raphael Stascheit
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf

def normalize(data, num_features=1):
    max = np.amax(a=data, axis=0)
    min = np.amin(a=data, axis=0)
    data = data - min
    diff = max - min
    if num_features > 1:
        data = data / diff[None,:]
    else:
        data = data / diff
    return data

def load_data(path_dataset, train_size=0.8, normalize_data=True):
    """
    Load data from *.csv file, shuffle and split into training/test-sets.
    :param path_dataset: Expected location is '../datasets/FOLDER_NAMED_AFTER_YOUR_DATASET/dataset.csv'
    :return: Loaded data split into (train_features, train_labels), (test_features, test_labels)
    """

    # read *.csv file to a dataframe
    dataframe = pd.read_csv(path_dataset, header=0)
    print(dataframe.columns)
    # get list of all existing labels
    labels = dataframe.Label.unique()

    # save names as numpy array if doesn't exist yet
    path_dir = os.path.dirname(path_dataset)
    if not os.path.exists(path_dir + '/names_labels.npy'):
        np.save(path_dir + '/names_labels.npy', labels)
    if not os.path.exists(path_dir + '/names_features.npy'):
        np.save(path_dir + '/names_features.npy', dataframe.columns.values.tolist())

    # tensorflow needs integers as label
    labeldict = dict(zip(labels, range(0, labels.__len__())))
    dataframe = dataframe.replace(labeldict)

    # shuffle data
    np.random.seed(3)  # seed random function for consistent results
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))

    # split data in train_size*100% trainset and (1-train_size)*100% testset
    setsplit_length = int(dataframe.__len__() * train_size)
    trainset = dataframe[0:setsplit_length]
    testset = dataframe[setsplit_length:]

    # split data to features and labels
    train_features, train_labels = trainset, trainset.pop('Label')
    
    test_features, test_labels = testset, testset.pop('Label')
    
    if normalize_data:
        # normalize data
        X_train = train_features.values
        num_features=X_train.shape[1]
        X_train = normalize(data=X_train, num_features=num_features)
        train_features=pd.DataFrame(data=X_train, columns=train_features.columns)
        X_test = test_features.values
        X_test = normalize(data=X_test, num_features=num_features)
        test_features=pd.DataFrame(data=X_test, columns=test_features.columns)

    return (train_features, train_labels), (test_features, test_labels)


class Names:
    """
    Reads column descriptions and label names from saved numpy files.
    Needs estimator.py to be run before.
    """
    def __init__(self, path_dataset):
        self.path_dir = os.path.dirname(path_dataset)
        try:
            self.labels = np.load(self.path_dir + '/names_labels.npy')
            self.features = np.load(self.path_dir + '/names_features.npy')
        except Exception as ex:
            print("Error:\n" + str(ex) + "\nNo saved model found! Please first build a model using 'estimator.py'")
            exit()


def train_input_fn(features, labels, batch_size):
    """
    Input function for training. Will be repeatedly called while training.
    :param features: train_features as returned by load_data.load_data
    :param labels: test_labels as returned by load_data.load_data
    :param batch_size:
    :return: tf.data.Dataset
    """
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000, seed=3).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """
    An input function for evaluation or prediction
    :param features: test_features as returned by load_data.load_data
    :param labels: test_labels as returned by load_data.load_data
    :param batch_size:
    :return: tf.data.Dataset
    """
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (dict(features), labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
