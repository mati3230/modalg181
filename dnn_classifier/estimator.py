#!/usr/bin/env python3
"""
Deep Neural Network (DNN) Classifier for the Iris dataset.

Created on 08.04.2018
@author: Raphael Stascheit
"""

import tensorflow as tf  # neural network
import os  # operating system
import shutil  # shell utility
import datetime  # needed for printable timestamp
import load_data  # local *.py file for loading datasets
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameter
BATCH_SIZE = 30
TRAIN_STEPS = 1000
HIDDEN_LAYER = [3]
TRAIN_SIZE=0.6
NORMALIZE=False

# Flags
FLAG_RECOMPUTE = True  # True: Retrain (former models will be deleted!) False: If former model exists continue training
FLAG_VERBOSE = True  # print detailed information to console when training
FLAG_PLOT = True # plot histogram to see class distribution in train and test set

# Chose dataset (Must be the name of the according subfolder of '../datasets/')
NAME_DATASET = 'iris_flower'
# NAME_DATASET = 'Magic_Gamma_Telescope'


# Create dataset path from name (Do NOT change!)
PATH_DATASET = '../datasets/' + NAME_DATASET + '/dataset.csv'


def path_model():
    """
    Concats string for the model-directory. Files are named after the shape of the DNN
    :return: Model directory
    """
    path = './logs/' + NAME_DATASET
    for index, neurons in enumerate(HIDDEN_LAYER):
        path += '_' + str(neurons)

    return path


def save_results(result):
    """Saves results to a *.txt file in order to be able to compare different hyperparameter-settings"""
    # check if result file already exists
    flag_exists = os.path.exists("./logs/results.txt")
    with open("./logs/results.txt", "a+") as f:
        # if file doesn't exist yet, write a description to the first line
        if not flag_exists:
            f.write("Date\t\t\t\tDataset\t\t\t\t\tHiddenLayer\t\tBatchSize\tSteps\tTrainSize\tNormalize\tGlobalSteps\tAccuracy\tAverageLoss\n")

        # append a line with timestamp, hyperparameters and results
        f.write(str(datetime.datetime.now().strftime("%d %m %Y %H:%M")) + "\t" +
                NAME_DATASET + "\t" +
                str(HIDDEN_LAYER) + "\t" +
                str(BATCH_SIZE) + "\t\t\t" +
                str(TRAIN_STEPS) + "\t" +
                str(TRAIN_SIZE) + "\t" +
                str(NORMALIZE) + "\t" +
                str(result["global_step"]) + "\t\t" +
                '{:0.8f}'.format(result["accuracy"]) + "\t" +
                '{:0.8f}'.format(result["loss"]) + "\n")


def build_dnn(hidden_layer=HIDDEN_LAYER):
    """
    Builds a Tensorflow DNN-Classifier.
    :param hidden_layer: Default size of the DNN is the global array HIDDEN_LAYER.
    If called with arguments (e.g. when imported) you can use other values
    :return classifier: Tensorflow Classifier Object
    """
    names = load_data.Names(PATH_DATASET)

    # Feature columns describe how to use the input
    feature_columns = []
    for key in names.features[:-1]:  # ignore label column
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Print information about the current settings
    for index, neurons in enumerate(HIDDEN_LAYER):
        print('Hidden Layer', str(index) + ':\t', neurons, "Neurons")
    print('Features:', str(names.features))
    print('Labels:', names.labels)

    num_classes=names.labels.__len__()
    
    # TODO 1)
    # TODO assign DNNClassifier to classifier (hint: have a look at the documentation, tensorflow version 1.5)
    # TODO specify feature_columns of DNNClassifier
    # TODO specify hidden_units of DNNClassifier
    # TODO specify n_classes of DNNClassifier
    # TODO assign model_dir parameter of DNNClassifier with path_model() in order to save the trained network as a model file to ./logs/ folder;
    # TODO try to raise the performance with different hyperparameter (see above)
    classifier = None
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=hidden_layer,
        # Activation function, relu means rectified linear unit
        activation_fn=tf.nn.relu,
        # Number of different classifications (--> labels)
        n_classes=names.labels.__len__(),
        # Save the trained network as a model file to the ./logs/ folder;
        model_dir=path_model()
    )

    return classifier


def main(argv):
    """Main function to be run by tensorflow"""
    # Fetch the data
    (train_features, train_labels), (test_features, test_labels) = load_data.load_data(PATH_DATASET, TRAIN_SIZE, NORMALIZE)
    # Print head of data
    print("Head of training data:")
    print(train_features.values[:5,:])
    print("Train labels:")
    print(train_labels.values[:5])
    
    print("Head of test data:")
    print(test_features.values[:5,:])
    print("Test labels:")
    print(test_labels.values[:5])
    
    num_classes=np.amax(a=train_labels.values, axis=0)+1
    
    print("Number of classes: {0}".format(num_classes))
    print("Number of features: {0}".format(train_features.values.shape[1]))
    print("Number of training examples: {0}".format(train_labels.values.shape[0]))
    print("Number of test examples: {0}".format(test_labels.values.shape[0]))
    
    # TODO set FLAG_PLOT to True or False to toggle visualization
    if FLAG_PLOT:
        plt.hist(train_labels)
        plt.title("train label histogram")
        plt.show()
        
        plt.hist(test_labels)
        plt.title("test label histogram")
        plt.show()
    
    if FLAG_RECOMPUTE:
        # delete former log files
        shutil.rmtree(path_model(), ignore_errors=True)

    classifier = build_dnn()
    assert classifier is not None
    
    train_input_fn=lambda: load_data.train_input_fn(train_features, train_labels, BATCH_SIZE)
    # Train the Model
    # TODO 2)
    # TODO call the training function of the dnn classifier
    # TODO specify input_fn parameter of train function
    # TODO assign TRAIN_STEPS to steps parameter
    classifier.train(
        input_fn=train_input_fn,
        steps=TRAIN_STEPS)

    train_eval_input_fn=lambda: load_data.eval_input_fn(train_features, train_labels, BATCH_SIZE)
    # Evaluate training performance of the model
    # TODO 3)
    # TODO call the evaluate function of the dnn classifier with the train set and assign result to train_result
    train_result = None
    train_result = classifier.evaluate(input_fn=train_eval_input_fn)
    assert train_result is not None
    # print and save results
    print('\nTrain accuracy:', train_result["accuracy"])
    
    test_eval_input_fn=lambda: load_data.eval_input_fn(test_features, test_labels, BATCH_SIZE)
    # Evaluate test performance of the model
    # TODO 4)
    # TODO call the evaluate function of the dnn classifier with the test set
    eval_result = None
    eval_result = classifier.evaluate(input_fn=test_eval_input_fn)
    assert eval_result is not None
    # print and save results
    print('\nTest set accuracy:', eval_result["accuracy"])
    save_results(eval_result)

if __name__ == '__main__':
    """
    This will only be executed if this file is the main file.
    It will take no effect if when it's imported.
    """
    # if verbose mode, set verbosity to tensorflow
    if FLAG_VERBOSE:
        tf.logging.set_verbosity(tf.logging.INFO)

    # run tensorflow with 'main' as main function
    tf.app.run(main)
