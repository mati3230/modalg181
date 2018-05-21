#!/usr/bin/env python3
"""
Example of using a trained model to predict a classification for a unknown dataset.

Created on 08.04.2018
@author: Raphael Stascheit
"""

import pandas as pd  # read *.csv files as dataframe
import estimator  # import estimator to create the same classifier as used for the training
import load_data  # get column/row-descriptions

# build classifier of the same size as trained and defined in estimator.py
classifier = estimator.build_dnn()


# load unlabeled data
path_unlabeled = '../datasets/' + estimator.NAME_DATASET + '/unlabeled.csv'
predict_features = pd.read_csv(path_unlabeled, header=0)

# Generate prediction from the model
predict_input=lambda: load_data.eval_input_fn(predict_features,
                                             labels=None,
                                             batch_size=estimator.BATCH_SIZE)
prediction = None
# TODO 5)
# TODO call predict function of dnn classifier and assign result to prediction
assert prediction is not None
# get feature names (without label) from saved file
names = load_data.Names(estimator.PATH_DATASET)

# print predicted labels and probabilities
i = 0
for element in prediction:
    print("Prediction number {0}".format(i))
    print(element["probabilities"])
    class_id = element['class_ids'][0]
    probability = element['probabilities'][class_id]
    print('Prediction is:', names.labels[class_id], probability)
    i+=1
