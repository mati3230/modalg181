'''
Created on 16.01.2018

@author: mati3230
'''
import numpy as np
import decision_tree_model
import pandas as pd

def is_age_over(D):
    if np.ndim(D) == 2:
        num_rows=np.size(D, 0)
        b = np.zeros(num_rows)
        for i in range(0, num_rows):
            b[i] = D[i,5] > 20 # type in an age
    else:
        b = np.zeros(1)
        b[0] = D[5] > 20 # type in an age
    return b

def has_survived(D):
    if np.ndim(D) == 2:
        num_rows=np.size(D, 0)
        b = np.zeros(num_rows)
        for i in range(0, num_rows):
            b[i] = D[i,1] == 1
    else:
        b = np.zeros(1)
        b[0] = D[1] == 1
    return b
        
def is_class(D):
    class_nr = 2
    if np.ndim(D) == 2:
        num_rows=np.size(D, 0)
        b = np.zeros(num_rows)
        for i in range(0, num_rows):
            b[i] = D[i,2] == class_nr
    else:
        b = np.zeros(1)
        b[0] = D[2] == class_nr
    return b

def is_female(D):
    if np.ndim(D) == 2:
        num_rows=np.size(D, 0)
        b = np.zeros(num_rows)
        for i in range(0, num_rows):
            b[i] = D[i,4] == 'female'
    else:
        b = np.zeros(1)
        b[0] = D[4] == 'female' 
    return b

def has_siblings(D):
    if np.ndim(D) == 2:
        num_rows=np.size(D, 0)
        b = np.zeros(num_rows)
        for i in range(0, num_rows):
            b[i] = D[i,6] > 1
    else:
        b = np.zeros(1)
        b[0] = D[6] > 1
    return b

def is_fare_greater_then(D):
    if np.ndim(D) == 2:
        num_rows=np.size(D, 0)
        b = np.zeros(num_rows)
        for i in range(0, num_rows):
            b[i] = D[i,9] > 3
    else:
        b = np.zeros(1)
        b[0] = D[9] > 3
    return b

# load the dataset
dataset_path="../../datasets/titanic"
data_frame = pd.read_csv("{0}/data.csv".format(dataset_path), delimiter=",")
data = data_frame.as_matrix()

# split data into train and test dataset
num_examples = data.shape[0]
print("num examples: {0}".format(num_examples))
train_percentage=0.8
assert train_percentage < 1
num_train = int(train_percentage * num_examples)
train_data = data[:num_train, :]
test_data = data[num_train:, :]
print("num train data: {0}, num test data: {1}".format(num_train, num_examples - num_train))

# target column - which feature should be classified
t_col = 1

decision_tree = decision_tree_model.decision_tree_model()
# build the tree
decision_tree.train(D=train_data, questions=[is_age_over, is_class, is_female, has_siblings, is_fare_greater_then], t_col=t_col)

accuracy = decision_tree.test_tree(D=train_data, t_col=t_col)
print("train accuracy: {0}".format(accuracy))

accuracy = decision_tree.test_tree(D=test_data, t_col=t_col)
print("test accuracy: {0}".format(accuracy))