import pandas as pd
import numpy as np
from numpy import genfromtxt

#dataset_dir="../../datasets/szeged-weather"
#filename="weatherHistory.csv"
#train_percentage=0.6
#test_percentage=0.2

def normalize(data, num_features=1):
    mmax = np.amax(a=data, axis=0)
    mmin = np.amin(a=data, axis=0)
    data = data - mmin
    diff = mmax - mmin
    if num_features > 1:
        data = data / diff[None,:]
    else:
        data = data / diff
    return data

def load_data(dataset_dir="../../datasets/szeged-weather", filename="weatherHistory.csv"):
    feature_names=["Formatted Date","Summary","Precip Type","Temperature (C)",
                   "Apparent Temperature (C)","Humidity", "Wind Speed (km/h)",
                   "Wind Bearing (degrees)","Visibility (km)","Loud Cover",
                   "Pressure (millibars)","Daily Summary"]
    
    df = pd.read_csv("{0}/{1}".format(dataset_dir, filename), names=feature_names, low_memory=False)
    # convert to numpy array
    npdata = df.values
    # shuffle rows
    np.random.shuffle(npdata[1:,:])
    X = npdata[1:, 3:6].astype(np.float32)
    n_examples = X.shape[0]
    
    print("num examples: {0}".format(n_examples))
    y = npdata[1:, 10].astype(np.float32)
    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert X.shape[0] == y.shape[0]
    X = normalize(X)
    y = normalize(y)
    print(X.shape, y.shape)
    y = np.reshape(y, (y.shape[0], 1))
    X = np.append(X, y, axis=1)
    #X = np.append(np.ones((n_examples, 1)), X, axis=1)
    # uncomment to print head of examples
    print(X[0:5, :])
    
    return X, n_examples

def split(dataset_dir="../../datasets/szeged-weather", filename="weatherHistory.csv", train_percentage=0.6):
    test_percentage = 1-train_percentage
    X, num_examples = load_data()
    eval_percentage=1-train_percentage-test_percentage
    if eval_percentage <= 0:
        raise ValueError("train_percentage + test_percentage < 1! - train_percentage + test_percentage = ".format(train_percentage + test_percentage))
    num_train = int(train_percentage * num_examples)
    num_test = int(test_percentage * num_examples)
    num_eval = int(eval_percentage * num_examples)
    
    checksum = num_train + num_test + num_eval
    diff = num_examples - checksum
    num_train=num_train + diff
    
    train_data = X[:num_train,:]
    test_data = X[num_train:num_train+num_test, :]
    eval_data = X[num_train+num_test:num_examples, :]
    
    np.savetxt("{0}/train.csv".format(dataset_dir), train_data, delimiter=',')
    np.savetxt("{0}/test.csv".format(dataset_dir), test_data, delimiter=',')
    np.savetxt("{0}/eval.csv".format(dataset_dir), eval_data, delimiter=',')

def load_dataset(path, train_percentage=0.6):
    npdata = genfromtxt(path, delimiter=',')
    
    X = npdata[1:, 3:6].astype(np.float32)
    n_examples = X.shape[0]
    
    num_train = int(train_percentage * n_examples)
    
    print("num examples: {0}".format(n_examples))
    y = npdata[1:, 10].astype(np.float32)
    X = normalize(data=X, num_features=X.shape[1])
    y = normalize(data=y, num_features=1)
    
    X = np.append(np.ones((n_examples, 1)), X, axis=1)
    
    train_data = X[:num_train,:]
    test_data = X[num_train:, :]
    
    train_y = y[:num_train]
    test_y = y[num_train:]
    
    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert X.shape[0] == y.shape[0]
    # uncomment to print head of examples
    #print(X[0:5, :])
    #print(y[0:5])
    return train_data, test_data, train_y, test_y
    
#split()
#print("done")