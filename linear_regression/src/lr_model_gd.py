'''
Created on 04.05.2018

@author: Marcel
'''
import tensorflow as tf

def inference(x, batch_size=1, theta=None):
    # TODO compute prediction
    return tf.constant(0)

def loss(y_pred, y_true, batch_size=1):
    # TODO compute error
    return tf.constant(1)

def gradient_descent(x, y_true, theta, batch_size=1, n_features=4, learning_rate=0.001):
    # TODO implement gradient descent
    # TODO ignore batch_size
    y_pred = inference(x=x, batch_size=batch_size, theta=theta)
    # TODO change theta
    return y_pred, theta
    
def train(x, y_true, theta, batch_size=1, n_features=4, learning_rate=0.001):
    y_pred, theta = gradient_descent(x=x, y_true=y_true, theta=theta, batch_size=batch_size, n_features=n_features, learning_rate=learning_rate)
    ls = loss(y_pred=y_pred, y_true=y_true, batch_size=batch_size)
    return theta, ls