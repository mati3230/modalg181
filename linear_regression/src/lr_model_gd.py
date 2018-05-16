'''
Created on 04.05.2018

@author: Marcel
'''
import tensorflow as tf

def tile_theta(theta, batch_size=2):
    bs = tf.constant([batch_size])
    return tf.reshape(tf.tile(theta, bs), [ bs[0], tf.shape(theta)[0]])

def inference(x, batch_size=1, theta=None):
    # TODO compute prediction
    if batch_size > 1:
        theta = tile_theta(theta=theta, batch_size=batch_size)
    return tf.reduce_sum( tf.multiply( x, theta ), 1, keepdims=True )

def loss(y_pred, y_true, batch_size=1):
    # TODO compute error
    y_pred = tf.reshape(y_pred, [batch_size, 1])
    y_true = tf.reshape(y_true, [batch_size, 1])
    return tf.losses.mean_squared_error(y_true, y_pred)

def gradient_descent(x, y_true, theta, batch_size=1, n_features=4, learning_rate=0.001):
    # TODO implement gradient descent
    # TODO ignore batch_size
    y_pred = inference(x=x, batch_size=batch_size, theta=theta)
    # print(y_pred.get_shape())
    # TODO change theta
    y_true = tf.scalar_mul(-1.0, y_true)
    y_true = tf.reshape(y_true, (batch_size, 1))
    residual = tf.add(y_pred, y_true)  
    residual = tf.scalar_mul(learning_rate/batch_size, residual)
    if batch_size > 1:
        x_transpose = tf.transpose(x)
        residual = tf.matmul(x_transpose, residual)
    else:
        residual = tf.multiply(x, residual)
    residual = tf.reshape(residual, [n_features])
    theta = tf.subtract(theta, residual)
    theta = tf.reshape(theta, [n_features])
    return y_pred, theta
    
def train(x, y_true, theta, batch_size=1, n_features=4, learning_rate=0.001):
    y_pred, theta = gradient_descent(x=x, y_true=y_true, theta=theta, batch_size=batch_size, n_features=n_features, learning_rate=learning_rate)
    ls = loss(y_pred=y_pred, y_true=y_true, batch_size=batch_size)
    return theta, ls