'''
Created on 19.04.2018

@author: Marcel
'''

import tensorflow as tf
import re
FLAGS = tf.flags.FLAGS
TOWER_NAME = "tower"

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
  """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device("/cpu:0"):
        #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection("losses", weight_decay)
    return var

def inference(images):
    """Build the CNN model.
    Args:
      images: Images returned from inputs().
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    conv1_width = 5
    conv1_height = 5
    conv1_in_channels = 1
    conv1_out_channels = 64
    conv1_stride = 1
    # Tensor input should become 4-D: [Batch Size, Height, Width, Channel]
    images = tf.reshape(images, shape=[FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1])
    # the first and the last stride arg correspond with batch and depth
    with tf.variable_scope("conv1") as scope:
        kernel = _variable_with_weight_decay("weights",
                                           shape=[conv1_width, conv1_height, conv1_in_channels, conv1_out_channels],
                                           stddev=5e-2,
                                           wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, conv1_stride, conv1_stride, 1], padding="SAME")
        biases = _variable_on_cpu("biases", [conv1_out_channels], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)
    
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool1")
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name="norm1")
    
    conv2_width = 5
    conv2_height = 5
    conv2_in_channels = 64
    conv2_out_channels = 64
    conv2_stride = 1
    
    # conv2
    with tf.variable_scope("conv2") as scope:
        kernel = _variable_with_weight_decay("weights",
                                             shape=[conv2_width, conv2_height, conv2_in_channels, conv2_out_channels],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, conv2_stride, conv2_stride, 1], padding="SAME")
        biases = _variable_on_cpu("biases", [conv2_out_channels], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)
    
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name="norm2")
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding="SAME", name="pool2")
    
    num_fc3_neurons = 384
    
    # local3 - fully connected
    with tf.variable_scope("local3") as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay("weights", shape=[dim, num_fc3_neurons],
                                      stddev=0.04, wd=0.004)
        biases = _variable_on_cpu("biases", [num_fc3_neurons], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)
    
    num_fc4_neurons = 192
    
    # local4 - fully connected
    with tf.variable_scope("local4") as scope:
        weights = _variable_with_weight_decay("weights", shape=[num_fc3_neurons, num_fc4_neurons],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu("biases", [num_fc4_neurons], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)
    
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope("softmax_linear") as scope:
        # ****************** Cast stddev necessary? **************************************
        weights = _variable_with_weight_decay("weights", [num_fc4_neurons, FLAGS.num_classes],
                                              stddev=1/num_fc4_neurons, wd=None)
        biases = _variable_on_cpu("biases", [FLAGS.num_classes],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
    
    return softmax_linear

def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    tf.add_to_collection("losses", cross_entropy_mean)
    
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection("losses"), name="total_loss")

def _add_loss_summaries(total_loss):
    """Add summaries for losses in CNN model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
    losses = tf.get_collection("losses")
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + " (raw)", l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    
    return loss_averages_op

def train(total_loss, global_step):
    """Train CNN model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = FLAGS.num_examples_per_epoch_for_train / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
    tf.summary.scalar("learning_rate", lr)
    
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradients", grad)
    
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    return variables_averages_op