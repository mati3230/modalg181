'''
Created on 23.04.2018

@author: Marcel
'''
import tensorflow as tf
import cnn
from datetime import datetime
import time
import math

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("train_dir", "../../cnn_train", """Directory where to write event logs and checkpoint.""")
tf.flags.DEFINE_string("train_dataset_dir", "../../datasets/stars_train.tfrecords", """Directory where .tfrecord for training is placed.""")
tf.flags.DEFINE_integer("max_steps", 100000, """Number of batches to run.""")
tf.flags.DEFINE_boolean("log_device_placement", False, """Whether to log device placement.""")
tf.flags.DEFINE_integer("log_frequency", 100, """How often to log results to the console.""")
tf.flags.DEFINE_integer("buffer_size", 10000, """Number of elements from a dataset from which a new dataset will sample""")
tf.flags.DEFINE_integer("batch_size", 16, """Batch size""")
tf.flags.DEFINE_integer("repeat_dataset", 10000, """How many times should a dataset be repeated""")
tf.flags.DEFINE_integer("image_size", 140, """Size of an image""")
tf.flags.DEFINE_integer("num_classes", 5, """Number of persons you want to recognize""")
tf.flags.DEFINE_float("moving_average_decay", 0.9999, """Weights are filtered with exponential moving avarage filter, decay of the filter""")
tf.flags.DEFINE_float("learning_rate_decay_factor", 0.96, """decay factor which decreases learning rate""")
tf.flags.DEFINE_float("initial_learning_rate", 0.1, """Learning rate at the beginning of the training""")
tf.flags.DEFINE_float("num_epochs_per_decay", 20, """How many epochs should pass to apply a learning rate decay step""")
tf.flags.DEFINE_integer("num_examples_per_epoch_for_train", 512, """How many examples are used in one epoch for training""")
tf.flags.DEFINE_integer("save_checkpoint_secs", 240, """save the model every x seconds""")
tf.flags.DEFINE_float("momentum", 0.5, """Momentum factor""")

def dataset_input_fn():
    filenames = [FLAGS.train_dataset_dir]
    dataset = tf.data.TFRecordDataset(filenames)
    
    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Perform additional preprocessing on the parsed data.
        #image = tf.image.decode_jpeg(parsed["image"])
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, 3])

        # data augmentation
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_contrast(image, lower=1, upper=4)
        image = tf.image.random_brightness(image, max_delta=0.25)
        
        random_angle = tf.random_uniform([1], minval=0, maxval=math.pi, dtype=tf.float32)
        image = tf.contrib.image.rotate(image, angles=random_angle, interpolation="NEAREST")

        label = tf.cast(parsed["label"], tf.int64)
        
        return image, label
    
    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    # TODO shuffle dataset with FLAGS.buffer_size
    dataset = dataset.shuffle(buffer_size=FLAGS.buffer_size)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))
    # TODO repeat dataset with FLAGS.repeat_dataset
    dataset = dataset.repeat(FLAGS.repeat_dataset) # num epochs
    # TODO create a one shot iterator
    iterator = dataset.make_one_shot_iterator()
    
    # TODO assign the result of 'iterator.get_next()' to 'features, labels'
    features, labels = iterator.get_next()
    
    # apply grayscale to the batch
    features = tf.image.rgb_to_grayscale(features, name="grayscale_conversion")
    
    # normalize data
    features = tf.cast(features, dtype=tf.float32)
    features = tf.scalar_mul(1/255, features)
    
    # The input-function must return a dict wrapping the images.
    x = {"image": features}
    y = labels

    return x, y

def train():
    global_step = tf.train.get_or_create_global_step()
    
    # get images and labels for cnn
    # force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down. 
    with tf.device("/cpu:0"):
        images, labels = dataset_input_fn()
        imgs = images["image"]
    # Build a Graph that computes the logits predictions from the
    
    # TODO call 'cnn.inference' with 'imgs' as input argument and assign the result to a variable called 'logits'
    logits = cnn.inference(imgs)
    
    # TODO call 'cnn.loss' with 'logits' and 'labels' as input arguments and assign the result to a variable called 'loss'
    loss = cnn.loss(logits, labels)
    
    # TODO call 'cnn.train' with 'loss' and 'global_step' as input argument and assign the result to a variable called 'train_op'
    train_op = cnn.train(loss, global_step)
    
    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""
        
        def begin(self):
            self._step = -1
            self._start_time = time.time()
        
        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(loss)  # Asks for loss value.
        
        def after_run(self, run_context, run_values):
            if self._step % FLAGS.log_frequency == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time
                
                loss_value = run_values.results
                examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)
                
                format_str = ("%s: step %d, loss = %.2f (%.1f examples/sec; %.3f "
                              "sec/batch)")
                print (format_str % (datetime.now(), self._step, loss_value,
                                 examples_per_sec, sec_per_batch))
    
    with tf.train.MonitoredTrainingSession(save_checkpoint_secs=FLAGS.save_checkpoint_secs,
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
        while not mon_sess.should_stop():
            # TODO run the training session
            mon_sess.run(train_op)
                
def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
    print("done")
    
if __name__ == "__main__":
    tf.app.run()