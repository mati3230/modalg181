'''
Created on 26.04.2018

@author: Marcel
'''
import tensorflow as tf
import cnn
import numpy as np
import time
import math
from datetime import datetime

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("eval_dir", "../../cnn_test", """Directory where to write event logs.""")
tf.flags.DEFINE_string("eval_dataset_dir", "../../datasets/stars_from_google_images.tfrecords", """Directory where .tfrecord for testing is placed.""")
tf.flags.DEFINE_string("checkpoint_dir", "../../cnn_train", """Directory where to read model checkpoints.""")
tf.flags.DEFINE_integer("eval_interval_secs", 60 * 1, """How often to run the eval.""")
tf.flags.DEFINE_integer("num_examples", 100, """Number of examples to run.""")
tf.flags.DEFINE_boolean("run_once", True, """Whether to run eval only once.""")
tf.flags.DEFINE_float("moving_average_decay", 0.9999, """Weights are filtered with exponential moving avarage filter, decay of the filter""")
tf.flags.DEFINE_integer("image_size", 40, """Size of an image""")
tf.flags.DEFINE_integer("batch_size", 1, """Batch size""")
tf.flags.DEFINE_integer("num_classes", 5, """Number of persons you want to recognize""")
use_one_hot = False

def dataset_input_fn():
    filenames = [FLAGS.eval_dataset_dir]
    dataset = tf.data.TFRecordDataset(filenames)
    
    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        if use_one_hot:
            keys_to_features = {
                "image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.string, default_value="")
            } 
        else:
            keys_to_features = {
                "image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64))
            }
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Perform additional preprocessing on the parsed data.
        #image = tf.image.decode_jpeg(parsed["image"])
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(parsed["image"], tf.float32)
        image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size])
        
        if use_one_hot:
            label = tf.decode_raw(parsed["label"], tf.float64)
            label = tf.reshape(label, [FLAGS.num_classes])
        else:
            label = tf.cast(parsed["label"], tf.int64)
            #label = tf.cast(label, tf.float32)
        
        return image, label
    
    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    # TODO apply batch size of 'FLAGS.batch_size'
    # TODO create a one shot iterator

    # TODO assign the result of 'iterator.get_next()' to 'features, labels'
    # The input-function must return a dict wrapping the images.
    x = {"image": features}
    y = labels

    return x, y

def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cnn_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
    
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                print(predictions)
                true_count += np.sum(predictions)
                step += 1
            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
    
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    """Eval CNN for a number of steps."""
    g = tf.Graph()
    #with g.as_default():
    with tf.Session(graph=g) as sess:
        images, labels = sess.run(dataset_input_fn())
        imgs = images["image"]
        
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cnn.inference(imgs)
        
        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
        
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
                     
def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()
    print("done")
    
    
if __name__ == '__main__':
    tf.app.run()