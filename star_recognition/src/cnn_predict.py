'''
Created on 04.06.2018

@author: Marcel
'''
import tensorflow as tf
import cnn
import numpy as np
from datetime import datetime

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", "../../cnn_train", """Directory where to read model checkpoints.""")
tf.flags.DEFINE_float("moving_average_decay", 0.9999, """Weights are filtered with exponential moving avarage filter, decay of the filter""")
tf.flags.DEFINE_integer("image_size", 40, """Size of an image""")
tf.flags.DEFINE_integer("batch_size", 1, """Batch size""")
tf.flags.DEFINE_integer("num_classes", 5, """Number of persons you want to recognize""")

def predict():
    filename_queue = tf.train.string_input_producer(["../../datasets/stars_from_google_images/brad_pitt_test.jpeg"]) #  list of files to read
    labels = [1]
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    
    img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.
    
    # preprocessing
    img = tf.image.resize_images(img, [FLAGS.image_size, FLAGS.image_size], method=tf.image.ResizeMethod.BICUBIC)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.scalar_mul(2.0, img)
    img = img - tf.constant([1.0])        
    logits = cnn.inference(img)
    
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            
            true_count = 0
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
            # Compute precision @ 1.
            precision = true_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
    
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        
def main(argv=None):  # pylint: disable=unused-argument
    predict()
    print("done")
    
if __name__ == '__main__':
    tf.app.run()