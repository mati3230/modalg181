'''
Created on 04.06.2018

@author: Marcel
'''
import cv2
import tensorflow as tf
import cnn
import numpy as np
from datetime import datetime
from utils import load_image, preprocess_image, extract_faces, plot_image
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", "../../cnn_train", """Directory where to read model checkpoints.""")
tf.flags.DEFINE_float("moving_average_decay", 0.9999, """asdasd""")
tf.flags.DEFINE_integer("image_size", 40, """Size of an image""")
tf.flags.DEFINE_integer("batch_size", 1, """Batch size""")
tf.flags.DEFINE_integer("num_classes", 5, """Number of persons you want to recognize""")
tf.flags.DEFINE_string("img_path", "../../datasets/stars_from_google_images/brad_pitt_test1.jpg", """Directory where to read model checkpoints.""")
tf.flags.DEFINE_integer("img_label", 1, """Label of the image""")

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def get_image(filename, label):
    img=load_image(filename)
    #"""
    if img is None: 
        print("image is None")
        raise SystemExit(-1)
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    haar_face_cascade = cv2.CascadeClassifier()
    result = haar_face_cascade.load("{0}/../haarcascade_frontalface_alt.xml".format(dir_path))
    #result = haar_face_cascade.load("../haarcascade_frontalface_alt.xml")
    if not result:
        print("Error loading haarcascade")
        raise SystemExit(-1)
    
    face_imgs = extract_faces(img, haar_face_cascade=haar_face_cascade)
    if len(face_imgs)==0:
        print("No faces found")
        raise SystemExit(-1)
    img = face_imgs[0]
    resized_img = cv2.resize(img, (FLAGS.image_size, FLAGS.image_size), interpolation=cv2.INTER_CUBIC)
    print(resized_img.shape)
    plot_image(resized_img)
    plot_image(img)
    #"""
    img = preprocess_image(img, image_size=FLAGS.image_size)
    label = np.array([label])
    #img = cv2.imread(filename)
    return img, label

def predict():
    #labels = [FLAGS.img_label]
    #filenames = [FLAGS.img_path]
    
    img, lbl = get_image(FLAGS.img_path, FLAGS.img_label)

    img = tf.convert_to_tensor(img, dtype=tf.float32)
    label = tf.convert_to_tensor(lbl, dtype=tf.int32)
    
    logit = cnn.inference(img)
    
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logit, label, 1)
    
    with tf.Session() as sess:
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        
        saver = tf.train.Saver(variables_to_restore)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        
        true_count = 0
        predictions = sess.run([top_k_op])
        #print(predictions)
        true_count += np.sum(predictions)
        # Compute precision @ 1.
        precision = true_count
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        
def main(argv=None):  # pylint: disable=unused-argument
    predict()
    print("done")
    
if __name__ == '__main__':
    tf.app.run()