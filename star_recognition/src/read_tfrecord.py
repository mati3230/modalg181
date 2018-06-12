import tensorflow as tf
import numpy as np
import cv2
import math

use_one_hot = False
IMAGE_SIZE = 140

def dataset_input_fn():
    filenames = ["../../datasets/stars_train.tfrecords"]
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
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        
        #image = tf.image.adjust_saturation(image, saturation_factor = 1.3)
        #image = tf.image.adjust_hue(image, 0.05)
        #image = tf.image.adjust_contrast(image, contrast_factor=4)
        #image = tf.image.adjust_brightness(image, delta=0.25)
        
        #random_angle = tf.random_uniform([1], minval=0, maxval=math.pi, dtype=tf.float32)
        #image = tf.contrib.image.rotate(image, angles=random_angle, interpolation="NEAREST")
        
        if use_one_hot:
            label = tf.decode_raw(parsed["label"], tf.float64)
            label = tf.reshape(label, [5])
        else:
            label = tf.cast(parsed["label"], tf.int64)
            label = tf.cast(label, tf.float32)
        
        return image, label
    
    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    # TODO shuffle dataset with buffer_size=10000
    dataset = dataset.shuffle(buffer_size=10000)
    # TODO apply batch size of 32
    dataset = dataset.batch(32)
    # TODO create a one shot iterator
    iterator = dataset.make_one_shot_iterator()
    
    # TODO assign the result of 'iterator.get_next()' to 'features, labels'
    features, labels = iterator.get_next()
    
    features = tf.image.rgb_to_grayscale(features, name="grayscale_conversion")

    return features, labels

with tf.Session() as sess:
    images, labels = dataset_input_fn()
    
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(5): # iterate through 5 batches with 32 examples
        img, lbl = sess.run([images, labels])
        print("Shape of batch: {0}".format(img.shape))
        #print("Normalized image value of image 0 at pixel (20,20): {0}".format(img.item(0, 20, 20)))
        img = img.astype(np.uint8)
        if use_one_hot:
            lbl = lbl.astype(np.float64)
        else:
            lbl = lbl.astype(np.float32)
        print("Label: {0}".format(lbl[0]))
        #print(img[0].shape)
        cv2.imshow("image", img[0])
        cv2.resizeWindow("image", IMAGE_SIZE, IMAGE_SIZE)  
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()