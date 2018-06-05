'''
Created on 13.04.2018
TODO extend script to split data into training test and validation set
@author: Marcel
'''
import os
from utils import create_dir
import tensorflow as tf
import numpy as np
from utils import preprocess_image, load_image

IMAGE_SIZE = 40

def _load_image(addr):
    img = load_image(addr)
    img = preprocess_image(img, image_size=IMAGE_SIZE)
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

dataset_path = "../../datasets/stars_from_google_images"
create_dir(path=dataset_path)
stars = ["nicolas_cage", "brad_pitt", "angelina_jolie", "leonardo_dicaprio", "robert_downey_jr"]

use_one_hot = False

tfrecords_filename = "../../datasets/stars_from_google_images.tfrecords"
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for i in range(len(stars)):
    print("process star: {0}".format(stars[i]))
    star_path = "{0}/{1}/faces".format(dataset_path, stars[i])
    filenames = os.listdir(star_path)
    for filename in filenames:
        img = _load_image(addr="{0}/{1}".format(star_path, filename))
        
        if use_one_hot:
            lbl = np.zeros((1,5))
            lbl[np.arange(1), i] = 1
            #print(lbl[0].tostring())
            #print(lbl[0].dtype)
            feature = {
                "image": _bytes_feature(tf.compat.as_bytes(img.tostring())),
                #"label": _int64_feature(label)
                #"label": _floats_feature(lbl[0])
                "label": _bytes_feature(tf.compat.as_bytes(lbl[0].tostring()))
            }
        else:
            feature = {
                "image": _bytes_feature(tf.compat.as_bytes(img.tostring())),
                "label": _int64_feature(i)
            }
            
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

writer.close()

print("done")
        