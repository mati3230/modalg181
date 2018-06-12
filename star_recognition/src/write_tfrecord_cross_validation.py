'''
Created on 05.06.2018

@author: Marcel
'''
import os
import tensorflow as tf
from utils import load_image, create_dir, resize_image
from random import shuffle
import matplotlib.pyplot as plt

def _load_image(addr, image_size=40):
    img = load_image(addr)
    img = resize_image(img, image_size=image_size)
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _write_tfrecords(files, min_idx, max_idx, tfrecords_filename, image_size=40):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for i in range(min_idx, max_idx):
        label = files[i][0]
        img_filename = files[i][1]
        img = _load_image(addr=img_filename, image_size=image_size)

        feature = {
            "image": _bytes_feature(tf.compat.as_bytes(img.tostring())),
            "label": _int64_feature(label)
        }
            
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()

def _hist(files, min_idx, max_idx, name):
    labels = []
    for i in range(min_idx, max_idx):
        label = files[i][0]
        labels.append(label)
    
    plt.hist(labels)
    plt.title("{} label histogram".format(name))
    plt.show()

dataset_path = "../../datasets/stars_from_google_images"
create_dir(path=dataset_path)
stars = ["nicolas_cage", "brad_pitt", "angelina_jolie", "leonardo_dicaprio", "robert_downey_jr"]

image_size = 140
train_percentage = 0.6
test_percentage = 0.2
if train_percentage + test_percentage >= 1.0:
    raise Exception("train_percentage + test_percentage have to be < 1.0")
num_examples = 0
files = []

for i in range(len(stars)):
    label = i
    star_path = "{0}/{1}/faces".format(dataset_path, stars[i])
    filenames = os.listdir(star_path)
    for j in range(len(filenames)):
        files.append((label, "{}/{}".format(star_path, filenames[j])))
    num_examples += len(filenames)

if len(files) != num_examples:
    raise Exception("Error: More files ({}) than counted examples ({})".format(len(files), num_examples))

# randomize the dataset
shuffle(files)

train_max_idx = int(train_percentage * num_examples)
test_max_idx = train_max_idx + int(test_percentage * num_examples)
validate_max_idx = int((0.2 + 0.6) * num_examples)

print("process train set")
_write_tfrecords(files=files, min_idx=0, max_idx=train_max_idx, 
            tfrecords_filename="../../datasets/stars_train.tfrecords", 
            image_size=image_size)
print("train set saved")
_hist(files=files, min_idx=0, max_idx=train_max_idx, name="train")
print("process test set")
_write_tfrecords(files=files, min_idx=train_max_idx, max_idx=test_max_idx, 
            tfrecords_filename="../../datasets/stars_test.tfrecords", 
            image_size=image_size)
print("test set saved")
_hist(files=files, min_idx=train_max_idx, max_idx=test_max_idx, name="test")
print("process validation set")
_write_tfrecords(files=files, min_idx=test_max_idx, max_idx=len(files), 
            tfrecords_filename="../../datasets/stars_validation.tfrecords", 
            image_size=image_size)
print("validation set saved")
_hist(files=files, min_idx=test_max_idx, max_idx=len(files), name="validation")

print("done")