import tensorflow as tf

# DNN Classifier assumes ordinal encoding
use_one_hot = False
IMAGE_SIZE = 40

def dataset_input_fn():
    filenames = ["../../datasets/stars_from_google_images.tfrecords"]
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
        image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE])
        
        if use_one_hot:
            label = tf.decode_raw(parsed["label"], tf.float64)
            label = tf.reshape(label, [5])
        else:
            label = tf.cast(parsed["label"], tf.int64)
            #label = tf.cast(label, tf.float32)
        
        return image, label
    
    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    
    # TODO shuffle dataset with buffer_size=10000
    dataset = dataset.shuffle(buffer_size=10000)
    
    # TODO apply batch size of 32
    dataset = dataset.batch(32)
    
    # TODO repeat the dataset 3 times
    dataset = dataset.repeat(3) # num epochs
    
    # TODO create a one shot iterator
    iterator = dataset.make_one_shot_iterator()
    
    # TODO assign the result of 'iterator.get_next()' to 'features, labels'
    features, labels = iterator.get_next()
    
    # The input-function must return a dict wrapping the images.
    x = {"image": features}
    y = labels

    return x, y

feature_columns = [tf.feature_column.numeric_column("image", shape=[IMAGE_SIZE, IMAGE_SIZE])]

hidden_units = [IMAGE_SIZE, IMAGE_SIZE/2]
num_classes = 5

# TODO create a 'tf.estimator.DNNClassifier' (MLP) 
model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                   # TODO assign member 'hidden_units' to variable 'hidden_units'
                                   hidden_units=hidden_units,
                                   # TODO set the 'activation_fn' to ReLu-Activation function
                                   activation_fn=tf.nn.relu,
                                   # TODO assign 'n_classes' to num_classes
                                   n_classes=num_classes,
                                   # TODO set the 'model dir' to '../../star_recognition_checkpoint/'
                                   model_dir="../../star_recognition_checkpoint/")

print("start training")
model.train(input_fn=dataset_input_fn, steps=2)

print("start evaluation")
result = model.evaluate(input_fn=dataset_input_fn)

print(result)
