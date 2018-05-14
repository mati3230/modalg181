import tensorflow as tf
import numpy as np
import split_dataset
import lr_model_gd

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 0.1, """Learning Rate of Gradient Descent""")
tf.app.flags.DEFINE_string("train_dataset_dir", "../../datasets/szeged-weather/weatherHistory.csv", """Directory where train.csv for training is placed.""")
tf.app.flags.DEFINE_integer("batch_size", 1, """Batch size""")
tf.app.flags.DEFINE_integer("buffer_size", 10000, """Number of elements from a dataset from which a new dataset will sample""")
tf.app.flags.DEFINE_float("early_stop", 0.0005, """Training will stop if error is below this value""")
tf.app.flags.DEFINE_bool("use_test_batch", False, """If true, error computation at the end is just one step""")
tf.app.flags.DEFINE_float("train_percentage", 0.6, """How many percent of data will be used for training?""")

def train():
    X_train, X_test, y_train, y_test = split_dataset.load_dataset(path=FLAGS.train_dataset_dir, train_percentage=FLAGS.train_percentage)
    print("X_train: {0}, y_train: {1}, X_test: {2}, y_test: {3}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    
    n_features = X_train.shape[1]
    print("num features: {0}".format(n_features))

    theta = tf.Variable(tf.random_normal([n_features], stddev=1), name="theta")
    init_op = tf.global_variables_initializer()
    
    num_steps = int(X_train.shape[0] / FLAGS.batch_size)
    print("num steps: {0}".format(num_steps))
    
    print("Learning rate: {0}".format(FLAGS.learning_rate))
    print("batch size: {0}".format(FLAGS.batch_size))
    
    with tf.Session() as sess:
        sess.run(init_op)
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.shuffle(buffer_size=FLAGS.buffer_size)
        dataset = dataset.batch(batch_size=FLAGS.batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_element = sess.run(iterator.get_next())
        x_tensor = next_element[0]
        x_tensor = tf.cast(x_tensor, np.float32)
        y_tensor = next_element[1]
        y_tensor = tf.cast(y_tensor, np.float32)
        for i in range(num_steps):
            theta, loss = lr_model_gd.train(x=x_tensor, y_true=y_tensor, theta=theta, batch_size=FLAGS.batch_size, n_features=n_features, learning_rate=FLAGS.learning_rate)
            t, l = sess.run([theta, loss])
            print("loss: {0}, step: {1}".format(l, i))
            if l < FLAGS.early_stop:
                break
        print("training finished")
        
        sess.run(tf.Print(theta, [theta], "theta"))
            
        if FLAGS.use_test_batch:
            test_batch_size = X_test.shape[0]
        else:
            test_batch_size=1
            num_steps = X_test.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        dataset = dataset.shuffle(buffer_size=FLAGS.buffer_size)
        dataset = dataset.batch(batch_size=test_batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_element = sess.run(iterator.get_next())
        x_tensor = next_element[0]
        x_tensor = tf.cast(x_tensor, np.float32)
        y_tensor = next_element[1]
        y_tensor = tf.cast(y_tensor, np.float32)
        if FLAGS.use_test_batch:
            y_pred = lr_model_gd.inference(x=x_tensor, batch_size=test_batch_size, theta=theta)
            loss = lr_model_gd.loss(y_pred=y_pred, y_true=y_tensor, batch_size=test_batch_size)
            l = sess.run(loss)
            print("total loss: {0}".format(l))
        else:
            l=0
            for i in range(num_steps):
                y_pred = lr_model_gd.inference(x=x_tensor, batch_size=test_batch_size, theta=theta)
                loss = lr_model_gd.loss(y_pred=y_pred, y_true=y_tensor, batch_size=test_batch_size)
                l += sess.run(loss)/num_steps
                print("loss: {0}, progress: {1}%".format(l, (i/num_steps)*100))
            print("total loss: {0}".format(l))
            
def main(argv=None): 
    train()
    print("done")
    
if __name__ == "__main__":
    tf.app.run()
        