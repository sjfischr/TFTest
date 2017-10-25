# A simple Hello World! using TensorFlow
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
sess.run(hello)

# -> Hello, TensorFlow!
