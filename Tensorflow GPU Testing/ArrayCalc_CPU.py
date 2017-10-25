import tensorflow as tf
import numpy as np
import datetime

# Pair of numpy arrays.
matrix1 = 10 * np.random.random_sample((3, 4))
matrix2 = 10 * np.random.random_sample((4, 6))

# Create a pair of constant ops, add the numpy
# array matrices.
tf_matrix1 = tf.constant(matrix1)
tf_matrix2 = tf.constant(matrix2)

# Create a matrix multiplication operation, pass
# the TensorFlow matrices as inputs.
tf_product = tf.matmul(tf_matrix1, tf_matrix2)

# Launch a session, use default graph.
sess = tf.Session()

# Invoking run() with tf_product variable will
# execute the ops necessary to satisfy the request,
# storing result in 'result.'
#get start timestamp
ts_a = datetime.datetime.now()

result = sess.run(tf_product)

#get end timestamp
ts_b = datetime.datetime.now()

delta = ts_b - ts_a

print ("Total Runtime Duration was " + repr(int(delta.total_seconds() * 1000)) + " milliseconds.")

# Now let's have a look at the result.
print (result)

# Close the Session when we're done.
sess.close()