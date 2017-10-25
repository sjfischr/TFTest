import datetime
import platform
import subprocess

import numpy as np
import tensorflow as tf


def calcArray():
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

    # Launch a GPU session, use default graph.
    with tf.Session() as sessgpu:
        # Invoking run() with tf_product variable will
        # execute the ops necessary to satisfy the request,
        # storing result in 'result.'
        # get start timestamp
        ts_a = datetime.datetime.now()

        gpuresult = sessgpu.run(tf_product)

        # get end timestamp
        ts_b = datetime.datetime.now()

        delta = ts_b - ts_a

        print("Total GPU Runtime Duration was " + repr(int(delta.total_seconds() * 1000)) + " milliseconds.")

        # Now let's have a look at the result.
        print(gpuresult)

        # Close the Session when we're done.
        sessgpu.close()

    # turn off CUDA (turn off GPU for testing of CPU)
    with tf.Session() as sesscpu:
        with tf.device("/cpu:0"):
            print(get_processor_info())  # print the CPU info

            # get start timestamp
            ts_a = datetime.datetime.now()

            cpuresult = sesscpu.run(tf_product)

            # get end timestamp
            ts_b = datetime.datetime.now()

            delta = ts_b - ts_a

            print("Total CPU Runtime Duration was " + repr(int(delta.total_seconds() * 1000)) + " milliseconds.")

            # Now let's have a look at the result.
            print(cpuresult)

            # Close the Session when we're done.
            sesscpu.close()


def get_processor_info():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        return subprocess.check_output(['/usr/sbin/sysctl', "-n", "machdep.cpu.brand_string"]).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        return subprocess.check_output(command, shell=True).strip()
    return ""


calcArray()
