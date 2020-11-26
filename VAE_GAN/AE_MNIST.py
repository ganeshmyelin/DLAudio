import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np

# PATHS
home_dir = pathlib.Path.cwd()
print(f'\n ** home_dir now is ** : {home_dir} ')

def check_gpu():
    if tf.test.is_gpu_available():
        print("\n GPU is Available")
        # print(tf.de)
        '''
        if run_on_cpu:
            tf.config.set_visible_devices([], 'GPU')
            # Set CPU as available physical device
            # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
            # tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
            # To find out which devices your operations and tensors are assigned to

        tf.debugging.set_log_device_placement(True)

        # Create some tensors and perform an operation
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)

        print(c)
        '''

    else:
        print("\n GPU is not available")


if __name__ == '__main__':
    check_gpu()
    # load mnist data
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    tf.keras.layers.Conv2D()
    # reshape to (28,28,1) and normalize input images
    x_train = np.reshape(x_train)