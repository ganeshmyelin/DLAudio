import os
import music21 as m21
import pathlib
import json
import tensorflow as tf
import numpy as np

from musicgenRNN_preprocess import generating_training_sequences

# PATHS
home_dir = pathlib.Path.cwd()
print(f'\n ** home_dir now is ** : {home_dir} ')
#directory_to_run_on = 'test'
directory_to_run_on = 'erk'
#encoded_song_directory = 'test_encoded_songs'
encoded_song_directory = 'erk_encoded_songs'
DATASET_PATH = pathlib.Path(home_dir / 'data/esac/deutschl' / directory_to_run_on)
print(DATASET_PATH)
ENCODED_DATASET_PATH = os.path.join(DATASET_PATH, encoded_song_directory)
SINGLE_FILE_DATASET_PATH = os.path.join(ENCODED_DATASET_PATH, 'single_file_of_all_songs_encoded_MIDI')
#MAPPING_PATH = "mapping.json"
MAPPING_PATH = "mapping_erk.json"

# CONSTANTS
OUTPUT_UNITS = 38
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUMBER_NEURONS_IN_HIDDEN_LAYERS = [256, 256]  # 2 layers of hidden layers of 256 each
EPOCHS = 1
BATCH_SIZE = 8
SAVE_MODEL_PATH = "model.h5"
SEQUENCE_LENGTH = 64
RUN_ON_CPU = False


def check_gpu(run_on_cpu):
    if tf.test.is_gpu_available():
        print("\n GPU Available: Running on remote")
        #print(tf.device)
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



    else:
        print("\n NO GPU : Running on local")

def build_model(output_units, num_units, loss, learning_rate):

    # create model architecture with functional api approach (more flexible than sequential)
    input = tf.keras.layers.Input(shape=(None, output_units))
    x = tf.keras.layers.LSTM(units = NUMBER_NEURONS_IN_HIDDEN_LAYERS[0])(input)
    x = tf.keras.layers.Dropout(rate=0.2)(x)

    output = tf.keras.layers.Dense(output_units,activation='softmax')(x)

    model = tf.keras.Model(input,output)

    # compile model
    model.compile(loss=LOSS,
                  metrics=['accuracy'])
    # model.compile(loss=LOSS,
    #            optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
    #           metrics=['accuracy'])

    model.summary()

    return model


def train(output_units=OUTPUT_UNITS,
          num_units= NUMBER_NEURONS_IN_HIDDEN_LAYERS,
          loss=LOSS,
          learning_rate=LEARNING_RATE,
          run_on_cpu=False):

    # generate the training sequence
    inputs, targets = generating_training_sequences(sequence_length=SEQUENCE_LENGTH,
                                                    single_song_encoded_dataset=SINGLE_FILE_DATASET_PATH,
                                                    mapping_path=MAPPING_PATH)
    print(len(inputs))

    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    print(f"\n Num Epochs to run : {EPOCHS}")
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
    '''
    if run_on_cpu :
        with tf.device('/cpu:0'):
            model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
    else:
        with tf.device('/gpu:0'):
            model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
    '''

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == '__main__':
    check_gpu(run_on_cpu=False)
    train(run_on_cpu=False)
    print('Done')