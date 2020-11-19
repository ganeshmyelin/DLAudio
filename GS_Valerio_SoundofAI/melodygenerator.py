import os
import music21 as m21
import pathlib
import json
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from musicgenRNN_train import check_gpu, SEQUENCE_LENGTH, MAPPING_PATH


class MelodyGenerator:

    def __init__(self, model_path="model.h5"):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, 'r') as fp:
            self.mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

        # seed : '64 _ 63 _ _" 5-10 symbols that will be fed in to the netwrok to create next
        # num_steps : # of num_steps (time series) for the netwrok to generate
        # create seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to integers
        seed = [self.mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit the seed to max_sequence_length
            seed1 = seed[-max_sequence_length:]

            # one hot encode seed
            onehot_seed = keras.utils.to_categorical(seed1, num_classes=len(self.mappings))
            # (max_sequence_length, len(self.mappings))
            # keras expects a batch dimension for predict/training
            # so need to add a 1 batch dimension
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # will look like this, say for 4 symbols [0.1, 0.2, 0.1, 0.6] , totalling to 1, due to softmax

            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map into encoding
            output_symbol = [k for k, v in self.mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update the melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        # temperature -> infinity
        # temperature -> 0
        # temperature = 1.
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))  # [0,1,2,3]
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel2.mid"):

        # create a music stream
        stream = m21.stream.Stream()

        # parse all the symbols in the melody and create note/rest objects
        # say melody is a list of 60 _ _ _ r _ 62 _ -> music21 objects
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):
                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter  # 0.25*4 = 1

                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarter_length_duration=quarter_length_duration)
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarter_length=quarter_length_duration)

                    stream.append(m21_event)
                    step_counter = 1

                start_symbol = symbol

            # handle prolongation
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == "__main__":
    # PATHS
    home_dir = pathlib.Path.cwd()
    print(f'\n ** home_dir now is ** : {home_dir} ')
    check_gpu(run_on_cpu=False)
    mg = MelodyGenerator()
    seed1 = "55 _ 60 _ 60 _ _ _ 62 _"

    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.9)
    print(melody)
    mg.save_melody(melody)
