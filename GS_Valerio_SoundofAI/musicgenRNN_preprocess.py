import os
import music21 as m21
import pathlib
import json
import tensorflow as tf
import numpy as np

# PATHS
'''
home_dir = pathlib.Path.cwd()
print(f'\n ** home_dir now is ** : {home_dir} ')
# directory_to_run_on = 'test'
directory_to_run_on = 'erk'
encoded_song_directory = 'erk_encoded_songs'
DATASET_PATH = pathlib.Path(home_dir / 'data/esac/deutschl' / directory_to_run_on)
print(DATASET_PATH)
ENCODED_DATASET_PATH = os.path.join(DATASET_PATH, encoded_song_directory)
SINGLE_FILE_DATASET_PATH = os.path.join(ENCODED_DATASET_PATH, 'single_file_of_all_songs_encoded_MIDI')
MAPPING_PATH = "mapping_erk.json"
'''

# VARIABLES used in code below
ACCEPTABLE_DURATIONS = [
    0.25,  # 16th note 1/4 of a beat
    0.5,
    0.75,  # dotted note : 8th+16th note
    1,
    1.5,
    2,
    3,
    4,  # whole note (4 quarter notes)
]
SEQUENCE_LENGTH = 64


def check_gpu():
    if tf.test.is_gpu_available():
        print("\n GPU Available: Running on remote")
    else:
        print("\n NO GPU : Running on local")


# music 21 : Symbolic music converter, analyzing music, representing music in OO manner with attributes etc.
# can read symbolic music from kern, MIDI,MusicXML.. and convert, save into kern, MIDI..

def load_songs_in_kern(dataset_path):
    songs = []
    # go through all files in the dataset, load with music music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # print(f"key : {key}")
    # get interval for transposition e.g BMaj -> Cmaj => transpose by 1 interval. so need to figure out interval for
    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song


def encode_song(song, time_step=0.25):
    # pitch = 60, duration = 1.0 -> [60,"-","-","_"]

    encoded_song = []
    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert note/rest to time series Notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to string
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path, encoded_dataset_path):
    # pass

    # load the folks songs
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs")
    # song = songs[0]
    # song.show() #can see this song in musescore as staff Notation

    # filter out songs that have non acceptable durations
    # 16th, 8th notes etc - standard note durations

    for i, song in enumerate(songs):
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to Cmaj/Amin
        song = transpose(song)

        # encode songs wihth music time representation
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(encoded_dataset_path, str(i))
        with open(save_path, 'w') as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    # load encoded songs and add delimiters
    songs = ""

    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    songs = songs[:-1]

    # save string that contains all dataset files
    with open(file_dataset_path, 'w') as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocabulary
    songu = songs.split()
    vocabulary = list(set(songu))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save vocabulary to json file
    with open(mapping_path, 'w') as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs, mapping_path):
    int_songs = []

    # load mappings from mapping json file
    with open(mapping_path, 'r') as fp:
        mappings = json.load(fp)

    # cast songs string to list of str
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generating_training_sequences(sequence_length, single_song_encoded_dataset, mapping_path):
    # [11,12,13,14,...] -> input i: [11,12], relative target t: 13; i:,[12,13], t:14,....

    # load songs and map to integers
    songs = load(single_song_encoded_dataset)
    int_songs = convert_songs_to_int(songs, mapping_path)

    # generate training sequences
    # for as dataset of. 100 symbols say with 64 sequences length
    # the # of sequences will be 100-64 characters= 36
    # how : 64 sets at a time. so 1 set of 64, slide 1 to next set of 64.. 36 times
    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length

    for i in range(num_sequences):
        inputs.append(int_songs[i:i + sequence_length])
        targets.append(int_songs[i + sequence_length])

    # one-hot encode the sequences
    # length of encode == length of the vocabulary
    # every integer is now hot encoding in a series of 1 and 0. only 1 position is 1 for a given integer
    # in this dataset, the vocabulary contains 18 unique elements. (See mapping.json length)
    vocabulary_size = len(set(int_songs))
    inputs = tf.keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def start():
    # preprocess data
    preprocess(DATASET_PATH, ENCODED_DATASET_PATH)

    songs1 = create_single_file_dataset(dataset_path=ENCODED_DATASET_PATH,
                                        file_dataset_path=SINGLE_FILE_DATASET_PATH,
                                        sequence_length=SEQUENCE_LENGTH)

    create_mapping(songs1, MAPPING_PATH)

    '''
    inputs, targets = generating_training_sequences(sequence_length=SEQUENCE_LENGTH,
                                                    single_song_encoded_dataset=SINGLE_FILE_DATASET_PATH)
    
    '''

    print('here')


if __name__ == '__main__':
    check_gpu()
    start()
    print('out here')
