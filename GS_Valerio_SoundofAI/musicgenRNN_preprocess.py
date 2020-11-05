import os
import music21 as m21
import pathlib

ACCEPTABLE_DURATIONS = [
    0.25, #16th note 1/4 of a beat
    0.5,
    0.75, # dotted note : 8th+16th note
    1,
    1.5,
    2,
    3,
    4, #whole note (4 quarter notes)
]


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

def has_acceptable_durations(song,acceptable_durations):
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

    print(f"key : {key}")
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
        if isinstance(event,m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event,m21.note.Rest):
            symbol = "r"

        # convert note/rest to time series Notation
        steps = int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    #cast encoded song to string
    encoded_song = " ".join(map(str,encoded_song))

    return encoded_song

def preprocess(dataset_path):
    #pass

    # load the folks songs
    songs = load_songs_in_kern(dataset_path)
    song = songs[0]
    # song.show()

    # filter out songs that have non acceptable durations
    # 16th, 8th notes etc - standard note durations

    for i, song in enumerate(songs):
        if not has_acceptable_durations(song,ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to Cmaj/Amin
        song = transpose(song)

        # encode songs wihth music time representation
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(dataset_path/'esac_encoded_songs', str(i))
        with open(save_path, 'w') as fp :
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp :
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path,file_dataset_path,sequence_length=64):

    new_song_delimiter = "/ " * sequence_length
    # load encoded songs and add delimiters
    songs = ""

    for path,_,files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    songs = songs[:-1]

    # save string that contains all dataset files
    with open(file_dataset_path, 'w') as fp:
        fp.write(songs)

    return songs

def create_mapping(songs):
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i



    # save vocabulary to json file

if __name__ == '__main__':

    home_dir = pathlib.Path.cwd()
    print(f'\n ** home_dir now is ** : {home_dir} ')
    DATASET_PATH = pathlib.Path(home_dir/'data/esac/deutschl/test')

    print(DATASET_PATH)

    songs = load_songs_in_kern(DATASET_PATH)
    print(f"Loaded {len(songs)} songs")
    song = songs[0]
    preprocess(DATASET_PATH)
    '''
    print(f"has acceptable_durations? {has_acceptable_durations(song,ACCEPTABLE_DURATIONS)}")
    transposed_song = transpose(song)
    transposed_song.show()   
    '''
    encoded_dataset_path = os.path.join(DATASET_PATH,'esac_encoded_songs')
    file_dataset_path = os.path.join(encoded_dataset_path,'foo')

    songs1 = create_single_file_dataset(dataset_path=encoded_dataset_path,
                                       file_dataset_path=file_dataset_path,
                                       sequence_length=64)

