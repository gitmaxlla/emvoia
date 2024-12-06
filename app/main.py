import keras as k
import sys, os
sys.path.append('../') # To import local modules of the project

import sounddevice as sd
import numpy as np

from collections import Counter
from config import Config
from processing import get_features, preprocess


model_path = '../models/emvoia.keras'
labels = {0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful',
          4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised'}

counter = Counter()
average_over_counter = 0
AVERAGE_OVER = 3

EMPTY_TO_CLEAR = 3
to_clear_counter = 0

def count_label(label):
    global counter
    global average_over_counter
    global AVERAGE_OVER
    counter.update([label])
    average_over_counter += 1
    if average_over_counter == AVERAGE_OVER:
        print(counter)
        counter.clear()
        average_over_counter = 0

if __name__ == '__main__':
    cfg = Config() # Using the defaults

    if not os.path.isfile(model_path):
        print('Model not found')
        exit(-1)

    model = k.saving.load_model(model_path)

    stream = sd.Stream(samplerate=cfg.sr, channels=1, blocksize=cfg.input_size)

    stream.start()
    while True:
        indata, overflowed = stream.read(cfg.input_size*2)
        rec_preprocessed = preprocess(np.array(indata.T.tolist()[0]), cfg)
        if rec_preprocessed is not None:
            prediction = labels[np.argmax(model.predict(get_features(np.expand_dims(rec_preprocessed[:cfg.input_size], axis=0), cfg), verbose=0))]
            count_label(prediction)
            # stream.write(rec_preprocessed[:cfg.input_size].astype('float32'))
        else:
            to_clear_counter += 1
            if to_clear_counter == EMPTY_TO_CLEAR:
                counter.clear()
                to_clear_counter = 0
                average_over_counter = 0
