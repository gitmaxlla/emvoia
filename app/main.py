import keras as k
import sys, os
sys.path.append('../') # To import local modules of the project

import sounddevice as sd
import numpy as np
import silero_vad

from collections import Counter
from config import Config
from processing import get_features, preprocess_voice, extract_voice


models = ['../models/66accuracy_mfcc_bigger.keras', '../models/64accuracy_mfcc_light.keras']

labels = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad',
          4: 'surprise'}

counter = Counter()
average_over_counter = 0
to_clear_counter = 0

rec_data = np.array([])

AVERAGE_OVER = 3 # How many continuous predictions should be done before showing the results
EMPTY_TO_CLEAR = 5 # How many empty recording segments to reset the collected data

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
    if not all([os.path.isfile(path) for path in models]):
        print('Model not found')
        exit(-1)

    models_loaded = []

    for path in models:
        models_loaded.append(k.saving.load_model(path))

    stream = sd.Stream(samplerate=cfg.sr, channels=1, blocksize=cfg.input_size)

    stream.start()
    while True:
        indata, overflowed = stream.read(cfg.input_size)
        current_rec = extract_voice(indata.T.tolist()[0])
        if current_rec is None:
            to_clear_counter += 1
            if to_clear_counter == EMPTY_TO_CLEAR:
                counter.clear()
                to_clear_counter = 0
                current_rec = np.array([])
        else:
            rec_data = np.append(rec_data, current_rec)
            if len(rec_data) >= cfg.input_size:
                rec_preprocessed = preprocess_voice(current_rec[:cfg.input_size], cfg)
                if rec_preprocessed is not None:
                    prediction_data = np.expand_dims(np.swapaxes(get_features(rec_preprocessed, cfg), 0, 1), 0)
                    prediction = models_loaded[0].predict(prediction_data, verbose=0)
                    for i in range(len(models_loaded) - 1):
                        prediction += models_loaded[i + 1].predict(prediction_data, verbose=0)
                    count_label(labels[np.argmax(prediction)])

                rec_data = np.array([])
