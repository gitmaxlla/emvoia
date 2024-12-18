import librosa
import numpy as np, pandas as pd
import silero_vad

vad_model = silero_vad.load_silero_vad()

def get_features(data, cfg):
    return librosa.feature.melspectrogram(y=data, sr=cfg.sr, n_fft=cfg.n_fft, win_length=cfg.window_len, window=cfg.window_type, hop_length=cfg.hop_len)
    # return librosa.feature.mfcc(y=data, sr=cfg.sr, n_fft=cfg.n_fft, n_mfcc=32, win_length=cfg.window_len, window=cfg.window_type, hop_length=cfg.hop_len)

def preprocess(data, cfg):
    vad_result = silero_vad.get_speech_timestamps(data, vad_model, return_seconds=False)

    voice_data = []

    for voice_detection in vad_result:
        voice_data.append(data[voice_detection['start']:voice_detection['end']])

    if not voice_data: return None

    result = np.hstack(voice_data)

    if len(result) < cfg.input_size: return None

    # result = librosa.effects.preemphasis(result)

    # b, a = scipy.signal.butter(5, 25 / cfg.sr, 'high') # A highpass filter with 20Hz cutoff frequency
    # result = scipy.signal.filtfilt(b, a, result, method='gust', axis=0) # Gust method works better than the default one

    res_max = np.abs(result).max() # Data normalization
    result = (result) / (res_max)

    return result
