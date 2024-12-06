import librosa
import numpy as np, pandas as pd
import scipy

def get_features(data, cfg):
    return librosa.feature.mfcc(y=data, sr=cfg.sr, n_fft=cfg.n_fft, n_mfcc=32, win_length=cfg.window_len, window=cfg.window_type, hop_length=cfg.hop_len // 2)

def preprocess(data, cfg):
    result = data

    mask = []
    cutoff_value = 0.02 # To remove empty parts of the recordings

    result_abs = pd.Series(result).apply(np.abs)
    result_abs_mean = result_abs.rolling(window=int(cfg.sr / 10), min_periods=1, center=True).mean()

    for mean in result_abs_mean:
        if mean > cutoff_value:
            mask.append(True)
        else:
            mask.append(False)

    result = np.array(pd.Series(result)[mask])

    if len(result) < cfg.input_size: return None

    result = librosa.effects.preemphasis(result)

    b, a = scipy.signal.butter(5, 25 / cfg.sr, 'high') # A highpass filter with 20Hz cutoff frequency
    result = scipy.signal.filtfilt(b, a, result, method='gust', axis=0) # Gust method works better than the default one

    return result
