class Config:
    def __init__(self, sr=16000, capture_sec=1.0, n_fft=512, n_filters=32, n_mfcc=50, window_type='hamming', window_ms=30, window_overlap=0.548):
        self.sr = sr # All audio captures are converted to this sample rate first
        self.input_size = int(capture_sec * sr) # How many ints are needed to store one chunk
        self.n_mfcc = n_mfcc # The number of features produced by MFCC calculation
        self.n_fft = n_fft # Window size for the fast transform
        self.n_filters = n_filters # How many filters are to be used during STFT
        self.window_type = window_type # What windowing method to use during feature extraction
        self.window_len = int(sr * (window_ms / 1000)) # Calculated length of the window during feature extraction
        self.hop_len = int(self.window_len * (1 - window_overlap)) # How much a window overlaps the previous windowed frame
