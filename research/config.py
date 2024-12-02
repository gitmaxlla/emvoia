class Config:
    def __init__(self, sr=8000, capture_sec=0.2, dataset_frame_overlap=0.1, n_fft=512, n_filters=32, n_mfcc=20, window_type='hamming', window_ms=30, window_overlap=0.5):
        self.sr = sr # All audio captures are converted to this sample rate first
        self.dataset_frame_overlap = dataset_frame_overlap # How big is the overlap between dataset samples when splitting them into chunks
        self.input_size = int(capture_sec * sr) # How many ints are needed to store one chunk
        self.n_mfcc = n_mfcc # The number of features produced by MFCC calculation
        self.n_fft = n_fft # Window size for the fast transform
        self.n_filters = n_filters # How many filters are to be used during STFT
        self.window_type = window_type # What windowing method to use during feature extraction
        self.window_len = int(sr * (window_ms / 1000)) # Calculated length of the window during feature extraction
        self.hop_len = int(self.window_len * window_overlap) # How much a window overlaps the previous windowed frame
