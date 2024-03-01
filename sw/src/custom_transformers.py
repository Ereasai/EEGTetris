# custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import butter, lfilter
import numpy as np

# Custom filter bank transformer
class FilterBank(BaseEstimator, TransformerMixin):
    def __init__(self, filters='LowpassBank'):
        self.filters = filters
        if filters == 'LowpassBank':
            self.freqs_pairs = [[0.5, 4], [4, 8], [8, 13], [13, 30], [30, 40]]
        else:
            self.freqs_pairs = filters

    def fit(self, X, y=None):
        return self  # Nothing to do here

    def transform(self, X, y=None):
        X_filtered = np.concatenate([self.apply_filter(X, pair) for pair in self.freqs_pairs], axis=1)
        return X_filtered

    def apply_filter(self, X, freqs):
        fs = 250.0  # Sampling frequency
        nyq = 0.5 * fs
        low = freqs[0] / nyq
        high = freqs[1] / nyq
        b, a = butter(5, [low, high], btype='band')
        return lfilter(b, a, X, axis=0)