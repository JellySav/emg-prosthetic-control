import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

class EMGFilter:
    def __init__(self, sample_rate=2000, lowcut=20.0, highcut=450.0, notch_freq=50.0, notch_q=30.0):
        self.fs = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.notch_q = notch_q

    def butter_bandpass(self):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        return b, a

    def notch_filter(self):
        nyq = 0.5 * self.fs
        freq = self.notch_freq / nyq
        b, a = iirnotch(freq, self.notch_q)
        return b, a

    def process(self, signal):
        """Aplica filtro de banda y filtro notch a una señal de N canales."""
        b_band, a_band = self.butter_bandpass()
        b_notch, a_notch = self.notch_filter()
        
        filtered = filtfilt(b_band, a_band, signal, axis=0)
        filtered = filtfilt(b_notch, a_notch, filtered, axis=0)
        return filtered