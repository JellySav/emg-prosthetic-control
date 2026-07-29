import numpy as np
import pywt

def extract_time_features(window):
    """
    Extrae características temporales clásicas de una ventana EMG [samples, channels]:
    - Root Mean Square (RMS)
    - Mean Absolute Value (MAV)
    - Zero Crossings (ZC)
    - Waveform Length (WL)
    """
    rms = np.sqrt(np.mean(window**2, axis=0))
    mav = np.mean(np.abs(window), axis=0)
    
    # Zero Crossings
    zc = np.sum(np.diff(np.sign(window) != 0, axis=0), axis=0)
    
    # Waveform Length
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    
    return np.concatenate([rms, mav, zc, wl])

def window_signal(signal, labels, window_size, step_size):
    """Divide la señal continua en ventanas con solapamiento."""
    windows = []
    win_labels = []
    
    n_samples = signal.shape[0]
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        win = signal[start:end, :]
        lbl = labels[start:end]
        
        # Asignar la clase más frecuente en la ventana
        mode_label = np.bincount(lbl.flatten()).argmax()
        
        windows.append(win)
        win_labels.append(mode_label)
        
    return np.array(windows), np.array(win_labels)