import pytest
import numpy as np
from src.signal_processing.filters import EMGFilter
from src.signal_processing.feature_extraction import window_signal, extract_time_features

def test_emg_filter_dimensions():
    """Verifica que el filtrado mantenga las dimensiones de la señal original."""
    sample_rate = 2000
    signal = np.random.normal(0, 1, (2000, 12))  # 1 segundo de señal sintética
    filt = EMGFilter(sample_rate=sample_rate)
    
    filtered_signal = filt.process(signal)
    
    assert filtered_signal.shape == signal.shape, "La forma de la señal filtrada debe ser idéntica a la original."

def test_window_signal_shape():
    """Verifica la correcta división en ventanas temporales."""
    signal = np.random.normal(0, 1, (2000, 12))
    labels = np.zeros(2000, dtype=int)
    win_size = 500  # 250 ms a 2000 Hz
    step_size = 100 # Overlap
    
    windows, win_labels = window_signal(signal, labels, win_size, step_size)
    
    assert windows.ndim == 3, "La salida de ventanas debe ser un tensor 3D: [Ventanas, Muestras, Canales]."
    assert windows.shape[1] == win_size
    assert windows.shape[2] == 12

def test_feature_extraction():
    """Verifica que las características temporales no contengan NaNs o Infs."""
    window = np.random.normal(0, 1, (500, 12))
    features = extract_time_features(window)
    
    assert not np.isnan(features).any(), "Las características extraídas no deben contener NaNs."
    assert not np.isinf(features).any(), "Las características extraídas no deben contener Infs."