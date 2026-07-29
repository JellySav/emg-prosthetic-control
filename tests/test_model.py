import pytest
import torch
from src.models.cnn1d_lstm import Conv1DLSTM

def test_model_forward_pass():
    """Verifica la consistencia de las dimensiones de entrada y salida del modelo Conv1D-LSTM."""
    batch_size = 8
    time_steps = 500
    num_channels = 12
    num_classes = 5
    
    model = Conv1DLSTM(num_channels=num_channels, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, time_steps, num_channels)
    
    output = model(dummy_input)
    
    assert output.shape == (batch_size, num_classes), f"Se esperaba la forma [{batch_size}, {num_classes}], pero se obtuvo {output.shape}."