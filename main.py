import yaml
import scipy.io
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.signal_processing.filters import EMGFilter
from src.signal_processing.feature_extraction import window_signal
from src.models.cnn1d_lstm import Conv1DLSTM
from src.robotics.hand_sim import HandSimulation

def main():
    print("Iniciando Pipeline EMG Prosthetic Control...")
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 1. Cargar datos
    data_path = "data/raw/S1_E1_A1.mat"
    mat = scipy.io.loadmat(data_path)
    emg_raw = mat['emg']
    labels = mat['restimulus'].flatten()
    
    # 2. Filtrado
    filt = EMGFilter(
        sample_rate=config['dataset']['sample_rate'],
        lowcut=config['processing']['lowcut'],
        highcut=config['processing']['highcut']
    )
    emg_filtered = filt.process(emg_raw)
    
    # 3. Ventaneo
    win_samples = int((config['dataset']['window_size_ms'] / 1000.0) * config['dataset']['sample_rate'])
    step_samples = int(((config['dataset']['window_size_ms'] - config['dataset']['overlap_ms']) / 1000.0) * config['dataset']['sample_rate'])
    
    X, y = window_signal(emg_filtered, labels, win_samples, step_samples)
    
    # Filtrar solo clases válidas (0 a num_classes-1)
    mask = y < config['dataset']['num_classes']
    X, y = X[mask], y[mask]
    
    print(f"Dataset procesado: {X.shape[0]} ventanas de forma {X.shape[1:]}")
    
    # 4. Inferencia con Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv1DLSTM(
        num_channels=config['dataset']['num_channels'],
        num_classes=config['dataset']['num_classes']
    ).to(device)
    
    model.eval()
    sample_input = torch.tensor(X[:1], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(sample_input)
        pred_class = torch.argmax(output, dim=1).item()
        
    print(f"Clase predicha para la primera ventana: {pred_class}")
    
    # 5. Ejecutar Simulación Robótica
    print("Iniciando simulación PyBullet...")
    sim = HandSimulation(gui=config['simulation']['gui'])
    sim.update_pose(pred_class)
    print("Demostración completada con éxito.")

if __name__ == "__main__":
    main()