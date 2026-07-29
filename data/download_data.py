import os
import urllib.request
import scipy.io
import numpy as np
import yaml

def generate_synthetic_ninapro(output_path, num_channels=12, sample_rate=2000, duration_sec=10):
    """
    Genera un archivo .mat sintético compatible con la estructura de NinaPro DB2
    para pruebas inmediatas sin necesidad de descargar el dataset completo.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_samples = sample_rate * duration_sec
    
    # Generar señal EMG sintética (ruido gaussiano modulado por gestos)
    emg = np.random.normal(0, 0.1, (n_samples, num_channels))
    restimulus = np.zeros((n_samples, 1), dtype=int)
    
    # Simular 5 gestos secuenciales
    chunk_size = n_samples // 6
    for i in range(1, 6):
        start = i * chunk_size
        end = start + chunk_size
        emg[start:end, :] *= (i + 1)  # Diferente amplitud según el gesto
        restimulus[start:end] = i
        
    data = {'emg': emg, 'restimulus': restimulus}
    scipy.io.savemat(output_path, data)
    print(f"✅ Archivo sintético NinaPro creado en: {output_path}")

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    target_file = "data/raw/S1_E1_A1.mat"
    if not os.path.exists(target_file):
        print("⚠️ No se encontró el archivo NinaPro DB2. Generando muestra sintética de prueba...")
        generate_synthetic_ninapro(target_file, config['dataset']['num_channels'], config['dataset']['sample_rate'])