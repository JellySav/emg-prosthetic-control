# EMG-Controlled Prosthetic Hand with Deep Learning & PyBullet

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)
![PyBullet](https://img.shields.io/badge/PyBullet-3.2.5-green)

Este proyecto implementa un pipeline completo en tiempo real para interpretar señales biomédicas de Electromiografía (EMG) de alta densidad y traducirlas en comandos de movimiento para una prótesis robótica en simulación 3D.


## Arquitectura del Sistema

```mermaid
graph LR
    A[Señal EMG Raw] --> B[Filtro Butterworth + Notch]
    B --> C[Ventaneo Temporal 250ms]
    C --> D[Modelo Conv1D + BiLSTM]
    D --> E[Clasificación de Gesto]
    E --> F[Control de Motores en PyBullet]
```

## Estructura del Proyecto
data/: Scripts de preparación para el dataset NinaPro DB2.

src/signal_processing/: Filtros biomédicos (20-450 Hz) y segmentación.

src/models/: Red Neuronal Convolucional + Recurrente en PyTorch.

src/robotics/: Control cinemático e integración con PyBullet.

## Instalación y Uso Rápido
Paso 1 - Clonar repositorio:

```Bash
git clone [https://github.com/tu-usuario/emg-prosthetic-control.git](https://github.com/tu-usuario/emg-prosthetic-control.git)
cd emg-prosthetic-control`
```

Paso 2 - Instalar dependencias:

```Bash
pip install -r requirements.txt
```

Paso 3 - Generar datos de prueba y ejecutar pipeline completo:

```Bash
python data/download_data.py
python main.py`
```

## Ejecución con Docker
```Bash
docker build -t emg-prosthetic .
docker run --rm emg-prosthetic
```