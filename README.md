# EMG-Controlled Prosthetic Hand with Deep Learning & PyBullet

[![CI/CD Pipeline](https://github.com/JellySav/emg-prosthetic-control/actions/workflows/ci.yml/badge.svg)](https://github.com/JellySav/emg-prosthetic-control/actions)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)
![PyBullet](https://img.shields.io/badge/PyBullet-3.2.5-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

En este repositorio se encuentra un proyecto sobre un sistema end-to-end de control mioeléctrico que procesa señales electromiográficas (EMG) de alta densidad mediante redes neuronales convolucionales y recurrentes (Conv1D + BiLSTM) para predecir la intención motora y operar una prótesis robótica articulada en un entorno de simulación 3D en tiempo real.


## Contexto y Problema

Las prótesis mioeléctricas tradicionales dependen principalmente de umbrales estáticos de amplitud eléctrica en músculos remanentes. Esto limita severamente a los usuarios a realizar **un solo movimiento a la vez** y exige un esfuerzo cognitivo elevado para conmutar entre patrones de agarre.

### Desafíos Principales:
1. **Ruido e Interferencia en Señales Biológicas:** Las señales EMG son no estacionarias y están fuertemente contaminadas por artefactos de movimiento, ruido de línea de red ($50/60 \text{ Hz}$) y diafonía (cross-talk) muscular.
2. **Latencia Crítica:** La percepción humana de control en tiempo real exige que el procesamiento, inferencia e inicio del movimiento ocurran en **menos de 300 ms**.
3. **Complejidad Espaciotemporal:** Capturar tanto la correlación espacial entre múltiples canales de superficie como la evolución temporal del patrón muscular.


## ¿Qué hace este proyecto?

Este repositorio aborda el problema tratando el control de prótesis como un reto de **clasificación de secuencias biomédicas de múltiples canales**:

* **Procesamiento de Señal Digital (DSP):** Limpia las señales bioeléctricas mediante filtros digitales Butterworth y Notch, aislando el espectro de activación muscular real ($20 - 450 \text{ Hz}$).
* **Inferencia Deep Learning:** Utiliza una arquitectura híbrida **Conv1D + BiLSTM** que extrae automáticamente patrones espaciales entre 12 canales EMG y patrones temporales mediante memoria recurrente bidireccional en ventanas deslizantes de $250 \text{ ms}$.
* **Traducción Cinemática:** Un controlador mapea las predicciones de la red neuronal a comandos de posición articular en tiempo real dentro del motor físico **PyBullet**.


## Visión y Alcance Futuro

El objetivo a largo plazo de esta iniciativa es cerrar la brecha entre el modelado teórico de señales biomédicas y la robótica de rehabilitación accesible:

* [ ] **Control Continuo (Proporcional):** Transicionar de la clasificación discreta de gestos a la estimación continua de fuerza y ángulos articulares.
* [ ] **Adaptación Intra-Sujeto (Domain Adaptation):** Implementar *Transfer Learning* para reducir el tiempo de calibración cuando un nuevo usuario utiliza la prótesis.
* [ ] **Hardware In-the-Loop (HIL):** Integrar la inferencia en microcontroladores/SBC de borde (ej. Raspberry Pi / NVIDIA Jetson) conectados a sensores EMG físicos (ADS1299 / Myo Armband).

## Arquitectura del Sistema

```mermaid
graph LR
    A[Señal EMG Raw<br/>12 Canales @ 2000Hz] --> B[Filtro Butterworth<br/>20-450 Hz + Notch 50Hz]
    B --> C[Ventaneo Temporal<br/>250ms / Overlap 50ms]
    C --> D[Modelo Híbrido<br/>Conv1D + BiLSTM]
    D --> E[Clasificación de Gesto<br/>Softmax Output]
    E --> F[Controlador Cinemático<br/>Mapeo de Articulaciones]
    F --> G[Simulación 3D<br/>PyBullet Hand Environment]
```

## Estructura del Proyecto
```text
emg-prosthetic-control/
├── .github/workflows/   # Pipeline de Integración Continua (CI) con GitHub Actions
├── config/              # Parámetros globales (frecuencia, modelo, simulación) en YAML
├── data/                # Scripts de descarga y generación sintética (NinaPro DB2)
├── notebooks/           # Análisis exploratorio (EDA) y curvas de entrenamiento
├── src/
│   ├── signal_processing/ # Filtros digitales y extracción de características
│   ├── models/            # Arquitectura PyTorch (Conv1D + BiLSTM) y evaluación
│   └── robotics/          # Control de articulaciones y simulación PyBullet
├── tests/               # Pruebas unitarias de integración con pytest
├── Dockerfile           # Entorno dockerizado reproducible
├── main.py              # Entrypoint principal / Demo end-to-end
└── requirements.txt     # Dependencias del proyecto
```


## Instalación y Uso Rápido
Paso 1 - Clonar repositorio:

```Bash
git clone [https://github.com/tu-usuario/emg-prosthetic-control.git](https://github.com/tu-usuario/emg-prosthetic-control.git)
cd emg-prosthetic-control`
```

Paso 2 - Crear entorno virtual e instalar dependencias:

```Bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Paso 3 - Ejecutar pruebas unitarias:

```Bash
pytest tests/
```

Paso 4 - Ejecutar el Pipeline Completo:
El siguiente comando generará los datos de prueba necesarios, procesará la señal, ejecutará la inferencia con el modelo y abrirá la interfaz de simulación robótica 3D:

```Bash
python data/download_data.py
python main.py
```

## Ejecución con Docker
```Bash
# Construir la imagen
docker build -t emg-prosthetic .

# Ejecutar el contenedor
docker run --rm emg-prosthetic
```

## Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
