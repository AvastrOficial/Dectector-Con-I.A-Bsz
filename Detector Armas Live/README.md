# Detector de Armas en Video 
Este proyecto implementa un sistema de detección de armas en transmisiones de video en tiempo real utilizando técnicas de inteligencia artificial. 

## Características
- Detección en tiempo real de armas de fuego en video.
- Utiliza modelos de detección de objetos (YOLOv8 o similar) preentrenados.
- Visualización en vivo con cuadros delimitadores sobre las armas detectadas.
- Posibilidad de usar cámara web o archivos de video como entrada.

## Requisitos
- Python 3.x
- Librerías:
  - OpenCV (`opencv-python`)
  - PyTorch (o framework compatible con el modelo)
  - Ultralytics YOLO (si se usa YOLOv8)
  - Numpy
  - (Otros módulos según el código específico)

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/AvastrOficial/Dectector-Con-I.A-Bsz.git
   
   cd Detector\ Armas\ Live/dteccion_video
   ```
## Instala las dependencias:
   ```bash
pip install -r requirements.txt

```
## Ejecuta el script de detección:
   ```bash
python deteccion_video.py
   ```
> Selecciona la fuente de video (cámara o archivo).

> Visualiza las detecciones en tiempo real.

Estructura del Proyecto
   ```bash
.
├── deteccion_video.py   # Script principal para detección en video
├── requirements.txt     # Dependencias del proyecto
└── modelos/         
   ```
### Inport Moderlo De Dectector de arma H5
https://drive.google.com/file/d/1ClUUCWUxKPh-vus_cLB6f1d2IoRgMm6Q/view?usp=drive_link
