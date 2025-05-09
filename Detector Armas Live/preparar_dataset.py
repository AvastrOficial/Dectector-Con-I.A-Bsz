import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def cargar_y_preprocesar_imagenes(ruta_carpeta, etiqueta, tamaño_objetivo):
    imagenes = []
    etiquetas = []
    for nombre_archivo in os.listdir(ruta_carpeta):
        ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
        imagen = cv2.imread(ruta_completa)
        if imagen is not None:
            imagen_redimensionada = cv2.resize(imagen, tamaño_objetivo)
            imagenes.append(imagen_redimensionada)
            etiquetas.append(etiqueta)
        else:
            print(f"Error al cargar la imagen: {ruta_completa}")
    return np.array(imagenes), np.array(etiquetas)

# Define las rutas a tus carpetas y el tamaño objetivo
ruta_base = r"C:\mi proyecto armas"
ruta_armas = os.path.join(ruta_base, "imagenes entrenar", "dataset", "armas")
ruta_no_armas = os.path.join(ruta_base, "imagenes entrenar", "dataset", "no_armas")
tamaño_objetivo = (128, 128)

# Cargar y preprocesar las imágenes de armas (etiqueta 1)
imagenes_armas, etiquetas_armas = cargar_y_preprocesar_imagenes(ruta_armas, 1, tamaño_objetivo)

# Cargar y preprocesar las imágenes de no armas (etiqueta 0)
imagenes_no_armas, etiquetas_no_armas = cargar_y_preprocesar_imagenes(ruta_no_armas, 0, tamaño_objetivo)

# Combinar las imágenes y las etiquetas de ambas clases
todas_las_imagenes = np.concatenate([imagenes_armas, imagenes_no_armas])
todas_las_etiquetas = np.concatenate([etiquetas_armas, etiquetas_no_armas])

# Normalizar los valores de los píxeles al rango [0, 1]
todas_las_imagenes = todas_las_imagenes.astype('float32') / 255.0

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    todas_las_imagenes, todas_las_etiquetas, test_size=0.2, random_state=42, stratify=todas_las_etiquetas
)

print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño del conjunto de prueba: {len(X_test)}")
print(f"Forma de las imágenes de entrenamiento: {X_train.shape}")
print(f"Forma de las etiquetas de entrenamiento: {y_train.shape}")
print(f"Forma de las imágenes de prueba: {X_test.shape}")
print(f"Forma de las etiquetas de prueba: {y_test.shape}")

print("\n¡Dataset cargado, preprocesado y dividido!")
print("Ahora puedes usar X_train, y_train para entrenar tu modelo y X_test, y_test para evaluarlo.")