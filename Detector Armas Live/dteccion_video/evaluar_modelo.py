import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time  # Importamos la librería 'time'

# Define las rutas a tus carpetas y el tamaño objetivo
ruta_base = r"C:\mi proyecto armas"
ruta_armas = os.path.join(ruta_base, "imagenes entrenar", "dataset", "armas")
ruta_no_armas = os.path.join(ruta_base, "imagenes entrenar", "dataset", "no_armas")
tamaño_objetivo = (128, 128)

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

# Cargar y preprocesar el dataset
imagenes_armas, etiquetas_armas = cargar_y_preprocesar_imagenes(ruta_armas, 1, tamaño_objetivo)
imagenes_no_armas, etiquetas_no_armas = cargar_y_preprocesar_imagenes(ruta_no_armas, 0, tamaño_objetivo)

todas_las_imagenes = np.concatenate([imagenes_armas, imagenes_no_armas])
todas_las_etiquetas = np.concatenate([etiquetas_armas, etiquetas_no_armas])
todas_las_imagenes = todas_las_imagenes.astype('float32') / 255.0

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    todas_las_imagenes, todas_las_etiquetas, test_size=0.2, random_state=42, stratify=todas_las_etiquetas
)

# Definir la arquitectura del modelo
modelo = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compilar el modelo
modelo.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Definir los callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
callbacks_list = [early_stopping, reduce_lr]

# Entrenar el modelo y guardar el historial
epochs = 200  # Aumentamos el número de épocas
batch_size = 32
print("\nIniciando el entrenamiento...")
start_time = time.time()  # Tiempo de inicio del entrenamiento

historial = modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=1,
                        callbacks=callbacks_list)

end_time = time.time()  # Tiempo de finalización del entrenamiento
total_time = end_time - start_time
print(f"\n¡Entrenamiento completado en: {time.strftime('%H:%M:%S', time.gmtime(total_time))}!")

print("\n¡Modelo entrenado!")

# Evaluar el modelo en el conjunto de prueba
perdida, precision = modelo.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida en el conjunto de prueba: {perdida:.4f}")
print(f"Precisión en el conjunto de prueba: {precision:.4f}")

print("\n¡Evaluación completada!")

# Visualizar la pérdida del entrenamiento y la validación
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historial.history['loss'], label='Pérdida de entrenamiento')
plt.plot(historial.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Visualizar la precisión del entrenamiento y la validación
plt.subplot(1, 2, 2)
plt.plot(historial.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()

print("\n¡Gráficas de pérdida y precisión mostradas!")

# Predicciones del modelo en el conjunto de prueba
y_pred_prob = modelo.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Generar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(cm)

# Visualizar la matriz de confusión con seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Arma', 'Arma'], yticklabels=['No Arma', 'Arma'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

# Calcular otras métricas de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Arma', 'Arma']))

# Calcular el área bajo la curva ROC (AUC)
auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nÁrea bajo la curva ROC (AUC): {auc:.4f}")

print("\n¡Métricas de evaluación completas!")

# Guardar el modelo entrenado
ruta_guardado_modelo = os.path.join(ruta_base, "modelo_detector_armas.h5")
modelo.save(ruta_guardado_modelo)
print(f"\n¡Modelo guardado en: {ruta_guardado_modelo}")