from tensorflow import keras
from tensorflow.keras import layers

# Definir la arquitectura del modelo CNN
modelo = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 3)),  # Dimensiones de las imágenes de entrada (alto, ancho, canales)
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # Capa de salida con una neurona y función de activación sigmoide para clasificación binaria
    ]
)

# Mostrar un resumen de la arquitectura del modelo
modelo.summary()

# Compilar el modelo
modelo.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

print("\n¡Modelo CNN construido y compilado!")
print("Ahora puedes usar este modelo con tus datos de entrenamiento.")