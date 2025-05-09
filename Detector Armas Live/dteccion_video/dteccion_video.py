import cv2
import numpy as np
from tensorflow import keras
import os
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import time
import asyncio

# Define la ruta al modelo guardado
ruta_modelo = os.path.join(r"C:\mi proyecto armas", "modelo_detector_armas.h5")

# Cargar el modelo entrenado
try:
    modelo_cargado = keras.models.load_model(ruta_modelo)
    print(f"¡Modelo cargado exitosamente desde: {ruta_modelo}!")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Configuración del bot de Telegram
TOKEN_BOT = '7695975562:AAEjLpaOBHzs-76At0cMWuHR58mp_w_5G38'  # Reemplázalo con tu token de BotFather
CHAT_ID = 6165088900     # Reemplázalo con tu ID numérico de chat

# Crear la aplicación de Telegram
app = ApplicationBuilder().token(TOKEN_BOT).build()

# Variables para control de tiempo de detección
arma_detectada_inicio = None
arma_mensaje_enviado = False

# Enviar mensaje a Telegram de forma asíncrona
async def enviar_mensaje():
    try:
        await app.bot.send_message(chat_id=CHAT_ID, text="⚠️ Arma detectada.")
        print("Mensaje enviado por Telegram: Arma detectada.")
    except Exception as e:
        print(f"Error al enviar mensaje de Telegram: {e}")

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

print("Iniciando detección en tiempo real con filtro CLAHE. Presiona 'q' para salir.")

# Inicializar el objeto CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    # Leer un fotograma
    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo leer el fotograma.")
        break

    # Convertir a espacio de color HSV para aplicar CLAHE en el canal de luminancia (V)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Aplicar CLAHE al canal de luminancia
    v_clahe = clahe.apply(v)

    # Fusionar los canales de nuevo
    hsv_clahe = cv2.merge([h, s, v_clahe])

    # Convertir de nuevo a BGR para la entrada del modelo
    frame_filtrado = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    # Redimensionar y normalizar para la predicción
    img_redimensionada_prediccion = cv2.resize(frame_filtrado, (128, 128))
    img_normalizada = img_redimensionada_prediccion.astype('float32') / 255.0
    img_expandida = np.expand_dims(img_normalizada, axis=0)

    # Realizar la predicción
    prediccion = modelo_cargado.predict(img_expandida)[0][0]

    # Definir el umbral
    umbral = 0.5

    # Obtener las dimensiones del fotograma
    height, width, _ = frame.shape

    if prediccion > umbral:
        # Se detecta un arma: cuadro rojo y texto rojo
        etiqueta = f"Arma: {prediccion:.2f}"
        color_cuadro = (0, 0, 255)  # Rojo
        color_texto = (0, 0, 255)  # Rojo

        # Dibujar el cuadro
        cv2.rectangle(frame, (0, 0), (width, height), color_cuadro, 2)

        # Mostrar la etiqueta
        cv2.putText(frame, etiqueta, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)

        # Inicia conteo si no ha iniciado
        if arma_detectada_inicio is None:
            arma_detectada_inicio = time.time()

        # Si han pasado más de 5 segundos y aún no se envió mensaje
        elif time.time() - arma_detectada_inicio >= 5 and not arma_mensaje_enviado:
            # Llamar la función asíncrona para enviar mensaje
            asyncio.run(enviar_mensaje())
            arma_mensaje_enviado = True
    else:
        # No se detecta arma: mostrar el texto en verde
        texto_no_detectado = "Armas no detectada por el momento"
        color_texto_no_detectado = (0, 255, 0)  # Verde
        cv2.putText(frame, texto_no_detectado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto_no_detectado, 2)
        
        # Reiniciar conteo si se pierde la detección
        arma_detectada_inicio = None
        arma_mensaje_enviado = False

    # Mostrar el fotograma original
    cv2.imshow("Detección de Armas en Tiempo Real", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()

print("Detección en tiempo real finalizada.")

