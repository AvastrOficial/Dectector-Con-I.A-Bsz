# Detector de Armas - BSZ

## Descripción
Este proyecto permite detectar armas en tiempo real en videos o transmisiones en vivo. Utiliza la biblioteca TensorFlow.js con el modelo Coco-SSD para la detección de objetos, y el resultado es mostrado en una interfaz web que resalta los objetos detectados como armas. Además, proporciona estadísticas en tiempo real sobre las armas detectadas.

## Tecnologías utilizadas
- **HTML5**: Para la estructura de la página web.
- **CSS3**: Para el estilo de la página web.
- **JavaScript**: Para la interacción y la lógica de detección.
- **TensorFlow.js**: Para la implementación del modelo de detección de objetos.
- **Coco-SSD**: Un modelo preentrenado de detección de objetos en imágenes y videos.

## Librerías y Recursos

- **TensorFlow.js**:
  - Versión: 4.18.0
  - CDN: [TensorFlow.js CDN](https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0)
  - Propósito: Permite el uso de modelos de Machine Learning directamente en el navegador utilizando JavaScript.

- **Coco-SSD**:
  - Modelo: Coco-SSD
  - CDN: [Coco-SSD CDN](https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd)
  - Propósito: Modelo preentrenado para la detección de objetos, utilizado aquí para detectar armas (u otros objetos como teléfonos móviles o pelotas de deporte).

## Funcionalidades
- **Carga de video**: Los usuarios pueden cargar un archivo de video en formato MP4, que será procesado en tiempo real.
- **Detección de armas**: Utiliza el modelo Coco-SSD para identificar objetos que podrían ser armas (como cuchillos o pistolas) en el video.
- **Resaltado en tiempo real**: Los objetos detectados son resaltados en el video con un cuadro rojo, y el nombre del objeto detectado (con el porcentaje de certeza) es mostrado sobre el cuadro.
- **Estadísticas**: Se muestran estadísticas en tiempo real sobre la cantidad de armas detectadas en el cuadro actual y el total acumulado en todo el video.

# Funcionamiento :
### Carga de video
- El usuario puede cargar un video MP4 mediante un botón de carga.
- El video se sube a un servidor y se obtiene una URL para reproducirlo.

### Detección de objetos
- El video es procesado por el modelo Coco-SSD utilizando TensorFlow.js. Cada cuadro del video es analizado para detectar objetos.
- Si se detecta un objeto relevante (como un cuchillo o un arma), se resalta con un cuadro rojo en el video.

### Interfaz de usuario
- La interfaz es simple, con un panel de navegación a la izquierda que incluye información sobre el sistema, el autor, las tecnologías y el uso del sistema.
- En el panel principal se muestra el video con las armas detectadas resaltadas.
- En la parte inferior, se muestran las estadísticas de detección en tiempo real.

## Código JavaScript
Cargar y usar el modelo Coco-SSD
```javascript
cocoSsd.load().then(loadedModel => {
    model = loadedModel;
    console.log('Modelo Coco-SSD cargado ✅');
    startCanvasRendering();
});
```
Este bloque carga el modelo Coco-SSD y, una vez cargado, inicia el procesamiento del video para la detección.


## Detección en tiempo real : 
```javascript
model.detect(video).then(predictions => {
    predictions.forEach(pred => {
        if (pred.class === 'knife' || pred.class === 'arma') {
            // Resaltar el objeto detectado
        }
    });
});
```
Aquí, el modelo detecta objetos en el video y verifica si el objeto detectado es un arma (como un cuchillo). Si es un arma, se dibuja un cuadro alrededor del objeto y se actualizan las estadísticas.


## Subir y cambiar el video
```javascript
document.getElementById('fileInput').addEventListener('change', function(e) {
    // Subir video
});
```
Este código permite al usuario cargar un archivo de video, que luego se sube a un servidor y se usa como fuente para el análisis.

## Instalación
Clona este repositorio en tu máquina local:

```bash
git clone https://github.com/tu_usuario/detector-de-armas-bsz.git
```
Abre el archivo index.html en tu navegador para comenzar a usar la aplicación.

![image](https://github.com/user-attachments/assets/407f52ad-6c24-4892-96a9-bdf7459f180e)
