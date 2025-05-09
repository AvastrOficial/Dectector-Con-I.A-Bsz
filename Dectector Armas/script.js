       const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let model;
let totalDetected = 0; // Total de armas detectadas en todo el video
let detectedObjects = []; // Arreglo para almacenar las armas detectadas
let currentCount = 0; // Armas detectadas en el cuadro actual

// Cargar el modelo Coco-SSD
cocoSsd.load().then(loadedModel => {
    model = loadedModel;
    console.log('Modelo Coco-SSD cargado ✅');
    startCanvasRendering();
});

// Subida y cambio de video
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('reqtype', 'fileupload');
    formData.append('time', '1h');
    formData.append('fileToUpload', file);

    alert('Subiendo archivo a Litterbox...');

    fetch('https://litterbox.catbox.moe/resources/internals/api.php', {
        method: 'POST',
        body: formData
    })
    .then(res => res.text())
    .then(url => {
        alert('Subido correctamente ✅\nURL: ' + url);
        changeVideoSourceSafely(url);
    })
    .catch(err => {
        alert('Error al subir: ' + err);
    });
});

// Cambiar la fuente del video
function changeVideoSourceSafely(newUrl) {
    const source = video.querySelector('source');
    source.src = newUrl;
    video.load();

    video.oncanplay = function() {
        video.play();
        totalDetected = 0;
        currentCount = 0;
        detectedObjects = []; // Resetear objetos detectados
        document.getElementById('totalCount').innerText = totalDetected;
        document.getElementById('currentCount').innerText = currentCount;
        startCanvasRendering();
    };
}

// Dibuja video + detección
function startCanvasRendering() {
    function draw() {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (model) {
            model.detect(video).then(predictions => {
                let newDetectedObjects = []; // Array temporal para las nuevas detecciones
                let newCurrentCount = 0;

                predictions.forEach(pred => {
                    // Cambiar la clase de 'cell phone' y 'sports ball' a 'arma'
                    let label = pred.class;
                    if (label === 'cell phone' || label === 'sports ball') {
                        label = 'arma';
                    }

                    // Detectar si el objeto es relevante
                    if (label === 'knife' || label === 'arma') {
                        const [x, y, width, height] = pred.bbox;
                        const centerX = x + width / 2;
                        const centerY = y + height / 2;

                        let isNewObject = true;
                        for (let i = 0; i < detectedObjects.length; i++) {
                            const obj = detectedObjects[i];
                            const distance = Math.sqrt(Math.pow(centerX - obj.centerX, 2) + Math.pow(centerY - obj.centerY, 2));

                            if (distance < 50) {
                                isNewObject = false;
                                break;
                            }
                        }

                        if (isNewObject) {
                            newDetectedObjects.push({ centerX, centerY });
                            newCurrentCount++;
                        }

                        ctx.strokeStyle = 'red';
                        ctx.lineWidth = 3;
                        ctx.strokeRect(x, y, width, height);
                        ctx.font = '16px Arial';
                        ctx.fillStyle = 'red';
                        ctx.fillText(label + ' (' + Math.round(pred.score * 100) + '%)', x, y > 10 ? y - 5 : y + 15);
                    }
                });

                detectedObjects = newDetectedObjects;
                currentCount = newCurrentCount;

                document.getElementById('currentCount').innerText = currentCount;

                if (newCurrentCount > 0) {
                    totalDetected += newCurrentCount;
                    document.getElementById('totalCount').innerText = totalDetected;
                }
            });
        }

        requestAnimationFrame(draw);
    }

    draw();
}
