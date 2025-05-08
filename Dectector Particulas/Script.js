let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let currentCountText = document.getElementById('currentCount');
let totalCountText = document.getElementById('totalCount');

let totalCount = 0;
let processingStarted = false;

function onOpenCvReady() {
    console.log('OpenCV.js cargado ✅');
    video.addEventListener('loadeddata', () => {
        video.play();
        if (!processingStarted) {
            startProcessing();
            processingStarted = true;
        }
    });
}

function startProcessing() {
    const cap = new cv.VideoCapture(video);
    let frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let gray = new cv.Mat();
    let fgMask = new cv.Mat();
    let blurred = new cv.Mat();
    const fgbg = new cv.BackgroundSubtractorMOG2(500, 16, true);

    function processFrame() {
        if (video.paused || video.ended) {
            requestAnimationFrame(processFrame);
            return;
        }

        cap.read(frame);
        cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);
        cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
        fgbg.apply(blurred, fgMask);

        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(fgMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'lime';
        ctx.lineWidth = 1;

        let currentCount = 0;
        for (let i = 0; i < contours.size(); ++i) {
            let cnt = contours.get(i);
            let area = cv.contourArea(cnt);
            if (area > 5 && area < 200) {
                let rect = cv.boundingRect(cnt);
                ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
                currentCount++;
                totalCount++;
            }
            cnt.delete();
        }

        currentCountText.textContent = currentCount;
        totalCountText.textContent = totalCount;

        hierarchy.delete();
        contours.delete();

        requestAnimationFrame(processFrame);
    }

    requestAnimationFrame(processFrame);
}

// SUBIR A LITTERBOX
document.getElementById('fileInput').addEventListener('change', function(e) {
    let file = e.target.files[0];
    if (!file) return;

    let formData = new FormData();
    formData.append('reqtype', 'fileupload');
    formData.append('time', '1h');  // 1 día de duración
    formData.append('fileToUpload', file);

    alert('Subiendo archivo a Litterbox... espera unos segundos.');

    fetch('https://litterbox.catbox.moe/resources/internals/api.php', {
        method: 'POST',
        body: formData
    })
    .then(res => res.text())
    .then(url => {
        alert('Subido correctamente ✅\nURL: ' + url);
        changeVideoSource(url);
    })
    .catch(err => {
        alert('Error al subir: ' + err);
    });
});

function changeVideoSource(newUrl) {
    video.pause();
    let source = video.querySelector('source');
    source.src = newUrl;
    video.load();
    video.play();
}
