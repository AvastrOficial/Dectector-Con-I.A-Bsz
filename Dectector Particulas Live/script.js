  let video = document.getElementById('video');
    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    let currentCountText = document.getElementById('currentCount');
    let totalCountText = document.getElementById('totalCount');
    let statusText = document.getElementById('status');

    let totalCount = parseInt(localStorage.getItem("totalDetections") || "0");
    let processingStarted = false;
    let model;

    // 🧠 Crear modelo "neuronal" simple
    async function createModel() {
      model = tf.sequential();
      model.add(tf.layers.dense({inputShape: [1], units: 4, activation: 'relu'}));
      model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
      model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
      statusText.textContent = 'Modelo cargado y listo ✅';
    }

    async function trainModel(data) {
      const xs = tf.tensor2d(data.map((v, i) => [i]));
      const ys = tf.tensor2d(data.map(v => [v / 10]));  // Normalización simple
      await model.fit(xs, ys, {epochs: 10});
      localStorage.setItem("modelData", JSON.stringify(data));
      statusText.textContent = 'Modelo entrenado 🧠';
    }

    function onOpenCvReady() {
      navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
        video.onloadeddata = () => {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          createModel().then(() => {
            if (!processingStarted) startProcessing();
            processingStarted = true;
          });
        };
      });
    }

    function startProcessing() {
      const cap = new cv.VideoCapture(video);
      let frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
      let gray = new cv.Mat();
      let fgMask = new cv.Mat();
      let blurred = new cv.Mat();
      const fgbg = new cv.BackgroundSubtractorMOG2(500, 16, true);

      let detections = [];

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
        localStorage.setItem("totalDetections", totalCount.toString());

        detections.push(currentCount);
        if (detections.length > 10) {
          trainModel(detections);
          detections = [];
        }

        hierarchy.delete();
        contours.delete();
        requestAnimationFrame(processFrame);
      }

      requestAnimationFrame(processFrame);
    }
