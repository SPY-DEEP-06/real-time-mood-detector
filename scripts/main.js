const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const moodDisplay = document.getElementById('mood-display');
const loadingMessage = document.getElementById('loading-message');
let detectionInterval;

const expressionHistory = [];
const HISTORY_LENGTH = 10;

async function loadModels() {
    const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';
    try {
        await Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
            faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
            faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
        ]);
        startVideo();
    } catch (e) {
        console.error("Error loading models: ", e);
        loadingMessage.innerText = 'Error loading models. Please refresh.';
    }
}

async function startVideo() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        console.error('Error accessing webcam:', err);
        loadingMessage.innerText = 'Error: Could not access webcam.';
    }
}

video.addEventListener('playing', () => {
    loadingMessage.style.display = 'none';

    const displaySize = { width: video.clientWidth, height: video.clientHeight };
    faceapi.matchDimensions(canvas, displaySize);

    if (detectionInterval) {
        clearInterval(detectionInterval);
    }

    detectionInterval = setInterval(async () => {
        const detections = await faceapi.detectAllFaces(
            video,
            new faceapi.SsdMobilenetv1Options({ minConfidence: 0.6 })
        ).withFaceLandmarks().withFaceExpressions();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

        if (resizedDetections.length > 0) {
            const expressions = resizedDetections[0].expressions;

            expressionHistory.push(expressions);
            if (expressionHistory.length > HISTORY_LENGTH) {
                expressionHistory.shift();
            }

            const avgExpressions = expressionHistory.reduce((avg, exp) => {
                for (const key in exp) {
                    avg[key] = (avg[key] || 0) + exp[key];
                }
                return avg;
            }, {});

            for (const key in avgExpressions) {
                avgExpressions[key] /= expressionHistory.length;
            }

            let dominantEmotion = 'neutral';
            let maxScore = 0;
            for (const [emotion, score] of Object.entries(avgExpressions)) {
                if (score > maxScore) {
                    maxScore = score;
                    dominantEmotion = emotion;
                }
            }

            moodDisplay.innerText = dominantEmotion;

            resizedDetections.forEach(result => {
                const { detection } = result;
                const box = detection.box;
                const text = `${dominantEmotion} (${Math.round(maxScore * 100)}%)`;
                const drawBox = new faceapi.draw.DrawBox(box, {
                    label: text,
                    boxColor: 'rgba(76, 175, 80, 1)'
                });
                drawBox.draw(canvas);
            });

        } else {
            moodDisplay.innerText = 'No Face Detected';
            expressionHistory.length = 0;
        }
    }, 200);
});

loadModels();
