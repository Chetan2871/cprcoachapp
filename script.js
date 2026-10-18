// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://esm.sh/@mediapipe/tasks-vision"

const demosSection = document.getElementById("demos")

let poseLandmarker = undefined
let runningMode = "IMAGE"
let enableWebcamButton
let webcamRunning = false
const videoHeight = "360px"
const videoWidth = "480px"

// Before we can use PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createPoseLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
  )
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numPoses: 1
  })
  demosSection.classList.remove("invisible")
}
createPoseLandmarker()

/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/

// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.
const imageContainers = document.getElementsByClassName("detectOnClick")
const wrist15History = [];
const wrist16History = [];
const MAX_POINTS = 150;
    const graphCanvas = document.getElementById("wristGraph");
    const graphCtx = graphCanvas.getContext("2d");

    function saveWristValue(history, value) {
      history.push(value);

      // remove oldest value when more than 150
      if (history.length > MAX_POINTS) {
        history.shift();
      }
    }

    function drawLiveGraph() {
      const values = [...wrist15History, ...wrist16History];
      if (values.length < 2) return;

      let minY = Math.min(...values);
      let maxY = Math.max(...values);

      if (maxY - minY < 0.005) {
        minY -= 0.005;
        maxY += 0.005;
      }

      const padding = (maxY - minY) * 0.1;
      minY -= padding;
      maxY += padding;

      graphCtx.clearRect(
        0, 0, graphCanvas.width, graphCanvas.height
      );

      function drawLine(history, color) {
        if (history.length < 2) return;

        graphCtx.beginPath();
        graphCtx.strokeStyle = color;
        graphCtx.lineWidth = 2;

        history.forEach((value, index) => {
          const x =
            (index / (history.length - 1)) * graphCanvas.width;

          const y =
            ((value - minY) / (maxY - minY)) *
            graphCanvas.height;

          if (index === 0) {
            graphCtx.moveTo(x, y);
          } else {
            graphCtx.lineTo(x, y);
          }
        });

        graphCtx.stroke();
      }

      drawLine(wrist15History, "blue");
      drawLine(wrist16History, "orange");
    }

// Now let's go through all of these and add a click event listener.
for (let i = 0; i < imageContainers.length; i++) {
  // Add event listener to the child element whichis the img element.
  imageContainers[i].children[0].addEventListener("click", handleClick)
}

const myNumbers=[5,9,10,5];
function avg(myNumbers){
  let sum=0;
  for(let i of myNumbers){
    sum+=i;
  }
  return sum/(myNumbers.length);
}
console.log(avg(myNumbers));
// When an image is clicked, let's detect it and display results!
async function handleClick(event) {
  if (!poseLandmarker) {
    console.log("Wait for poseLandmarker to load before clicking!")
    return
  }

  if (runningMode === "VIDEO") {
    runningMode = "IMAGE"
    await poseLandmarker.setOptions({ runningMode: "IMAGE" })
  }
  // Remove all landmarks drawed before
  const allCanvas = event.target.parentNode.getElementsByClassName("canvas")
  for (var i = allCanvas.length - 1; i >= 0; i--) {
    const n = allCanvas[i]
    n.parentNode.removeChild(n)
  }

  // We can call poseLandmarker.detect as many times as we like with
  // different image data each time. The result is returned in a callback.
  poseLandmarker.detect(event.target, result => {
    const canvas = document.createElement("canvas")
    canvas.setAttribute("class", "canvas")
    canvas.setAttribute("width", event.target.naturalWidth + "px")
    canvas.setAttribute("height", event.target.naturalHeight + "px")
    canvas.style =
      "left: 0px;" +
      "top: 0px;" +
      "width: " +
      event.target.width +
      "px;" +
      "height: " +
      event.target.height +
      "px;"

    event.target.parentNode.appendChild(canvas)
    const canvasCtx = canvas.getContext("2d")
    const drawingUtils = new DrawingUtils(canvasCtx)
    for (const landmark of result.landmarks) {
      drawingUtils.drawLandmarks(landmark, {
        radius: data => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
      })
      drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS)
    }
  })
}

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")
const drawingUtils = new DrawingUtils(canvasCtx)

const videoFile = document.getElementById("videoFile")
const uploadedVideo = document.getElementById("uploadedVideo");

uploadedVideo.addEventListener("play", startMetronome);
uploadedVideo.addEventListener("pause", stopMetronome);
uploadedVideo.addEventListener("ended", stopMetronome);
const videoCanvas = document.getElementById("videoCanvas")
const videoCanvasCtx = videoCanvas.getContext("2d")
const videoDrawingUtils = new DrawingUtils(videoCanvasCtx)
const wristData = []

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton")
  enableWebcamButton.addEventListener("click", enableCam)
} else {
  console.warn("getUserMedia() is not supported by your browser")
}

videoFile.addEventListener("change", (event) => {

  const file = event.target.files[0]

  if (!file) {
    return
  }

  const videoURL = URL.createObjectURL(file)

  uploadedVideo.src = videoURL

  uploadedVideo.play()
})

// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!poseLandmarker) {
    console.log("Wait! poseLandmarker not loaded yet.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    stopMetronome();
    stopWebcamReminderTimer();

    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null;
    }

    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    return;
  }
    // getUsermedia parameters
  const constraints = {
    video: true
  };

  navigator.mediaDevices.getUserMedia(constraints)
    .then(stream => {
      video.srcObject = stream;
      webcamRunning = true;

      enableWebcamButton.innerText = "DISABLE PREDICTIONS";
      startMetronome();

      video.addEventListener("loadeddata", predictWebcam, {
        once: true
      });
    })
    .catch(error => {
      console.error("Could not access webcam:", error);
      webcamRunning = false;
      enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    });
}


let lastVideoTime = -1
async function predictWebcam() {
  canvasElement.style.height = videoHeight
  video.style.height = videoHeight
  canvasElement.style.width = videoWidth
  video.style.width = videoWidth
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO"
    await poseLandmarker.setOptions({ runningMode: "VIDEO" })
  }
  let startTimeMs = performance.now()
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime
    poseLandmarker.detectForVideo(video, startTimeMs, result => {
      canvasCtx.save()
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height)
      for (const landmark of result.landmarks) {
        console.log(landmark[15].y);
        if (wristsAreVisible(landmark)) {
          saveWristValue(wrist15History, landmark[15].y);
          saveWristValue(wrist16History, landmark[16].y);

          let wristY = (landmark[15].y + landmark[16].y) / 2;

          countCompression(wristY, true);
          drawLiveGraph();
        } else {
          resetCompressionTracking();
        }
        drawingUtils.drawLandmarks(landmark, {
          radius: data => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
        })
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS)
      }
      canvasCtx.restore()
    })
  }

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam)
  }
}

let lastUploadedVideoTime = -1
let uploadedLoopRunning = false

uploadedVideo.addEventListener("play", () => {
  if (!poseLandmarker) {
    console.log("Model not loaded yet")
    return
  }
  if (uploadedLoopRunning) return
  uploadedLoopRunning = true
  predictUploadedVideo()
})
uploadedVideo.addEventListener("loadedmetadata", () => {
  videoCanvas.width = uploadedVideo.videoWidth
  videoCanvas.height = uploadedVideo.videoHeight
  console.log("intrinsic:", uploadedVideo.videoWidth, uploadedVideo.videoHeight)
  console.log("displayed:", uploadedVideo.clientWidth, uploadedVideo.clientHeight)
})
uploadedVideo.addEventListener("ended", () => { uploadedLoopRunning = false })

async function predictUploadedVideo() {
  try {
    if (runningMode === "IMAGE") {
      runningMode = "VIDEO"
      await poseLandmarker.setOptions({ runningMode: "VIDEO" })
    }
    


    // Only detect once the video has a decoded frame
    if (
      uploadedVideo.readyState >= 2 &&
      uploadedVideo.videoWidth > 0 &&
      lastUploadedVideoTime !== uploadedVideo.currentTime
    ) {
      lastUploadedVideoTime = uploadedVideo.currentTime
      poseLandmarker.detectForVideo(uploadedVideo, uploadedVideo.currentTime * 1000, result => {
        videoCanvasCtx.save()
        videoCanvasCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height)
        console.log("poses found:", result.landmarks.length)
        for (const landmark of result.landmarks) {
        if (wristsAreVisible(landmark)) {
          saveWristValue(wrist15History, landmark[15].y);
          saveWristValue(wrist16History, landmark[16].y);

          let wristY = (landmark[15].y + landmark[16].y) / 2;

          countCompression(wristY);
          drawLiveGraph();
        } else {
          resetCompressionTracking();
        }
          console.log(
            "Graph samples:",
            wrist15History.length,
            wrist16History.length
          );
          videoDrawingUtils.drawLandmarks(landmark, {
            radius: d => DrawingUtils.lerp(d.from.z, -0.15, 0.1, 5, 1)
          })
          videoDrawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS)
        }
        videoCanvasCtx.restore()
      })
    }
  } catch (err) {
    console.error("Detection error:", err)
  }
  if (!uploadedVideo.paused && !uploadedVideo.ended) {
    if (uploadedVideo.requestVideoFrameCallback) {
      uploadedVideo.requestVideoFrameCallback(() => predictUploadedVideo())
    } else {
      window.requestAnimationFrame(predictUploadedVideo)
    }
  } else {
    uploadedLoopRunning = false
  }
}
let compression = 0;
let topY = null;
let bottomY = null;
let movingDown = false;
let lastCompressionTime=null;
let rate=null;
function wristsAreVisible(landmark) {
  const left = landmark[15];
  const right = landmark[16];

  return [left, right].every(w =>
    w &&
    (w.visibility ?? 0) > 0.5 &&
    w.x >= 0 && w.x <= 1 &&
    w.y >= 0 && w.y <= 1
  );
}

function resetCompressionTracking() {
  topY = null;
  bottomY = null;
  movingDown = false;
  lastCompressionTime = null;

  document.getElementById("compressionRate").textContent = "--";
}
function countCompression(y, fromWebcam = false) {
  if (topY === null) topY = y;

  if (!movingDown) {
    topY = Math.min(topY, y);

    if (y > topY + 0.025) {
      movingDown = true;
      bottomY = y;
    }
  } else {
    bottomY = Math.max(bottomY, y);

    if (y < bottomY - 0.025) {
      compression++;
      const remaining = 30 - (compression % 30);

      if (compression % 30 === 0) {
        document.getElementById("breathReminder").textContent =
          "If trained, give 2 breaths now!";
      } else {
        document.getElementById("breathReminder").textContent =
          remaining + " compressions until the next cycle.";
      }
      if (fromWebcam && webcamReminderTimer === null) {
        startWebcamReminderTimer();
      }
      document.getElementById("compressionCount").textContent = compression;
      const currentTime = Date.now();
      if (lastCompressionTime!== null){
        rate=60000/(currentTime-lastCompressionTime);
        document.getElementById("compressionRate").textContent =
         Math.round(rate);
          const feedback = document.getElementById("rateFeedback");

        if (rate < 100) {
          feedback.textContent = "Push faster!";
          feedback.style.color = "red";
        } else if (rate > 120) {
          feedback.textContent = "Slow down!";
          feedback.style.color = "red";
        } else {
          feedback.textContent = "Rate is good!";
          feedback.style.color = "blue";
        }
      }
      lastCompressionTime = currentTime;
      movingDown = false;
      topY = y;
    }
  }
}
function startMetronome() {
  const sound = document.getElementById("metronome");
  sound.currentTime = 0;

  sound.play().catch(error => {
    console.log("Could not play metronome:", error);
  });
}

function stopMetronome() {
  const sound = document.getElementById("metronome");
  sound.pause();
  sound.currentTime = 0;
}
let webcamReminderTimer = null;

function playCompressionReminder() {
  const sound = document.getElementById("compressionReminder");
  sound.currentTime = 0;

  sound.play().catch(error => {
    console.log("Could not play reminder:", error);
  });
}

function startWebcamReminderTimer() {
  if (webcamReminderTimer !== null) return;

  webcamReminderTimer = setInterval(
    playCompressionReminder,
    2*60* 1000
  );
}

function stopWebcamReminderTimer() {
  if (webcamReminderTimer !== null) {
    clearInterval(webcamReminderTimer);
  }

  webcamReminderTimer = null;

  const sound = document.getElementById("compressionReminder");
  sound.pause();
  sound.currentTime = 0;
}

