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
const uploadToggle = document.getElementById("uploadToggle");
const uploadPanel = document.getElementById("uploadPanel");

uploadToggle.addEventListener("click", () => {
  uploadPanel.hidden = !uploadPanel.hidden;
  uploadToggle.textContent = uploadPanel.hidden ? "Upload video" : "Hide video";

  if (uploadPanel.hidden) {
    uploadedVideo.pause();  
  } else {
    uploadPanel.scrollIntoView({ behavior: "smooth" });
  }
});

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

      video.addEventListener("loadeddata", () => {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
        predictWebcam();
      }, { once: true });
    })
    .catch(error => {
      console.error("Could not access webcam:", error);
      webcamRunning = false;
      enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    });
}


let lastVideoTime = -1
async function predictWebcam() {

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
        
        if (wristsAreVisible(landmark)) {
          saveWristValue(wrist15History, landmark[15].y);
          saveWristValue(wrist16History, landmark[16].y);

          let wristY = (landmark[15].y + landmark[16].y) / 2;
          updateScale(landmark, video);
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
        for (const landmark of result.landmarks) {
        if (wristsAreVisible(landmark)) {
          saveWristValue(wrist15History, landmark[15].y);
          saveWristValue(wrist16History, landmark[16].y);

          let wristY = (landmark[15].y + landmark[16].y) / 2;
          updateScale(landmark, uploadedVideo);
          countCompression(wristY);
          drawLiveGraph();
        } else {
          resetCompressionTracking();
        }

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
    recentDepths.length = 0;
}
function countCompression(y, fromWebcam = false) {
  if (topY === null) topY = y;

  if (!movingDown) {
    topY = Math.min(topY, y);

    if (y > topY + getThreshold()) {
      movingDown = true;
      bottomY = y;
    }
  } else {
    bottomY = Math.max(bottomY, y);

    if (y < bottomY - getThreshold()) {
      compression++;
      showDepth(bottomY - topY);
      const reminder = document.getElementById("breathReminder");

      if (reminder) {
        const remaining = 30 - (compression % 30);

        if (compression % 30 === 0) {
          reminder.textContent = "If trained, give 2 breaths now!";
        } else {
          reminder.textContent =
            remaining + " compressions until the next cycle.";
        }
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
          feedback.style.color = "green";
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

let armLength = null;
try { armLength = parseFloat(localStorage.getItem("armLengthIn")) || null; } catch (e) {}

const armInput = document.getElementById("armInput");
const armStatus = document.getElementById("armStatus");
const depthBox = document.getElementById("depthBox");

function showArm() {
  armStatus.textContent = armLength
    ? "Saved: " + armLength + " in. Depth gauge is on."
    : "Not set. Everything works, just without the depth gauge.";
  depthBox.hidden = !armLength;
  if (armLength) armInput.value = armLength;
}
showArm();

document.getElementById("saveArm").addEventListener("click", () => {
  const value = parseFloat(armInput.value);
  if (armInput.value === "") {
    armLength = null;                       
  } else if (value >= 15 && value <= 32) {
    armLength = value;
  } else {
    armStatus.textContent = "Enter a number from 15 to 32.";
    return;
  }
  try {
    if (armLength) localStorage.setItem("armLengthIn", armLength);
    else localStorage.removeItem("armLengthIn");
  } catch (e) {}
  showArm();
});


document.getElementById("emergencyBtn").addEventListener("click", () => {
  document.getElementById("homeScreen").hidden = true;
  document.getElementById("placeScreen").hidden = false;
});

document.getElementById("startBtn").addEventListener("click", () => {
  document.getElementById("placeScreen").hidden = true;
  demosSection.hidden = false;
  enableCam();                               
});

let pixelsPerInch = null;
let frameHeight = null;
const armHistory = [];
const recentDepths = [];

function updateScale(landmark, videoEl) {
  if (!armLength) return;

  const leftVis = (landmark[11].visibility ?? 0) + (landmark[15].visibility ?? 0);
  const rightVis = (landmark[12].visibility ?? 0) + (landmark[16].visibility ?? 0);
  const shoulder = leftVis > rightVis ? landmark[11] : landmark[12];
  const wrist = leftVis > rightVis ? landmark[15] : landmark[16];


  const armPx = Math.hypot(
    (shoulder.x - wrist.x) * videoEl.videoWidth,
    (shoulder.y - wrist.y) * videoEl.videoHeight
  );

  armHistory.push(armPx);
  if (armHistory.length > 90) armHistory.shift();   // about the last 3 seconds

  pixelsPerInch = Math.max(...armHistory) / armLength;
  frameHeight = videoEl.videoHeight;
}


function showDepth(travel) {
  if (!armLength || !pixelsPerInch) return;

  recentDepths.push((travel * frameHeight) / pixelsPerInch);
  if (recentDepths.length > 3) recentDepths.shift();
  const average = recentDepths.reduce((a, b) => a + b, 0) / recentDepths.length;

  const el = document.getElementById("depthFeedback");
  if (average < 1.8) {
    el.textContent = "Push harder";
    el.style.color = "red";
  } else if (average > 2.6) {
    el.textContent = "Ease up";
    el.style.color = "orange";
  } else {
    el.textContent = "Good depth";
    el.style.color = "green";
  }
}

function getThreshold() {
  if (pixelsPerInch && frameHeight) {
    return (0.7 * pixelsPerInch) / frameHeight;
  }
  return 0.015;
}
