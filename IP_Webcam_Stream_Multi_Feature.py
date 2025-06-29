import cv2
from flask import Flask, Response, render_template_string, request
import threading
import time
import atexit
import numpy as np

app = Flask(__name__)

# Global variable to hold the camera object
# Using a lock to prevent race conditions if multiple clients access the camera simultaneously
camera = None
camera_lock = threading.Lock()

# --- Configuration for Object Detection (MobileNet-SSD) ---
# YOU MUST DOWNLOAD THESE FILES FOR OBJECT DETECTION TO WORK:
# 1. MobileNetSSD_deploy.prototxt
# 2. MobileNetSSD_deploy.caffemodel
# 3. coco.names (a text file with class names)
OBJ_PROTO_PATH = "MobileNetSSD_deploy.prototxt"
OBJ_MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
OBJ_CLASSES_FILE = "coco.names"

obj_net = None
OBJ_CLASSES = []
obj_model_load_error = False
OBJ_MIN_CONFIDENCE = 0.5

# --- Configuration for Age Detection ---
# YOU MUST DOWNLOAD THESE FILES FOR AGE DETECTION TO WORK:
# 1. deploy.prototxt (for face detection)
# 2. res10_300x300_ssd_iter_140000.caffemodel (for face detection)
# 3. age_deploy.prototxt (for age prediction)
# 4. age_net.caffemodel (for age prediction)
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

face_net = None
age_net = None
age_model_load_error = False
MODEL_MEAN_VALUES = (78.42628377603, 87.768914374447, 114.8958477461283)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
FACE_CONFIDENCE_THRESHOLD = 0.7

# --- Load all models at startup ---
def load_all_models():
    global obj_net, OBJ_CLASSES, obj_model_load_error
    global face_net, age_net, age_model_load_error

    # Load Object Detection Models
    try:
        obj_net = cv2.dnn.readNetFromCaffe(OBJ_PROTO_PATH, OBJ_MODEL_PATH)
        with open(OBJ_CLASSES_FILE, "r") as f:
            OBJ_CLASSES = [line.strip() for line in f.readlines()]
        print("Object Detection (MobileNet-SSD) models loaded successfully.")
    except Exception as e:
        obj_model_load_error = True
        print("\n" + "="*80)
        print("!!! WARNING: Object Detection Models Not Loaded !!!")
        print("This feature requires 3 pre-trained model files. Please download them:")
        print(f"1. MobileNetSSD_deploy.prototxt (e.g., https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt)")
        print(f"2. MobileNetSSD_deploy.caffemodel (e.g., http://www.kunalcho.net/files/MobileNetSSD_deploy.caffemodel)")
        print(f"3. coco.names (e.g., a text file with classes like: background, aeroplane, bottle, etc.)")
        print("Place ALL THREE files in the SAME directory as this Python script for object detection to work.")
        print(f"Specific error: {e}")
        print("="*80 + "\n")
        obj_net = None

    # Load Age Detection Models
    try:
        face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        print("Face and Age Detection DNN models loaded successfully.")
    except Exception as e:
        age_model_load_error = True
        print("\n" + "="*80)
        print("!!! WARNING: Age Detection Models Not Loaded !!!")
        print("This feature requires 4 pre-trained model files. Please download them:")
        print(f"1. Face Detector Proto: {FACE_PROTO} (https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)")
        print(f"2. Face Detector Model: {FACE_MODEL} (https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/res10_300x300_ssd_iter_140000.caffemodel)")
        print(f"3. Age Predictor Proto: {AGE_PROTO} (https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_deploy.prototxt)")
        print(f"4. Age Predictor Model: {AGE_MODEL} (https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/age_net.caffemodel)")
        print("Place ALL FOUR files in the SAME directory as this Python script for age detection to work.")
        print(f"Specific error: {e}")
        print("="*80 + "\n")
        face_net = None
        age_net = None

load_all_models() # Call to load models when the script starts

def get_camera():
    """
    Initializes and returns the camera object.
    Uses a lock to ensure thread-safe camera access.
    """
    global camera
    with camera_lock:
        if camera is None:
            try:
                # 0 typically refers to the default webcam.
                # You might need to change this number if you have multiple cameras.
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    raise IOError("Cannot open webcam. Please check if it's connected and not in use by another application.")
                print("Webcam initialized successfully. Starting stream...")
            except Exception as e:
                print(f"Error initializing camera: {e}")
                camera = None # Ensure camera is None on error, so generate_frames can show placeholder
        return camera

def release_camera():
    """
    Releases the camera object if it exists.
    Ensures the camera resource is freed when the application exits.
    """
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            print("Webcam released.")

# Register a function to release the camera when the Flask app shuts down.
# This is important to prevent the camera from being locked after the script exits.
atexit.register(release_camera)

def create_placeholder_image():
    """
    Creates and saves a simple JPEG image to display when the webcam is offline.
    This prevents broken image icons in the browser.
    """
    try:
        with open("webcam_offline_placeholder.jpg", "rb") as f:
            pass # File already exists
    except FileNotFoundError:
        # Create a simple grey image with "WEBCAM OFFLINE" text
        placeholder_frame = np.zeros((450, 600, 3), dtype=np.uint8) # Grey image (HxWxChannels)
        cv2.rectangle(placeholder_frame, (0, 0), (600, 450), (50, 50, 200), -1) # BGR: Dark Red

        text = "WEBCAM OFFLINE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_color = (255, 255, 255) # White text

        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (placeholder_frame.shape[1] - text_size[0]) // 2
        text_y = (placeholder_frame.shape[0] + text_size[1]) // 2
        cv2.putText(placeholder_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        camera_off_text = "🚫" # Emoji camera-off symbol
        cv2.putText(placeholder_frame, camera_off_text, (text_x + text_size[0] // 2 - 30, text_y - 80), font, 2.0, text_color, 2, cv2.LINE_AA)

        cv2.imwrite("webcam_offline_placeholder.jpg", placeholder_frame)
        print("Created 'webcam_offline_placeholder.jpg' for offline camera display.")


def generate_frames(mode="basic"):
    """
    Generator function to capture frames from the webcam, encode them as JPEG,
    and yield them as part of an MJPEG stream.
    Applies different computer vision logic based on the 'mode'.
    """
    create_placeholder_image()

    camera_instance = get_camera()

    while True:
        if camera_instance is None or not camera_instance.isOpened():
            try:
                with open("webcam_offline_placeholder.jpg", "rb") as f:
                    frame_bytes = f.read()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1)
            except FileNotFoundError:
                print("Error: Placeholder image not found.")
                break
            print("Attempting to re-initialize camera after a delay...")
            time.sleep(5)
            camera_instance = get_camera()
            continue

        success, frame = camera_instance.read()
        if not success:
            print("Failed to read frame from camera. Releasing and attempting to re-initialize...")
            release_camera()
            camera_instance = get_camera()
            continue

        h, w, _ = frame.shape

        if mode == "object_detection":
            if obj_net is not None and not obj_model_load_error:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                obj_net.setInput(blob)
                detections = obj_net.forward()
                
                detected_any_object = False
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > OBJ_MIN_CONFIDENCE:
                        idx = int(detections[0, 0, i, 1])
                        if idx < len(OBJ_CLASSES):
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            label = f"{OBJ_CLASSES[idx]}: {confidence:.2f}"
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            detected_any_object = True
                
                if detected_any_object:
                    cv2.putText(frame, "OBJECTS DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Object Detection Disabled (Model Error)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        
        elif mode == "age_detection":
            if face_net is not None and age_net is not None and not age_model_load_error:
                face_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=False)
                face_net.setInput(face_blob)
                face_detections = face_net.forward()

                detected_any_face = False
                for i in range(face_detections.shape[2]):
                    confidence = face_detections[0, 0, i, 2]
                    if confidence > FACE_CONFIDENCE_THRESHOLD:
                        detected_any_face = True
                        box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)

                        face_roi = frame[startY:endY, startX:endX]
                        if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                            age_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                            age_net.setInput(age_blob)
                            age_preds = age_net.forward()
                            
                            age = AGE_LIST[age_preds[0].argmax()]
                            
                            label = f"Age: {age}"
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
                            text_y = startY - 10 if startY - 10 > 10 else startY + 20
                            cv2.putText(frame, label, (startX, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                
                if detected_any_face:
                    cv2.putText(frame, "AGE DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Age Detection Disabled (Model Error)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        
        elif mode == "basic":
            cv2.putText(frame, "Basic Stream (No AI)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpeg', frame)
        if not ret:
            print("Failed to encode frame. Skipping frame.")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


# HTML template string for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IP Webcam Stream 📸</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .spinner-custom {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #fff;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .aspect-video {
            position: relative;
            width: 100%;
            padding-bottom: 75%; /* 4:3 aspect ratio */
            background-color: #000;
        }
        .aspect-video img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .toast-container-custom {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1060;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .toast-custom {
            background-color: #e0f7fa;
            border: 1px solid #00bcd4;
            color: #005f6b;
            padding: 15px 20px;
            border-radius: 0.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInOutToast 4s forwards;
        }
        .toast-custom.success {
            background-color: #d4edda;
            border: 1px solid #28a745;
            color: #155724;
        }
        .toast-custom.error {
            background-color: #f8d7da;
            border: 1px solid #dc3545;
            color: #721c24;
        }
        .toast-custom.warning {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
        }
        @keyframes fadeInOutToast {
            0% { opacity: 0; transform: translateY(20px); }
            10% { opacity: 1; transform: translateY(0); }
            90% { opacity: 1; transform: translateY(0); }
            100% { opacity: 0; transform: translateY(20px); }
        }

        .status-fade-out {
            animation: fadeOutStatus 3s forwards;
            animation-delay: 2s;
        }

        @keyframes fadeOutStatus {
            0% { opacity: 1; visibility: visible; }
            99% { opacity: 0; }
            100% { opacity: 0; visibility: hidden; }
        }
    </style>
</head>
<body class="bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center min-h-screen p-4 sm:p-6 md:p-8">
    <div class="bg-white p-6 md:p-8 rounded-xl shadow-2xl max-w-lg w-full text-center border-t-8 border-blue-600 transform hover:scale-105 transition-transform duration-300">
        <h1 class="text-3xl sm:text-4xl font-extrabold text-gray-800 mb-6 flex items-center justify-center gap-3">
            <span class="text-blue-600 text-5xl">📸</span> IP Webcam Stream
        </h1>

        <div id="videoContainer" class="relative w-full overflow-hidden rounded-lg border-4 border-gray-300 bg-black flex items-center justify-center aspect-video">
            <img id="flaskVideoStream" src="/video_feed?mode=basic" alt="Webcam Stream">
            <div id="statusMessage" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-white text-lg font-semibold bg-gray-800 bg-opacity-70 px-6 py-3 rounded-full flex items-center gap-3 transition-colors duration-300">
                <div class="spinner-custom"></div> Connecting...
            </div>
            <canvas id="hiddenCanvas" style="display: none;"></canvas>
        </div>

        <div class="mt-6 flex flex-col gap-4 justify-center">
            <div class="flex flex-col sm:flex-row gap-3 justify-center mb-4">
                <label for="streamMode" class="sr-only">Select Stream Mode</label>
                <select id="streamMode" class="form-select px-4 py-2 rounded-full shadow-sm text-gray-700 bg-white border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <option value="basic">Basic Stream (No AI)</option>
                    <option value="object_detection">Object Detection</option>
                    <option value="age_detection">Age Detection</option>
                </select>
                <button id="startStreamBtn" class="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 active:scale-95">
                    Start Stream
                </button>
            </div>

            <button id="takeSnapshotBtn" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed">
                📸 Take Snapshot
            </button>
            <button id="fullscreenBtn" class="bg-indigo-500 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 active:scale-95">
                📺 Full Screen
            </button>
        </div>

        <p class="text-sm text-gray-600 mt-6 leading-relaxed">
            Ensure your Python server is running and your webcam is accessible. If the stream doesn't appear, check your webcam connection or if another application is using it.
        </p>
        <p id="featureStatus" class="text-sm text-gray-600 mt-2 leading-relaxed">
            <span class="font-semibold text-green-600" id="currentModeDisplay">Current Mode: Basic Stream (No AI)</span>
            <br><span class="text-xs text-red-600" id="modelWarning"></span>
        </p>
        <p class="text-xs text-red-600 mt-4 font-semibold p-2 bg-red-100 rounded-md border border-red-300">
            ⚠️ **SECURITY WARNING:** This is a basic demonstration for local use. Do NOT expose this webcam stream to the public internet without implementing robust security measures like authentication, HTTPS encryption, and proper firewall rules. This application is intended for local network access only.
        </p>
    </div>

    <div id="toast-container" class="toast-container-custom"></div>

    <script>
        const flaskVideoStream = document.getElementById('flaskVideoStream');
        const videoContainer = document.getElementById('videoContainer');
        const statusMessage = document.getElementById('statusMessage');
        const takeSnapshotBtn = document.getElementById('takeSnapshotBtn');
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        const hiddenCanvas = document.getElementById('hiddenCanvas');
        const toastContainer = document.getElementById('toast-container');
        const streamModeSelect = document.getElementById('streamMode');
        const startStreamBtn = document.getElementById('startStreamBtn');
        const currentModeDisplay = document.getElementById('currentModeDisplay');
        const modelWarning = document.getElementById('modelWarning');

        // Python variable values injected by Flask's render_template_string
        const objModelLoadError = {{ 'true' if obj_model_load_error else 'false' }};
        const ageModelLoadError = {{ 'true' if age_model_load_error else 'false' }};

        let currentMode = streamModeSelect.value; // Initial mode

        // Helper to display toast messages
        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast-custom ${type}`;
            toast.innerHTML = message;
            toastContainer.appendChild(toast);

            setTimeout(() => {
                toast.remove();
            }, 4000);
        }

        // Function to update status message with icons and colors
        function updateStatus(message, type = 'connecting') {
            statusMessage.innerHTML = '';
            let iconHtml = '';
            let bgColorClass = '';
            let fadeOutClass = '';

            statusMessage.style.visibility = 'visible';
            statusMessage.style.opacity = '1';
            statusMessage.classList.remove('status-fade-out');

            if (type === 'connecting') {
                iconHtml = '<div class="spinner-custom"></div>';
                bgColorClass = 'bg-gray-800 bg-opacity-70';
            } else if (type === 'streaming') {
                iconHtml = '✨';
                bgColorClass = 'bg-green-600 bg-opacity-80';
                fadeOutClass = 'status-fade-out';
            } else if (type === 'error') {
                iconHtml = '🚨';
                bgColorClass = 'bg-red-600 bg-opacity-80';
            }

            statusMessage.innerHTML = iconHtml + ' ' + message;
            statusMessage.className = `absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-white text-lg font-semibold px-6 py-3 rounded-full flex items-center gap-3 transition-colors duration-300 ${bgColorClass} ${fadeOutClass}`;

            if (takeSnapshotBtn) {
                takeSnapshotBtn.disabled = (type !== 'streaming');
            }
        }

        // Function to update the feature status text and warning based on mode
        function updateFeatureStatus(mode) {
            let statusText = "";
            let warningText = "";
            let textColorClass = "text-green-600";

            if (mode === "basic") {
                statusText = "Current Mode: Basic Stream (No AI)";
                warningText = "";
                textColorClass = "text-green-600";
            } else if (mode === "object_detection") {
                statusText = "Current Mode: Object Detection Active (Objects outlined in blue)";
                textColorClass = "text-blue-600";
                if (objModelLoadError === 'true') {
                    warningText = "Object Detection Disabled: Model files missing/invalid. Check server console for details.";
                    textColorClass = "text-red-600";
                }
            } else if (mode === "age_detection") {
                statusText = "Current Mode: Age Detection Active (Faces outlined in yellow with age)";
                textColorClass = "text-yellow-600";
                if (ageModelLoadError === 'true') {
                    warningText = "Age Detection Disabled: Model files missing/invalid. Check server console for details.";
                    textColorClass = "text-red-600";
                }
            }
            currentModeDisplay.textContent = statusText;
            currentModeDisplay.className = `font-semibold ${textColorClass}`;
            modelWarning.textContent = warningText;
        }

        // Initial status display and feature status update
        updateStatus('Connecting to webcam...', 'connecting');
        updateFeatureStatus(currentMode);


        flaskVideoStream.onload = function() {
            updateStatus('Streaming live...', 'streaming');
            // If the stream loads, and there was a model error, update the specific warning
            updateFeatureStatus(currentMode);
        };

        flaskVideoStream.onerror = function() {
            updateStatus('Stream failed. Retrying in 3s...', 'error');
            flaskVideoStream.src = "/webcam_offline_placeholder.jpg"; // Show placeholder if IP stream fails
            
            // Only show toast and persistent warning if the initial error was related to camera,
            // otherwise, the Python console already reported model errors.
            showToast('Webcam stream failed. Retrying...', 'error');

            setTimeout(() => {
                // Ensure to load the stream with the currently selected mode
                flaskVideoStream.src = `/video_feed?mode=${currentMode}&_t=${new Date().getTime()}`;
                updateStatus('Connecting to webcam...', 'connecting');
            }, 3000);
        };

        // Handle Start Stream button click
        startStreamBtn.addEventListener('click', () => {
            currentMode = streamModeSelect.value;
            updateStatus('Starting new stream...', 'connecting');
            updateFeatureStatus(currentMode);
            // Append timestamp to prevent caching issues when changing modes
            flaskVideoStream.src = `/video_feed?mode=${currentMode}&_t=${new Date().getTime()}`;
        });

        // Snapshot functionality
        if (takeSnapshotBtn) {
            takeSnapshotBtn.addEventListener('click', function() {
                if (flaskVideoStream.naturalWidth === 0 || flaskVideoStream.naturalHeight === 0) {
                    showToast('Webcam stream not active to take snapshot!', 'error');
                    return;
                }

                hiddenCanvas.width = flaskVideoStream.naturalWidth;
                hiddenCanvas.height = flaskVideoStream.naturalHeight;
                const context = hiddenCanvas.getContext('2d');

                context.drawImage(flaskVideoStream, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

                const dataURL = hiddenCanvas.toDataURL('image/png');

                const a = document.createElement('a');
                a.href = dataURL;
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
                a.download = `webcam-snapshot-${timestamp}.png`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);

                showToast('Snapshot saved successfully! ✨', 'success');
            });
        }

        // Fullscreen functionality
        if (fullscreenBtn) {
            fullscreenBtn.addEventListener('click', () => {
                if (videoContainer.requestFullscreen) {
                    videoContainer.requestFullscreen();
                } else if (videoContainer.mozRequestFullScreen) {
                    videoContainer.mozRequestFullScreen();
                } else if (videoContainer.webkitRequestFullscreen) {
                    videoContainer.webkitRequestFullscreen();
                } else if (videoContainer.msRequestFullscreen) {
                    videoContainer.msRequestFullscreen();
                }
            });
        }

        window.onload = function() {
            // Initial load of the stream based on default selected mode
            flaskVideoStream.src = `/video_feed?mode=${currentMode}&_t=${new Date().getTime()}`;
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """
    Renders the main HTML page for the webcam stream.
    Passes model load error flags to the frontend.
    """
    create_placeholder_image()
    # Pass the Python model error variables directly to the Jinja2 template
    return render_template_string(HTML_TEMPLATE, 
                                  obj_model_load_error=obj_model_load_error,
                                  age_model_load_error=age_model_load_error)

@app.route('/webcam_offline_placeholder.jpg')
def serve_placeholder():
    """Serves the placeholder image if the webcam is offline."""
    create_placeholder_image()
    try:
        return app.send_static_file('webcam_offline_placeholder.jpg')
    except FileNotFoundError:
        return "Placeholder image not found on server.", 404


@app.route('/video_feed')
def video_feed():
    """
    Streams webcam frames as an MJPEG multipart response,
    applying different CV logic based on the 'mode' query parameter.
    """
    mode = request.args.get('mode', 'basic') # Get mode from query param, default to 'basic'
    return Response(generate_frames(mode), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("IP Webcam Stream Application Starting...")
    print("---------------------------------------")
    print("Initializing webcam (this might take a moment)...")
    create_placeholder_image()

    print("Flask server will start on: http://127.0.0.1:5000")
    print("Open this URL in your web browser to view the stream.")
    print("Press Ctrl+C in this terminal to stop the server and release the webcam.")
    print("---------------------------------------")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        release_camera()
