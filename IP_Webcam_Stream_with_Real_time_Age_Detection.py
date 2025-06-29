import cv2
from flask import Flask, Response, render_template_string
import threading
import time
import atexit
import numpy as np

app = Flask(__name__)

# Global variable to hold the camera object
# Using a lock to prevent race conditions if multiple clients access the camera simultaneously
camera = None
camera_lock = threading.Lock()

# --- Global variables for Age Detection Models ---
# Paths to pre-trained Caffe models for face detection and age prediction
# ========================================================================
# !!! IMPORTANT: YOU MUST DOWNLOAD THESE FOUR FILES !!!
# AND PLACE THEM IN THE SAME DIRECTORY AS THIS PYTHON SCRIPT.
#
# 1. Face Detector:
#    - deploy.prototxt: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
#    - res10_300x300_ssd_iter_140000.caffemodel: https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/res10_300x300_ssd_iter_140000.caffemodel
#
# 2. Age Predictor:
#    - age_deploy.prototxt: https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_deploy.prototxt
#    - age_net.caffemodel: https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/age_net.caffemodel
# ========================================================================

FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

# Mean values for normalizing input images (standard for these Caffe models)
MODEL_MEAN_VALUES = (78.42628377603, 87.768914374447, 114.8958477461283)

# Defined age ranges that the model is trained to predict
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load the DNN models (face detector and age predictor)
face_net = None
age_net = None
age_model_load_error = False # Flag to indicate if model loading failed

try:
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    print("Face and Age Detection DNN models loaded successfully.")
except Exception as e:
    age_model_load_error = True
    print("\n" + "="*80)
    print("!!! ERROR: Age Detection Models Not Loaded !!!")
    print("This feature requires 4 pre-trained model files. Please download them:")
    print(f"1. Face Detector Proto: {FACE_PROTO} (from GitHub link provided in code comments)")
    print(f"2. Face Detector Model: {FACE_MODEL} (from GitHub link provided in code comments)")
    print(f"3. Age Predictor Proto: {AGE_PROTO} (from GitHub link provided in code comments)")
    print(f"4. Age Predictor Model: {AGE_MODEL} (from GitHub link provided in code comments)")
    print("Place ALL FOUR files in the SAME directory as this Python script.")
    print(f"Specific error: {e}")
    print("="*80 + "\n")
    face_net = None
    age_net = None

# Confidence threshold for face detections
FACE_CONFIDENCE_THRESHOLD = 0.7


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
        # Add a red background rectangle for emphasis
        cv2.rectangle(placeholder_frame, (0, 0), (600, 450), (50, 50, 200), -1) # BGR: Dark Red

        text = "WEBCAM OFFLINE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_color = (255, 255, 255) # White text

        # Get text size to center it
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (placeholder_frame.shape[1] - text_size[0]) // 2
        text_y = (placeholder_frame.shape[0] + text_size[1]) // 2
        cv2.putText(placeholder_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Add a camera-off icon
        camera_off_text = "🚫" # Emoji camera-off symbol
        cv2.putText(placeholder_frame, camera_off_text, (text_x + text_size[0] // 2 - 30, text_y - 80), font, 2.0, text_color, 2, cv2.LINE_AA)


        cv2.imwrite("webcam_offline_placeholder.jpg", placeholder_frame)
        print("Created 'webcam_offline_placeholder.jpg' for offline camera display.")


def generate_frames():
    """
    Generator function to capture frames from the webcam, encode them as JPEG,
    and yield them as part of an MJPEG stream.
    Includes Real-time Age Detection logic.
    If the camera is not available, it serves a placeholder image.
    """
    create_placeholder_image()

    camera_instance = get_camera()

    while True:
        if camera_instance is None or not camera_instance.isOpened():
            # If camera is not available, serve the placeholder image
            try:
                with open("webcam_offline_placeholder.jpg", "rb") as f:
                    frame_bytes = f.read()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1) # Serve placeholder at a slower rate
            except FileNotFoundError:
                print("Error: Placeholder image not found.")
                break # Cannot serve anything, so break
            # Try to re-initialize camera after a delay if it was disconnected
            print("Attempting to re-initialize camera after a delay...")
            time.sleep(5) # Wait before trying to re-initialize
            camera_instance = get_camera()
            continue # Go back to the start of the loop

        success, frame = camera_instance.read()
        if not success:
            print("Failed to read frame from camera. Releasing and attempting to re-initialize...")
            release_camera() # Release current broken camera
            camera_instance = get_camera() # Try to get a new one
            continue # Go back to the start of the loop

        # --- Real-time Age Detection Logic ---
        if face_net is not None and age_net is not None and not age_model_load_error: # Check the flag
            h, w, _ = frame.shape
            
            # Prepare blob for face detection
            face_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=False)
            face_net.setInput(face_blob)
            face_detections = face_net.forward()

            age_detected_in_frame = False

            for i in range(face_detections.shape[2]):
                confidence = face_detections[0, 0, i, 2]
                if confidence > FACE_CONFIDENCE_THRESHOLD:
                    # Face detected
                    age_detected_in_frame = True
                    box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Ensure bounding box is within frame dimensions
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, h) # Ensure endY does not exceed frame height

                    # Extract face region for age prediction
                    face_roi = frame[startY:endY, startX:endX]
                    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0: # Ensure ROI is valid
                        age_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        age_net.setInput(age_blob)
                        age_preds = age_net.forward()
                        
                        # Get predicted age range
                        age = AGE_LIST[age_preds[0].argmax()]
                        
                        # Draw bounding box and age on frame
                        label = f"Age: {age}"
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2) # Yellow bounding box (BGR)
                        # Ensure text is inside frame and above the box
                        text_y = startY - 10 if startY - 10 > 10 else startY + 20
                        cv2.putText(frame, label, (startX, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Add overall "Age Detection Active!" message at a fixed position
            if age_detected_in_frame:
                cv2.putText(frame, "AGE DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA) # Yellow text (BGR)
        else:
            # If models failed to load, display a warning
            cv2.putText(frame, "Age Detection Disabled (Model Error)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpeg', frame)
        if not ret:
            print("Failed to encode frame. Skipping frame.")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # Limit FPS to prevent excessive CPU usage


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
        /* Custom CSS for body font and spinner animation */
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
        /* Ensure the video container maintains aspect ratio */
        .aspect-video {
            position: relative;
            width: 100%;
            padding-bottom: 75%; /* 4:3 aspect ratio (height / width * 100) */
            background-color: #000; /* Black background for video area */
        }
        .aspect-video img { /* Only img element will be used for stream display */
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain; /* Ensure the video fits within the container */
        }
        /* Custom Toast Notifications */
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
        @keyframes fadeInOutToast {
            0% { opacity: 0; transform: translateY(20px); }
            10% { opacity: 1; transform: translateY(0); }
            90% { opacity: 1; transform: translateY(0); }
            100% { opacity: 0; transform: translateY(20px); }
        }

        /* New styles for fading out status message */
        .status-fade-out {
            animation: fadeOutStatus 3s forwards; /* 3 seconds to fade out after initial delay */
            animation-delay: 2s; /* Start fading out 2 seconds after streaming begins */
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
            <img id="flaskVideoStream" src="/video_feed" alt="Webcam Stream">
            <div id="statusMessage" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-white text-lg font-semibold bg-gray-800 bg-opacity-70 px-6 py-3 rounded-full flex items-center gap-3 transition-colors duration-300">
                <div class="spinner-custom"></div> Connecting...
            </div>
            <canvas id="hiddenCanvas" style="display: none;"></canvas>
        </div>

        <div class="mt-6 flex flex-col gap-4 justify-center">
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
        <p id="ageDetectionStatus" class="text-sm text-gray-600 mt-2 leading-relaxed">
            <span class="font-semibold text-yellow-600">Age Detection Active</span> (Detected faces will be outlined in yellow with age prediction.)
            <br><span class="text-xs text-red-600">
                !!! IMPORTANT: This feature requires 4 model files. Please download them (check Python script comments for links) and place them in the same directory as the script.
            </span>
        </p>
        <p class="text-xs text-red-600 mt-4 font-semibold p-2 bg-red-100 rounded-md border border-red-300">
            ⚠️ **SECURITY WARNING:** This is a basic demonstration for local use. Do NOT expose this webcam stream to the public internet without implementing robust security measures like authentication, HTTPS encryption, and proper firewall rules. This application is intended for local network access only.
        </p>
    </div>

    <div id="toast-container" class="toast-container-custom"></div>

    {% raw %}
    <script>
        const flaskVideoStream = document.getElementById('flaskVideoStream');
        const videoContainer = document.getElementById('videoContainer');
        const statusMessage = document.getElementById('statusMessage');
        const takeSnapshotBtn = document.getElementById('takeSnapshotBtn');
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        const hiddenCanvas = document.getElementById('hiddenCanvas');
        const toastContainer = document.getElementById('toast-container');
        const ageDetectionStatusDiv = document.getElementById('ageDetectionStatus');

        // This line caused the error due to string literal termination issues.
        // It's now correctly escaped using Python's f-string literal representation.
        let modelLoadingFailedFrontend = `{{ "true" if age_model_load_error else "false" }}`;

        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast-custom ${type}`;
            toast.innerHTML = message;
            toastContainer.appendChild(toast);

            setTimeout(() => {
                toast.remove();
            }, 4000);
        }

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
                if (modelLoadingFailedFrontend === 'true') { // Check if it's explicitly 'true'
                     statusMessage.style.visibility = 'visible';
                     statusMessage.style.opacity = '1';
                     statusMessage.classList.remove('status-fade-out');
                }
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

        updateStatus('Connecting to webcam...', 'connecting');

        flaskVideoStream.onload = function() {
            updateStatus('Streaming live...', 'streaming');
            if (modelLoadingFailedFrontend === 'true') { // Check if it's explicitly 'true'
                ageDetectionStatusDiv.innerHTML = `<span class="font-semibold text-red-600">Age Detection Disabled</span> (Model files not found or invalid. See console for details.)`;
                ageDetectionStatusDiv.classList.remove('text-yellow-600');
                ageDetectionStatusDiv.classList.add('text-red-600');
            }
        };

        flaskVideoStream.onerror = function() {
            updateStatus('Stream failed. Retrying in 3s...', 'error');
            flaskVideoStream.src = "/webcam_offline_placeholder.jpg";
            if (modelLoadingFailedFrontend !== 'true') { // Only show this if not already a model error
                modelLoadingFailedFrontend = true; // Set to true for subsequent checks
                ageDetectionStatusDiv.innerHTML = `<span class="font-semibold text-red-600">Age Detection Disabled</span> (Model files likely missing or invalid. Check your server console!)`;
                ageDetectionStatusDiv.classList.remove('text-yellow-600');
                ageDetectionStatusDiv.classList.add('text-red-600');
                showToast('Age Detection model files not found. Check server console for instructions!', 'error');
            }

            setTimeout(() => {
                flaskVideoStream.src = "/video_feed?" + new Date().getTime();
                updateStatus('Connecting to webcam...', 'connecting');
            }, 3000);
        };

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
            flaskVideoStream.src = "/video_feed?" + new Date().getTime();
        };
    </script>
    {% endraw %}
</body>
</html>
"""

@app.route('/')
def index():
    """
    Renders the main HTML page for the webcam stream.
    Also ensures the placeholder image exists before serving the page.
    """
    create_placeholder_image() # Ensure placeholder exists
    # Pass the Python variable to the Jinja2 template directly
    return render_template_string(HTML_TEMPLATE, age_model_load_error=age_model_load_error)

@app.route('/webcam_offline_placeholder.jpg')
def serve_placeholder():
    """Serves the placeholder image if the webcam is offline."""
    # Ensure the placeholder image is created before attempting to serve it
    create_placeholder_image()
    try:
        return app.send_static_file('webcam_offline_placeholder.jpg')
    except FileNotFoundError:
        return "Placeholder image not found on server.", 404


@app.route('/video_feed')
def video_feed():
    """
    Streams webcam frames as an MJPEG multipart response.
    This is the core video streaming endpoint.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("IP Webcam Stream Application Starting...")
    print("---------------------------------------")
    print("Initializing webcam (this might take a moment)...")
    create_placeholder_image() # Ensure placeholder is ready

    print("Flask server will start on: http://127.0.0.1:5000")
    print("Open this URL in your web browser to view the stream.")
    print("Press Ctrl+C in this terminal to stop the server and release the webcam.")
    print("---------------------------------------")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        release_camera() # Ensure camera is released properly on exit.
