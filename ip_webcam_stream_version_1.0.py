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
    If the camera is not available, it serves a placeholder image.
    """
    # Ensure placeholder image exists at startup
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

        ret, buffer = cv2.imencode('.jpeg', frame)
        if not ret:
            print("Failed to encode frame. Skipping frame.")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Optional: Add a small delay to control frame rate and reduce CPU usage
        # time.sleep(0.03) # Approx 30 FPS


# HTML template string for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IP Webcam Stream 📸</title>
    <!-- Tailwind CSS CDN for quick styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Google Fonts - Inter for modern typography -->
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
        .aspect-video img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain; /* Ensure the video fits within the container */
        }
    </style>
</head>
<body class="bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center min-h-screen p-4 sm:p-6 md:p-8">
    <div class="bg-white p-6 md:p-8 rounded-xl shadow-2xl max-w-lg w-full text-center border-t-8 border-blue-600 transform hover:scale-105 transition-transform duration-300">
        <h1 class="text-3xl sm:text-4xl font-extrabold text-gray-800 mb-6 flex items-center justify-center gap-3">
            <span class="text-blue-600 text-5xl">📸</span> IP Webcam Stream
        </h1>

        <div class="relative w-full overflow-hidden rounded-lg border-4 border-gray-300 bg-black flex items-center justify-center aspect-video">
            <!-- The img tag's src points to our Flask video feed endpoint -->
            <img id="videoStream" src="/video_feed" alt="Webcam Stream">
            <!-- Status message overlay -->
            <div id="statusMessage" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-white text-lg font-semibold bg-gray-800 bg-opacity-70 px-6 py-3 rounded-full flex items-center gap-3 transition-colors duration-300">
                <div class="spinner-custom"></div> Connecting...
            </div>
        </div>

        <p class="text-sm text-gray-600 mt-6 leading-relaxed">
            Ensure your Python server is running and your webcam is accessible. If the stream doesn't appear, check your webcam connection or if another application is using it.
        </p>
        <p class="text-xs text-red-600 mt-4 font-semibold p-2 bg-red-100 rounded-md border border-red-300">
            ⚠️ **SECURITY WARNING:** This is a basic demonstration for local use. Do NOT expose this webcam stream to the public internet without implementing robust security measures like authentication, HTTPS encryption, and proper firewall rules. This application is intended for local network access only.
        </p>
    </div>

    {% raw %}
    <script>
        const videoStream = document.getElementById('videoStream');
        const statusMessage = document.getElementById('statusMessage');

        // Function to update status message with icons and colors
        function updateStatus(message, type = 'connecting') {
            statusMessage.innerHTML = ''; // Clear previous content
            let iconHtml = '';
            let bgColorClass = '';

            if (type === 'connecting') {
                iconHtml = '<div class="spinner-custom"></div>';
                bgColorClass = 'bg-gray-800 bg-opacity-70';
            } else if (type === 'streaming') {
                iconHtml = '✨'; // Sparkles emoji
                bgColorClass = 'bg-green-600 bg-opacity-80';
            } else if (type === 'error') {
                iconHtml = '🚨'; // Siren emoji
                bgColorClass = 'bg-red-600 bg-opacity-80';
            }

            statusMessage.innerHTML = iconHtml + ' ' + message;
            statusMessage.className = `absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-white text-lg font-semibold px-6 py-3 rounded-full flex items-center gap-3 transition-colors duration-300 ${bgColorClass}`;
        }

        // Initial status display
        updateStatus('Connecting to webcam...', 'connecting');

        // Event listener for when the video stream successfully loads an image
        videoStream.onload = function() {
            updateStatus('Streaming live...', 'streaming');
        };

        // Event listener for when the video stream encounters an error loading an image
        videoStream.onerror = function() {
            updateStatus('Stream failed. Retrying in 3s...', 'error');

            // Set the src to a local placeholder image path if the stream fails.
            // This placeholder will be served by Flask from the same directory.
            videoStream.src = "/webcam_offline_placeholder.jpg";

            // Basic retry logic for the actual video feed
            setTimeout(() => {
                // Append a timestamp to the URL to bypass browser caching,
                // ensuring it tries to fetch a fresh frame.
                videoStream.src = "/video_feed?" + new Date().getTime();
                updateStatus('Connecting to webcam...', 'connecting'); // Reset status on retry attempt
            }, 3000);
        };

        // When the page is fully loaded, attempt to initialize the stream.
        // This is primarily useful if the browser loads the page before the Python server is ready.
        window.onload = function() {
             // Initial attempt to load the video feed.
             // The onerror will catch it if the server isn't ready or camera fails.
             videoStream.src = "/video_feed?" + new Date().getTime();
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
    return render_template_string(HTML_TEMPLATE)

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
    get_camera() # Attempt to get camera early for initial status
    create_placeholder_image() # Ensure placeholder is ready

    print("Flask server will start on: http://127.0.0.1:5000")
    print("Open this URL in your web browser to view the stream.")
    print("Press Ctrl+C in this terminal to stop the server and release the webcam.")
    print("---------------------------------------")
    try:
        # Run Flask app. host='0.0.0.0' makes it accessible on your local network,
        # but for security, keep it '127.0.0.1' (localhost) unless you explicitly need network access.
        # For simplicity of testing, '0.0.0.0' is used, but user should be aware of implications.
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        release_camera() # Ensure camera is released properly on exit.

    print("IP Webcam Stream Application Stopped.")