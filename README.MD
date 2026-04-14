# IP Webcam Stream Projects: A Journey Through Computer Vision 📸

This document outlines the various stages and features developed for the IP Webcam Stream application, demonstrating a progression from a basic stream to advanced real-time computer vision capabilities. Each project phase built upon the last, culminating in a versatile multi-feature application.

## Project Overview & Evolution 📈

### Phase 1: Basic IP Webcam Stream
* **Purpose:** The foundational project. Establishes a Flask web server to stream live video from your local webcam to a web browser. It provides a simple, low-latency video feed.
* **Key Features:** Live MJPEG streaming, snapshot capture, full-screen viewing.

### Phase 2: Object Counting (Initial Iteration - Deprecated)
* **Purpose:** An early attempt to add computer vision by counting specific, static objects.
* **Status:** This specific iteration was superseded by more robust real-time object detection due to limitations in simple counting methods for complex scenes.

### Phase 3: Real-time Object Detection
* **Purpose:** Enhanced the stream with dynamic, real-time object detection using a pre-trained deep learning model (MobileNet-SSD). This allows the application to identify and label multiple objects in motion or static within the video frame.
* **Key Features:** Real-time bounding boxes and labels for detected objects (e.g., bottles, books, persons, cars).

### Phase 4: Real-time Age Detection
* **Purpose:** Shifted focus to human-centric computer vision, specifically detecting faces and estimating age ranges. This involves a two-step process: first, detecting faces, and then, analyzing those facial regions for age prediction.
* **Key Features:** Real-time face bounding boxes with predicted age ranges displayed on the stream.

### Phase 5: Multi-Feature IP Webcam Stream (Current Version) ✨
* **Purpose:** The culmination of all previous efforts, this application integrates the basic streaming, real-time object detection, and real-time age detection into a single, user-selectable interface. Users can choose their desired mode directly from the web page.
* **Key Features:**
    * **Selectable Modes:** Switch between Basic Stream (No AI), Object Detection, and Age Detection.
    * **Snapshot Functionality:** Capture and download still images from any active stream.
    * **Full Screen Mode:** View the webcam feed in full-screen.
    * **Dynamic UI Updates:** Status messages and feature information update based on the selected mode and model loading status.
    * **Webcam Offline Placeholder:** Displays a clear "WEBCAM OFFLINE" message if the camera is not accessible.
    * **Responsive Design:** Styled with Tailwind CSS for basic adaptability.

---

## Technologies Used 💻

* **Python 3.x**
* **Flask:** A micro web framework for handling web requests and serving the video stream.
* **OpenCV-Python:** The primary library for all computer vision tasks (webcam access, image processing, DNN inference).
* **NumPy:** Essential for numerical operations within OpenCV.
* **HTML5, CSS3, JavaScript:** For the interactive web interface.
* **Tailwind CSS:** For streamlined UI styling.
* **Pre-trained Deep Learning Models:**
    * MobileNet-SSD (Caffe model for Object Detection)
    * Caffe-based Face Detector (for Age Detection prerequisite)
    * Caffe-based Age Predictor Model

---

## Prerequisites ✅
Before running the application, ensure you have:
* Python 3.x installed.
* pip (Python package installer) for library management.
* Install the necessary Python libraries:

```bash
    pip install opencv-python flask numpy
```

---

## !!! CRITICAL STEP: Download All Required AI Models !!! ⚠️

To enable the Object Detection and Age Detection features, you **MUST DOWNLOAD** several pre-trained deep learning model files and place them in the **SAME DIRECTORY** as your Python script (`ip_webcam_combined.py`).

If these files are missing or corrupted, the respective AI features will be disabled, and you will see "Model Error" warnings in the console and on the web interface.

### Required Model Files (Total 7 Files):

#### A) For Object Detection (3 files):
1.  **MobileNetSSD_deploy.prototxt** (Model Architecture)
    * [Direct Download](https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt)
    * **How to Download:** Click the link. When the plain text content loads, right-click and select "Save As...". Save as `MobileNetSSD_deploy.prototxt`.
2.  **MobileNetSSD_deploy.caffemodel** (Model Weights)
    * [Direct Download](https://sourceforge.net/projects/ip-cameras-for-vlc/files/MobileNetSSD_deploy.caffemodel/download)
    * **How to Download:** Clicking this link should trigger the download (approx. 23MB).
3.  **coco.names** (Class Labels)
    * **How to Create:** Open a text editor (Notepad). Paste the list of 21 objects (background, aeroplane... tvmonitor) and save as `coco.names`.

#### B) For Age Detection (4 files):
1.  **deploy.prototxt** (Face Detector Architecture)
    * [Direct Download](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
2.  **res10_300x300_ssd_iter_140000.caffemodel** (Face Detector Weights)
    * [Direct Download](https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/res10_300x300_ssd_iter_140000.caffemodel)
3.  **age_deploy.prototxt** (Age Predictor Architecture)
    * [Direct Download](https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_deploy.prototxt)
4.  **age_net.caffemodel** (Age Predictor Weights)
    * [Direct Download](https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/age_net.caffemodel)

**Ensure ALL SEVEN downloaded files are in the SAME FOLDER as your Python script!**

---

## How to Run the Multi-Feature Application ▶️

1.  **Save the Python Script:** Save the provided Python code as `ip_webcam_combined.py`.
2.  **Place Model Files:** Ensure all seven model files are in the same directory as your script.
3.  **Run the Script:**
    ```bash
    python ip_webcam_combined.py
    ```
4.  **Access the Web Stream:** Open the URL provided in the console (e.g., `http://127.0.0.1:5000`) in your browser.
5.  **Select Mode:** Use the dropdown menu to select a mode and click "Start Stream."

---

## Troubleshooting 🔍

* **"Model Error" Messages:** Verify all seven files are in the same folder and named correctly.
* **"Error initializing camera":** Ensure no other apps (Zoom/Skype) are using the camera. Try changing `cv2.VideoCapture(0)` to index `1` or `2`.
* **Web page shows "Connecting..." indefinitely:** Check the console for errors and test the webcam with a native camera app.
* **SyntaxError:** Ensure you have copied the entire Python code block exactly as provided.

---

## Security Warning ⚠️
This application is for **local use only**. It does not implement authentication or HTTPS. **DO NOT** expose this stream to the public internet without implementing proper security protocols.
