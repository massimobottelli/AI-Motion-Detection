# 🛡️ AI Motion Guard: Smart Detection & Recognition

**AI Motion Guard** is an intelligent video surveillance system that combines high-speed traditional **Motion Detection** with the precision of Deep Learning via **YOLO**. 

Unlike standard systems, this application drastically reduces false positives by analyzing video content only when real motion is detected. It utilizes a strategic 1-second delay from the initial trigger to ensure the object is fully visible and centered before performing AI recognition.


## ✨ Key Features

* **Hybrid Detection:** Ultra-fast motion detection (OpenCV MOG2) paired with Object Recognition (YOLO).
* **Smart Trigger (Delayed Analysis):** AI kicks in 1 second after motion starts to capture the best possible frame for identification.
* **Automated Recording:** Saves `.mp4` files only during active events, optimizing disk space.
* **Auto-Naming:** Videos are automatically renamed using a timestamp and the detected object label (e.g., `20260307-1830_person.mp4`).
* **Modular Architecture:** Clean, class-based Python code separating Vision, Recording, and Monitoring logic.
* **UDP Streaming Support:** Native support for remote video feeds (e.g., from a Raspberry Pi).



## 🛠️ Requirements

### Software
* **Python 3.9+**
* **GStreamer 1.0** (with `ugly` and `good` plugins for H.264 support)
* **Python Libraries:**
    ```bash
    pip install opencv-python numpy ultralytics PyGObject
    ```

### Hardware (Recommended)
* **IP Camera** or **Raspberry Pi** with a USB webcam.
* PC with a modern CPU or NVIDIA GPU (optional) for faster YOLO inference.



## ⚙️ Configuration (Parameters)

Key parameters are centralized in the `CONFIG` dictionary within the code:

| Parameter            | Description                                              | Default |
| :------------------- | :------------------------------------------------------- | :------ |
| `threshold`          | Motion sensitivity (lower = more sensitive)              | 250     |
| `min_area`           | Minimum pixel size of an object to trigger recording     | 1500    |
| `recognition_delay`  | Seconds to wait before YOLO analyzes the frame           | 1.0     |
| `cooldown`           | Seconds of no motion before closing the video file       | 2.0     |



## 🎥 Raspberry Pi Streaming Setup

To stream video from your Raspberry Pi to the system, use the following GStreamer command:

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    video/x-raw,width=640,height=360,framerate=20/1 ! \
    videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! \
    rtph264pay config-interval=1 ! \
    udpsink host=YOUR_PC_IP_ADDRESS port=5000

