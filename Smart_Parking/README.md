# Smart Parking Management 🅿️

A specialized computer vision module for the **Smart Campus** project designed to monitor parking occupancy in real-time. By leveraging **YOLOv8**, the system identifies available and occupied slots to optimize campus traffic flow.

---

## 📺 System Demonstration

#### 🎥 Full Video Demo
Experience the complete workflow, including spot selection and real-time status updates:

<p align="center">
  <a href="https://github.com/user-attachments/assets/4cd6c81e-4121-4e99-986a-1bc15d5ed5ba">
    <img src="https://img.shields.io/badge/▶_Watch_Parking_System_Demo-Video-red?style=for-the-badge&logo=youtube" alt="Watch Demo" />
  </a>
</p>

---

## ✨ Key Features
- **High-Precision Detection:** Powered by YOLOv8 to distinguish between vehicles and empty slots under various angles.
- **Dynamic Configuration:** Uses `spots.json` to store coordinate mappings, allowing for easy reconfiguration without code changes.
- **RTSP Integration:** Native support for campus CCTV feeds via `rtsp_stream_handler.py`.
- **Status Persistence:** Real-time logging of parking occupancy for high-level campus analytics.

---

## 🚀 Operations & Workflow

To get the system up and running, follow these steps:

1.  **Define Parking Slots:**
    Run `spots_selector.py` to open the GUI. Click to mark the four corners of each parking spot. This saves the coordinates into `spots.json`.
    ```bash
    python spots_selector.py
    ```

2.  **Start Monitoring:**
    Launch the detector to begin real-time analysis of the video stream.
    ```bash
    python spots_detector.py
    ```

3.  **Integration Test:**
    Use `test.py` to verify the connection between the detection logic and the server-side API.

---

## 📂 File Structure
| File | Role |
| :--- | :--- |
| `spots_selector.py` | Interactive GUI tool to define parking slot boundaries. |
| `spots_detector.py` | The main engine that runs YOLOv8 inference on defined spots. |
| `spots.json` | JSON database storing the ROI (Region of Interest) coordinates. |
| `test.py` | Unit testing for system deployment. |
