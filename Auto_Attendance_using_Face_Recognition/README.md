# Auto Attendance System 👤

Part of the Smart Campus graduation project. This module automates attendance through facial recognition, replacing traditional manual methods.

## ✨ Features
- **Face Detection:** Real-time localization using YOLOv8/DLIB.
- **Recognition:** Identity verification using DLIB for high accuracy.
- **IoT Integration:** Sends attendance logs directly to the central server via `api.py`.

## 📂 File Map
- `detection.py`: Handles initial face localization.
- `recognition.py`: Performs matching against the student database.
- `preprocessing.py`: Image alignment and normalization.
- `yolov8n-face.pt`: Pre-trained face detection weights.
