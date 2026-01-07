# Auto Attendance System 👤

Part of the **Smart Campus** graduation project. This module automates student attendance through real-time facial recognition, integrating Computer Vision with backend logging.

---

## 📺 System Demos

#### 🔍 Face Detection Stage
The system first localizes faces within the frame using a specialized YOLOv8-face model, ensuring high recall even in crowded campus environments.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8845f554-5286-4e1c-b121-387e716f0d41" width="600" alt="Face Detection Demo" />
</p>

#### ✅ Face Recognition & Verification
Once detected, the system extracts facial embeddings and matches them against the student database using DLIB to verify identity and log attendance.

<p align="center">
  <img src="https://github.com/user-attachments/assets/65763541-e75a-40f8-be29-75e5cf5af0fb" width="750" alt="Face Recognition Demo" />
</p>

---

## ✨ Key Features
* **Hybrid Detection:** Utilizes YOLOv8 for robust real-time face localization.
* **High-Accuracy Recognition:** Powered by DLIB for precise identity verification.
* **IoT & Cloud Integration:** Automatically pushes attendance logs to the central server via `api.py`.
* **Optimized Preprocessing:** Includes face alignment and normalization to handle varying lighting conditions.

---

## 📂 File Map
| File | Responsibility |
| :--- | :--- |
| `detection.py` | Handles the primary face localization logic. |
| `recognition.py` | Manages identity matching against the student database. |
| `preprocessing.py` | Performs image alignment and normalization for better accuracy. |
| `tasks.py` | Manages asynchronous background tasks for server communication. |
| `yolov8n-face.pt` | Pre-trained weights for the face detection model. |

---
