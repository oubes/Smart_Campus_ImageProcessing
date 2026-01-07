# Smart Campus (IoT-based) - Graduation Project 🎓

<p align="center">
  <img src="https://img.shields.io/badge/Timeline-Oct%202023%20--%20Jul%202024-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
</p>

An integrated AI-driven ecosystem for smart campuses. This project focuses on the **Image Processing & AI** core, combining Computer Vision with IoT and Web modules to automate security, attendance, and parking management.

---

## 📂 Repository Structure

| Module | Description | Key Tech |
| :--- | :--- | :--- |
| 👤 **[Auto Attendance](./Auto_Attendance_using_Face_Recognition/)** | Automated student tracking via facial recognition. | YOLOv8, DLIB |
| 🛡️ **[LPR System YOLOv8](./LPR_System_YOLOv8/)** | License plate OCR for automated gate control. | EasyOCR, YOLOv8 |
| 🅿️ **[Smart Parking](./Smart_Parking/)** | Real-time monitoring of parking spot availability. | YOLOv8, OpenCV |

---

## 🛠 Tech Stack & Skills

- **AI/ML:** Deep Learning, Machine Learning, Object Detection.
- **Computer Vision:** YOLOv8, DLIB, OpenCV, EasyOCR, PyTorch.
- **System Integration:** RTSP Streaming, API/Server connectivity, IoT Module Sync.

---

## 📺 System Demos

### 1️⃣ Auto Attendance (Face Recognition)
Automating campus attendance using high-precision face detection and verification.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8845f554-5286-4e1c-b121-387e716f0d41" width="45%" alt="Face Detection" />
  <img src="https://github.com/user-attachments/assets/65763541-e75a-40f8-be29-75e5cf5af0fb" width="53%" alt="Face Recognition" />
  <br>
  <em>Left: Face Detection in real-time | Right: Recognition & Identity Matching</em>
</p>

---

### 2️⃣ Smart Gate (LPR System)
Automated vehicle entry control through License Plate Recognition.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3f3ff223-2f23-4eaa-ac0e-93ffc7e2336e" width="80%" alt="LPR 1" />
  <br><br>
  <img src="https://github.com/user-attachments/assets/ca3c88c1-7751-4dc6-ba61-6dffa01c3f29" width="48%" />
  <br><br>
  <img src="https://github.com/user-attachments/assets/e79a429a-626a-4566-925d-2a62bd9f1634" width="48%" />
  <br><br>
  <img src="https://github.com/user-attachments/assets/0b489d0e-3d18-4452-9763-9358b44674e2" width="58%" />
  <br><br>
  <img src="https://github.com/user-attachments/assets/8029daf4-5b14-4d93-9f48-51d63284048f" width="38%" />
  <br><br>
  <em>The pipeline: Image Capturing > Vehicle Detection -> Plate Localization -> Character Extraction (OCR)</em>
</p>

---

### 3️⃣ Smart Parking
Real-time monitoring and reporting of parking lot occupancy.

<p align="center">
  <a href="https://github.com/user-attachments/assets/4cd6c81e-4121-4e99-986a-1bc15d5ed5ba">
    <img src="https://img.shields.io/badge/▶_Watch_Smart_Parking_Demo-Video-red?style=for-the-badge&logo=youtube" />
  </a>
  <br>
  <em>Click the badge above to watch the parking detection system in action.</em>
</p>

---

## 🛠 Core Utilities
Specialized handlers for handling large-scale campus camera feeds:
- `rtsp_stream_handler.py`: High-performance RTSP stream management.
- `rtsp_sampling_handler.py`: Frame sampling optimized for AI inference.
- `api.py`: Bridges the Image Processing modules with the central Web/Server.

---

## 👥 Contributors
* **Omar Gamal** ([@oubes](https://github.com/oubes))
* **Kareem Youssry** ([@KareemYoussry](https://github.com/KareemYoussry))
* **Eslam Hekal** ([@Hekal74](https://github.com/Hekal74))
