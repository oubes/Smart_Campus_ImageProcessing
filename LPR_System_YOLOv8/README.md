# Smart Gate: License Plate Recognition (LPR) 🚗

Designed for automated entry control within the **Smart Campus**, this module identifies authorized vehicles by detecting and reading license plates in real-time using **YOLOv8** and **EasyOCR**.

---

## 📺 System Workflow & Demos

The LPR system follows a multi-stage pipeline to ensure high accuracy in various environment conditions:

#### 1️⃣ Vehicle & Plate Detection
The system first identifies the vehicle and then pinpoints the exact location of the license plate using a specialized YOLO model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3f3ff223-2f23-4eaa-ac0e-93ffc7e2336e" width="850" alt="LPR Overview" />
</p>

#### 2️⃣ Detection Analysis
Multiple candidates are processed to ensure the best crop is selected for the OCR stage.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ca3c88c1-7751-4dc6-ba61-6dffa01c3f29" width="48%" alt="Plate Crop" />
  <img src="https://github.com/user-attachments/assets/e79a429a-626a-4566-925d-2a62bd9f1634" width="48%" alt="Plate Localization" />
</p>

#### 3️⃣ Optical Character Recognition (OCR)
Using **EasyOCR**, the system extracts alphanumeric characters, providing the plate text along with a confidence score for validation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0b489d0e-3d18-4452-9763-9358b44674e2" width="600" alt="OCR Analysis" />
  <br>
  <img src="https://github.com/user-attachments/assets/8029daf4-5b14-4d93-9f48-51d63284048f" width="400" alt="OCR Output Result" />
</p>

---

## 🛠 Technical Workflow
1.  **Vehicle Detection:** YOLOv8 identifies the vehicle approaching the gate.
2.  **LPD (License Plate Detection):** A specialized YOLO model crops the license plate area.
3.  **OCR Stage:** EasyOCR extracts the alphanumeric text from the cropped image.
4.  **Validation:** The system cross-checks the recognized plate against the campus database for gate authorization.

---

## 📂 Key Structure
| Folder/File | Description |
| :--- | :--- |
| `pretrained_models/` | Contains the specialized weights for vehicle and plate detection. |
| `data_processing/` | Scripts for handling and cleaning OCR results. |
| `image_processing/` | Enhancement filters to improve OCR readability. |
| `scripts/` | Optimization and performance testing utilities. |
| `vars.py` | Global configuration and path variables. |
