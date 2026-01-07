# Smart Gate: License Plate Recognition (LPR) 🚗

Designed for automated entry control within the Smart Campus. It identifies authorized vehicles by reading license plates in real-time.

## 🛠 Technical Workflow
1. **Vehicle Detection:** YOLOv8 identifies the vehicle approaching the gate.
2. **LPD (License Plate Detection):** A specialized YOLO model crops the plate area.
3. **OCR Stage:** EasyOCR extracts the alphanumeric text from the plate.
4. **Validation:** System checks the plate against the authorized database.

## 📂 Key Folders
- `pretrained_models/`: Contains the specialized LPD and vehicle weights.
- `scripts/`: Optimization and testing utilities.
