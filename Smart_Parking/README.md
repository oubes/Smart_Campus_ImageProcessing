# Smart Parking Management 🅿️

A YOLOv8-based computer vision system to monitor and manage parking space availability across the campus.

## ✨ Features
- **Car Detection:** Uses YOLOv8 for high-accuracy vehicle identification.
- **Real-time Map:** Updates the occupancy status in `spots.json`.
- **Stream Support:** Integrated with `rtsp_stream_handler` for campus-wide CCTV connectivity.

## 🚀 Operations
- Use `spots_selector.py` to define the parking grid coordinates.
- Launch `spots_detector.py` to start the live monitoring and server reporting.
