# AI Object Detection & Real-Time Tracking Dashboard

A powerful, high-performance computer vision application designed to perform real-time multi-object detection and tracking. This repository delivers a dual-interface system: a lightweight, robust desktop script powered by **OpenCV**, and a premium web dashboard built with **Streamlit** using customized dark-theme aesthetics. 

The core engine leverages a pre-trained **YOLOv8** architecture integrated with the advanced **ByteTrack** algorithm to accurately track unique targets across consecutive video frames.

---

## Key Features

- **Dual-Interface System:**
  - `main.py`: High-performance desktop solution with automatic video looping and live screen text overlays.
  - `app.py`: Interactive web UI featuring dynamic sidebar configuration, responsive layout, and customized metric card blocks.
- **State-of-the-Art Tracking:** Employs advanced spatial-temporal tracking algorithms (**ByteTrack**) that seamlessly assign and persist unique IDs to targets, reducing tracking identity switches.
- **Dynamic Class Filtering:** Real-time multi-select menu allowing users to isolate and track specific objects of interest (e.g., Persons, Cars, Bicycles) from the COCO dataset.
- **Real-Time Analytics:** Live dashboard counters tracking:
  - **Objects Present on Screen:** The current target count within the immediate frame.
  - **Total Unique Objects Tracked:** The cumulative count of distinct object IDs detected throughout the entire video session.
- **Adaptive Frame Optimization:** Auto-resizing capabilities and video box constraint fixes (`max-height: 450px`) to ensure full visual stability across different device screens.

---

## Tech Stack & Dependencies

- **Core Language:** Python 3
- **Computer Vision Framework:** OpenCV (`opencv-python`)
- **Deep Learning Architecture:** Ultralytics YOLOv8 (`yolov8n.pt`)
- **Web Application Framework:** Streamlit
- **Tracking Algorithm:** ByteTrack / BoT-SORT (Native YOLOv8 persistence)

---

## Project Directory Structure

```text
├── app.py             # Interactive Streamlit Web Dashboard
├── main.py            # High-performance OpenCV Desktop Script
├── requirements.txt   # Project environment dependencies
├── .gitignore         # Prevents heavy model weights (.pt) and raw videos from tracking
└── README.md          # Project comprehensive documentation