# Surface Defect Detection on Hot-Rolled Steel Strips (No Model Training)

This project provides a complete pipeline for detecting surface defects on hot-rolled steel strips using either pre-trained deep learning models (YOLOv5/YOLOv7) or classical image processing techniques. **No model training required.**

## Features
- Use pre-trained YOLOv5/YOLOv7 models for defect detection (public weights)
- Classical image processing fallback (OpenCV-based)
- Batch or real-time (video) processing
- Modular, well-documented code
- Visualization of detected defects (bounding boxes/masks)

## Project Structure
```
surface_defect_detection/
│
├── data/
│   ├── sample_images/           # Sample steel strip images for testing
│   └── videos/                  # Optional: sample video files for real-time detection
│
├── models/
│   ├── yolov5_weights.pt        # Pre-trained YOLOv5 weights (download instructions below)
│   └── yolov7_weights.pt        # Optional: YOLOv7 weights
│
├── src/
│   ├── defect_detection.py      # Main script with detection logic (both classical & DL inference)
│   ├── preprocess.py            # Image preprocessing functions (filtering, enhancement)
│   ├── inference.py             # Functions to load pre-trained models and run inference
│   └── utils.py                 # Utility functions (visualization, image loading)
│
├── requirements.txt             # Python dependencies
│
├── README.md                   # Project overview, setup instructions, usage guide
│
└── run_detection.py             # Simple CLI script to run defect detection on images/videos
```

## Setup Instructions
1. **Clone this repository** (or copy the folder structure).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download pre-trained weights:**
   - **YOLOv5:**
     - Download a YOLOv5 model trained on steel defect datasets (e.g., NEU-DET, Severstal) from [YOLOv5 releases](https://github.com/ultralytics/yolov5/releases) or [community models](https://github.com/ultralytics/yolov5/issues/7015).
     - Place the `.pt` file in the `models/` directory and rename as `yolov5_weights.pt`.
   - **YOLOv7 (optional):**
     - Download weights from [YOLOv7 releases](https://github.com/WongKinYiu/yolov7/releases).
     - Place in `models/` as `yolov7_weights.pt`.

## Usage Guide
- **Run detection on images or video:**
  ```bash
  python run_detection.py --input data/sample_images/ --method yolo
  python run_detection.py --input data/sample_images/ --method classical
  python run_detection.py --input data/videos/sample.mp4 --method yolo
  ```
- See `run_detection.py --help` for all options.

## Notes
- For best results with YOLO, use weights trained on steel defect datasets (NEU-DET, Severstal, etc.).
- Classical methods work without GPU or deep learning frameworks, but may be less accurate.

## References
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [NEU-DET Dataset](https://github.com/idealvin/neu-det)
- [Severstal Steel Defect Dataset](https://www.kaggle.com/competitions/severstal-steel-defect-detection) 