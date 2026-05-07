````md
# 🚀 AI Motion Twin

<p align="center">
  <b>Real-Time Human Pose Detection • Motion Analytics • Action Recognition</b><br>
  <i>Transform video into intelligent motion insights with visualization, tracking, and ML-ready datasets</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <img src="https://img.shields.io/badge/OpenCV-4.9-green">
  <img src="https://img.shields.io/badge/MediaPipe-0.10.14-orange">
  <img src="https://img.shields.io/badge/FastAPI-Backend-teal">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

---

## 🎯 Overview

AI Motion Twin is a real-time computer vision system that converts human motion into structured, analyzable data.

It integrates:

- 📷 Computer Vision  
- 🧠 Motion Intelligence  
- 🤖 Action Recognition  
- 📊 ML Dataset Generation  

### 🔥 Supports:
- 🎥 Live webcam processing  
- 🎬 Video file analysis  
- 🌐 Browser streaming via FastAPI  

---

## ✨ Features

### 🔍 Pose Detection
- 33-point human landmark detection (MediaPipe)
- High-confidence real-time tracking
- ~20–30 FPS performance

### 👥 Multi-Person Tracking
- Supports up to 3 individuals
- Persistent ID assignment
- Handles occlusion & re-entry

### ⚡ Motion Intelligence
- EMA-based smoothing
- Velocity & acceleration tracking
- Energy scoring (movement intensity)
- Symmetry analysis (body balance)

### 🎯 Action Recognition
Detects:
- Idle  
- Walking  
- Jumping  
- Squatting  
- Waving  
- T-Pose  
- Dancing  

### 🎨 Visualization Engine
- Stick figure rendering
- Motion heatmap overlay
- Side-by-side output (video + skeleton)
- Real-time HUD (FPS, metrics, actions)

### 📊 Dataset Generation
- Frame-by-frame CSV output
- ML-ready structured format
- Compatible with:
  - LSTM
  - Transformers
  - Classification models

---

## 🧠 Architecture

```text
Video Input
   ↓
OpenCV Processing
   ↓
MediaPipe Pose Detection
   ↓
Joint Processing
   ↓
Motion Smoothing (EMA)
   ↓
Tracking (Multi-Person)
   ↓
Action Recognition
   ↓
Rendering Engine
   ↓
Outputs:
  → Desktop UI
  → Web Streaming
  → CSV Dataset
````

---

## 📂 Project Structure

```text
AI_Motion_Twin/

├── main.py                 # CLI entry
├── api.py                  # FastAPI server

├── input_handler.py
├── pose_detector.py
├── skeleton_builder.py
├── motion_smoother.py
├── renderer.py
├── offline_analyzer.py
├── utils.py

├── templates/
│   └── index.html          # Required for web UI

├── static/                 # Optional frontend assets
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Recommended Versions

```txt
opencv-python==4.9.0.80
mediapipe==0.10.14
numpy==1.26.4
pandas>=2.0.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
```

---

## ▶️ Usage

### 🟢 Desktop Mode

```bash
python main.py
```

Advanced:

```bash
python main.py live --source 0 --multi --overlay --heatmap --record output.mp4
```

---

### 🌐 Web Mode

```bash
uvicorn api:app --reload
```

Open in browser:

```text
http://127.0.0.1:8000
```

---

### ⚠️ Required File

Create:

```text
templates/index.html
```

Minimal:

```html
<!DOCTYPE html>
<html>
<body>
<h2>AI Motion Twin</h2>
<img src="/video" width="900">
</body>
</html>
```

---

### 📊 Offline Analyzer

```bash
python main.py analyze --input ./videos --output ./dataset --merge
```

---

## 🎨 Visualization

| Panel   | Description      |
| ------- | ---------------- |
| Left    | Original Video   |
| Right   | Skeleton View    |
| Heatmap | Motion Intensity |

HUD displays:

* FPS
* People count
* Action label
* Energy
* Symmetry

---

## 📊 Dataset Schema

Each frame contains:

### Identity

* frame_id
* timestamp
* video_id

### Joint Data

* x, y, z coordinates
* visibility score
* normalized coordinates

### Motion

* velocity (vx, vy)
* acceleration (ax, ay)

### Metrics

* joint angles
* symmetry score
* energy score

### Labels

* action_label
* confidence

---

## ⚡ Performance

* Recommended resolution: ≤ 960px
* Disable heatmap for higher FPS
* Performance depends on CPU

---

## ❌ Common Issues

### NumPy Error

```text
numpy.core.multiarray failed to import
```

Fix:

```text
numpy==1.26.4
```

### OpenCV Error

```text
_ARRAY_API not found
```

Fix:

```text
opencv-python==4.9.0.80
```

### FastAPI Error

```text
templates/index.html not found
```

Fix:
Create the required file.

---

## 🧪 Tech Stack

| Layer      | Technology |
| ---------- | ---------- |
| Vision     | MediaPipe  |
| Processing | NumPy      |
| Rendering  | OpenCV     |
| Backend    | FastAPI    |
| Server     | Uvicorn    |
| Data       | Pandas     |

---

## 🚧 Roadmap

* Deep learning action classifier
* Transformer-based motion modeling
* YOLO pose integration
* WebRTC streaming
* GUI dashboard
* Docker deployment
* Cloud hosting

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
git commit -m "feat: your feature"
git push origin feature/your-feature
```

---

## 📄 License

MIT License

---

## 👨‍💻 Author

**Rishiraj Chaurasiya**
B.Tech — Artificial Intelligence & Machine Learning

GitHub: [https://github.com/chaurasiyarishiraj84-alt](https://github.com/chaurasiyarishiraj84-alt)

```
```
