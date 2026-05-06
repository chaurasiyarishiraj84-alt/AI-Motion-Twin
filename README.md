````md
# 🚀 AI Motion Twin

**Real-Time Human Pose Detection, Motion Analytics & Action Recognition System**

A modular computer vision pipeline that converts live or recorded video into an intelligent motion analysis system with visualization, tracking, and ML dataset generation.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Key Capabilities](#-key-capabilities)
* [System Architecture](#-system-architecture)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Configuration](#-configuration)
* [Usage](#-usage)
  * [Desktop Mode](#desktop-mode)
  * [Web Mode (FastAPI)](#web-mode-fastapi)
  * [Offline Analyzer](#offline-analyzer)
* [Visualization Details](#-visualization-details)
* [Dataset Schema](#-dataset-schema)
* [Performance Considerations](#-performance-considerations)
* [Common Issues & Fixes](#-common-issues--fixes)
* [Tech Stack](#-tech-stack)
* [Roadmap](#-roadmap)
* [Contributing](#-contributing)
* [License](#-license)
* [Author](#-author)

---

## 📖 Overview

AI Motion Twin is designed as a real-time AI system that bridges:

* Computer Vision
* Motion Analysis
* Human Action Recognition
* Data Engineering for ML

It supports:

* 🎥 Live webcam processing  
* 🎬 Offline video analysis  
* 🌐 Browser-based streaming via FastAPI  

---

## ✨ Key Capabilities

### 🔍 Pose Detection
* 33-point human landmark detection (MediaPipe)
* High-confidence tracking
* Real-time performance (20–30 FPS)

### 👥 Multi-Person Tracking
* Up to 3 simultaneous people
* Persistent ID assignment
* Occlusion-tolerant tracking

### ⚡ Motion Intelligence
* EMA-based smoothing
* Velocity & acceleration tracking
* Energy computation (movement intensity)
* Symmetry scoring (left vs right balance)

### 🎯 Action Recognition (Rule-Based)

Detects:
* Idle
* Walking
* Jumping
* Squatting
* Waving
* T-Pose
* Dancing

### 🎨 Visualization Engine
* Stick figure rendering
* Motion heatmap overlay
* Side-by-side composite output
* Real-time HUD (FPS + metrics)

### 📊 Dataset Generation
* Frame-wise CSV output
* ML-ready structured data
* Suitable for:
  * LSTM
  * Transformers
  * Classification models

---

## 🧠 System Architecture

```text
Video Input (Webcam / File)
        ↓
Frame Processing (OpenCV)
        ↓
Pose Detection (MediaPipe)
        ↓
Joint Derivation (Extra Keypoints)
        ↓
Motion Smoothing (EMA)
        ↓
Velocity & Acceleration
        ↓
Person Tracking (Centroid Matching)
        ↓
Action Classification
        ↓
Rendering Engine (Skeleton + Heatmap)
        ↓
Output:
    → Desktop UI
    → Web Streaming (FastAPI)
    → CSV Dataset
````

---

## 📂 Project Structure

```text
AI_Motion_Twin/
│
├── main.py
├── api.py
│
├── input_handler.py
├── pose_detector.py
├── skeleton_builder.py
├── motion_smoother.py
├── renderer.py
├── offline_analyzer.py
├── utils.py
│
├── templates/
│   └── index.html
│
├── static/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv311
venv311\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Recommended Stable Dependencies

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

## ⚙️ Configuration

| Parameter | Description         | Default |
| --------- | ------------------- | ------- |
| max_width | Frame resize width  | 960     |
| multi     | Enable multi-person | False   |
| overlay   | Draw on video       | False   |
| heatmap   | Enable heatmap      | False   |

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

### 🌐 Web Mode (FastAPI)

Run server:

```bash
uvicorn api:app --reload
```

Open:

```
http://127.0.0.1:8000
```

---

### ⚠️ Required File

Create:

```
templates/index.html
```

Minimal:

```html
<!DOCTYPE html>
<html>
<head>
    <title>AI Motion Twin</title>
</head>
<body>
    <h2>Live Feed</h2>
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

## 🎨 Visualization Details

* Left: Original video
* Right: Skeleton canvas
* Heatmap: Motion intensity
* HUD shows:

  * FPS
  * People count
  * Action
  * Speed
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

* x, y, z
* visibility
* normalized coords

### Motion Data

* velocity (vx, vy)
* acceleration (ax, ay)

### Metrics

* joint angles
* symmetry
* energy

### Labels

* action_label
* confidence

---

## ⚡ Performance Considerations

* Use ≤ 960px width
* FPS depends on CPU & people count
* Disable heatmap for better speed
* Close background apps

---

## ❌ Common Issues & Fixes

### NumPy Error

```
numpy.core.multiarray failed to import
```

Fix:

```
numpy==1.26.4
```

### OpenCV Error

```
_ARRAY_API not found
```

Fix:

```
opencv-python==4.9.0.80
```

### FastAPI Error

```
templates/index.html not found
```

Fix:
Create templates folder and file

---

## 🧪 Tech Stack

| Layer      | Tech      |
| ---------- | --------- |
| Vision     | MediaPipe |
| Processing | NumPy     |
| Rendering  | OpenCV    |
| Backend    | FastAPI   |
| Server     | Uvicorn   |
| Data       | Pandas    |

---

## 🚧 Roadmap

* Deep learning action classifier
* Transformer-based analysis
* YOLO pose integration
* WebRTC streaming
* GUI dashboard
* Docker support
* Cloud deployment

---

## 🤝 Contributing

```bash
git checkout -b feature/new-feature
git commit -m "feat: add feature"
git push origin feature/new-feature
```

---

## 📄 License

MIT License

---

## 👨‍💻 Author

**Rishiraj Chaurasiya**
B.Tech — Artificial Intelligence & Machine Learning

GitHub: [https://github.com/chaurasiyarishiraj84-alt](https://github.com/chaurasiyarishiraj84-alt)
LinkedIn: [https://linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)      convert this into terminal one for easy cp 

```

