---

# 🚀 AI Motion Twin

> **Real-Time Human Pose Detection, Motion Analytics & Action Recognition System**
> A modular computer vision pipeline that converts live or recorded video into an intelligent motion analysis system with visualization, tracking, and ML dataset generation.

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

# 📖 Overview

AI Motion Twin is designed as a **real-time AI system** that bridges:

* Computer Vision
* Motion Analysis
* Human Action Recognition
* Data Engineering for ML

It supports both:

* 🎥 **Live webcam processing**
* 🎬 **Offline video analysis**
* 🌐 **Browser-based streaming via FastAPI**

---

# ✨ Key Capabilities

### 🔍 Pose Detection

* 33-point human landmark detection (MediaPipe)
* High-confidence tracking
* Real-time performance (20–30 FPS typical)

### 👥 Multi-Person Tracking

* Up to 3 simultaneous people
* Persistent ID assignment
* Occlusion-tolerant tracking

### ⚡ Motion Intelligence

* EMA-based smoothing
* Velocity & acceleration tracking
* Energy computation (movement intensity)
* Symmetry scoring (left vs right body balance)

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

# 🧠 System Architecture

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
```

---

# 📂 Project Structure

```text
AI_Motion_Twin/
│
├── main.py                  # CLI entry point
├── api.py                   # FastAPI server
│
├── input_handler.py         # Frame capture (webcam/video)
├── pose_detector.py         # MediaPipe wrapper + tracking
├── skeleton_builder.py      # Joint derivation
├── motion_smoother.py       # EMA + velocity
├── renderer.py              # Visualization engine
├── offline_analyzer.py      # Dataset generator
├── utils.py                 # Helper functions
│
├── templates/
│   └── index.html           # REQUIRED for web UI
│
├── static/                  # Optional frontend assets
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Create Virtual Environment

```bash
python -m venv venv311
venv311\Scripts\activate
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Stable Dependencies

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

# ⚙️ Configuration

| Parameter | Description         | Default |
| --------- | ------------------- | ------- |
| max_width | Frame resize width  | 960     |
| multi     | Enable multi-person | False   |
| overlay   | Draw on video       | False   |
| heatmap   | Enable heatmap      | False   |

---

# ▶️ Usage

---

## 🟢 Desktop Mode

### Basic Run

```bash
python main.py
```

### Advanced Run

```bash
python main.py live --source 0 --multi --overlay --heatmap --record output.mp4
```

---

## 🌐 Web Mode (FastAPI)

### Run Server

```bash
uvicorn api:app --reload
```

### Open Browser

```
http://127.0.0.1:8000
```

---

## ⚠️ REQUIRED FILE

Create this or web will crash:

```text
templates/index.html
```

### Minimal HTML

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

## 📊 Offline Analyzer

```bash
python main.py analyze --input ./videos --output ./dataset --merge
```

---

# 🎨 Visualization Details

* **Left Panel:** Original video
* **Right Panel:** Skeleton rendering
* **Heatmap:** Motion intensity visualization
* **HUD Includes:**

  * FPS
  * Person count
  * Action label
  * Speed
  * Energy
  * Symmetry

---

# 📊 Dataset Schema

Each frame contains:

### Identity

* frame_id
* timestamp
* video_id

### Joint Data (36 joints)

* x, y, z
* visibility
* normalized coordinates

### Motion Data

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

# ⚡ Performance Considerations

* Optimal resolution: **≤ 960px width**
* FPS depends on:

  * CPU performance
  * Number of people
  * Heatmap enabled/disabled
* Recommended:

  * Close background apps
  * Use GPU-enabled OpenCV (optional)

---

# ❌ Common Issues & Fixes

---

## NumPy Error

```
numpy.core.multiarray failed to import
```

### Fix

```
numpy==1.26.4
```

---

## OpenCV Conflict

```
_ARRAY_API not found
```

### Fix

```
opencv-python==4.9.0.80
```

---

## FastAPI Error

```
FileNotFoundError: templates/index.html
```

### Fix

Create templates folder and index.html

---

# 🧪 Tech Stack

| Layer      | Technology |
| ---------- | ---------- |
| Vision     | MediaPipe  |
| Processing | NumPy      |
| Rendering  | OpenCV     |
| Backend    | FastAPI    |
| Server     | Uvicorn    |
| Data       | Pandas     |

---

# 🚧 Roadmap

* Deep learning action classifier (LSTM)
* Transformer-based motion analysis
* YOLO pose integration
* WebRTC streaming
* GUI dashboard (PyQt)
* Docker deployment
* Cloud hosting

---

# 🤝 Contributing

```bash
git checkout -b feature/new-feature
git commit -m "feat: add feature"
git push origin feature/new-feature
```

---

# 📄 License

MIT License

---

# 👨‍💻 Author

**Rishiraj Chaurasiya**

B.Tech — Artificial Intelligence & Machine Learning (AI/ML)

Developer of AI Motion Twin

GitHub: [https://github.com/your-username](https://github.com/your-username)
LinkedIn: [https://linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)

---