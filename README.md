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
