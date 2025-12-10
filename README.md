
---

# Facial Asymmetry Analysis Using Object Detection: YOLOv8 Implementation for Bell’s Palsy Identification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-orange)

This project integrates **YOLOv8** with **MediaPipe FaceMesh** to analyze facial asymmetry in *real-time* as an early indicator of possible **Bell’s Palsy**. The system detects facial landmarks, measures geometric imbalance, and provides an asymmetry score directly from the webcam feed.

---

## 📐 System Architecture Diagram

The diagram below illustrates the full processing pipeline from face detection to asymmetry scoring:

![System Diagram](image/Diagram_Arsitektur.png.png)

---

## 📌 Key Features

* **Face Detection (YOLOv8):** Accurately locates the facial region on each frame.
* **Landmark Extraction (MediaPipe):** Captures 468 facial landmarks for detailed measurement.
* **Asymmetry Measurement:** Calculates geometric imbalance using Euclidean distance.
* **Real-time Feedback:** Displays results, guides, and asymmetry scores instantly from webcam input.
* **Clear Visualization:** Bounding boxes and helper lines assist interpretation.

---

## 🛠️ Technologies Used

* Python 3.10
* Ultralytics YOLOv8
* MediaPipe Face Mesh
* OpenCV
* NumPy

---

## ⚙️ Installation

### 1. Create Conda Environment

```bash
conda create -n bellpalsy python=3.10
conda activate bellpalsy
```

### 2. Install Required Dependencies

```bash
pip install ultralytics mediapipe opencv-python numpy
```

### 3. Set Model Path

Open `Code/main.py` and update:

```python
BEST_MODEL_PATH = r'C:\Users\YourUser\...\bell_palsy_project\train_result_v12\weights\best.pt'
```

---

## 🚀 How to Run

1. Connect your webcam.
2. Open the project directory (`Project_Akhir_Compvis`).
3. Run the main script:

```bash
python Code/main.py
```

4. On-screen indicators:

   * **Green box** → YOLO face detection
   * **Yellow/Purple lines** → Facial asymmetry measurement
   * **Score text** → Asymmetry index

5. Press **q** to exit.

---

## 🧠 System Workflow

### 1. Face Detection — YOLOv8

The model detects the face and provides bounding box coordinates as the reference region.

### 2. Landmark Extraction — MediaPipe

FaceMesh generates 468 3D facial landmarks. The system focuses on:

* Left & right eyes
* Left & right eyebrows
* Mouth corners

### 3. Asymmetry Scoring

Asymmetry is calculated using:

**Score = (Eyebrow Difference + Mouth Corner Difference) / Eye Distance**

Normalization ensures stable results even when the face is closer or farther from the camera.

### 4. Classification

* **Score < 5.0 → SYMMETRICAL**
* **Score ≥ 5.0 → POSSIBLE BELL’S PALSY**

---

## 📸 Sample image

### Detection Result: Possible Bell’s Palsy

![Example 1](image/Bell_Palsy.png.jpg)

---

### Detection Result: Symmetrical

![Example 2](image/Simetris.png.jpg)

---

## ⚠️ Disclaimer

This system is developed for academic purposes and is **not** intended for medical diagnosis.
For clinical evaluation, please consult a medical professional.

---

## 👨‍💻 Authors

Students of the Faculty of Computer Science – Universitas Brawijaya:

1. **Barru Wira Yasa** (235150301111021)
2. **Muhammad Shean Elliora Ribah** (235150307111045)
3. **Rayhan Sulistyawan** (235150301111019)

---
