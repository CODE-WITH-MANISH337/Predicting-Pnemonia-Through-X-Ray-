# 🩺 Predicting Pneumonia Through Chest X-Rays

An end-to-end deep learning pipeline designed to analyze chest X-ray images and provide instant, accurate predictive insights for pneumonia classification. Built with a focus on model optimization and lightweight cloud deployment.

🚀 **Live Web Application:** [Launch Streamlit App](https://wydr7msozxs3nhgy6fcv2g.streamlit.app/)


---

## 📌 Features
* **Interactive UI**: Upload an X-ray image via an intuitive drag-and-drop web interface built with Streamlit.
* **Lightweight Inference**: Uses an optimized TensorFlow Lite (`.tflite`) model architecture for lightning-fast prediction speeds on cloud infrastructure.
* **Computer Vision Processing**: Implements real-time image resizing, color mapping, and normalization channels using OpenCV.

---

## 📂 Repository Structure
```text
├── .gitattributes          # Git LFS tracking configuration
├── .gitignore              # Multi-stage ignore targets (datasets/weights)
├── front.py                # Main Streamlit UI frontend application
├── model_optimized.tflite  # Optimized quantized model weights (91.61 MB)
├── requirements.txt        # Controlled environment dependency manifest
└── README.md               # Documentation manual
```

---

## ⚙️ Core Architecture & Optimization

### 1. Database Handling
The project leverages a robust chest X-ray database containing thousands of high-resolution images categorized into normal lung scans and cases presenting bacterial/viral pneumonia indicators.

### 2. Model Compression
To bypass server memory walls and achieve production-grade deployment speeds, the original model training checkpoints were quantized and compressed down into an optimized **TensorFlow Lite (`.tflite`)** runtime binary (under 100MB).

### 3. Cloud Infrastructure
Deployed seamlessly using Streamlit Cloud on a configured **Python 3.12** environment engine, running isolated serverless architecture containers.

---

## 🛠️ Local Installation & Setup

To clone and run this application locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com
cd Through-X-Ray-Predicting-Pnemonia
```

### 2. Configure Environment Dependencies
Install the required pre-compiled system wheels via the requirements configuration:
```bash
pip install -r requirements.txt
```

### 3. Launch the Application Locally
```bash
streamlit run front.py
```

---

## 💻 Tech Stack
* **Language:** Python 3.12
* **Framework:** Streamlit
* **Core CV Engine:** OpenCV (Headless) & Pillow
* **Inference Pipeline:** TensorFlow Runtime / TFLite Interpreter
* **Version Control:** Git Large File Storage (LFS)

---

## 👤 Author
* **Manish** - [Linkedin](https://www.linkedin.com/in/manish-jaiswar-13615b377/)
