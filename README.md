::: {align="center"}
# 👁️ AuthentiVision

### **Hybrid Feature Fusion for Screen-Retake Image Detection**

*Detect • Verify • Protect*

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-black?logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-success)
:::

------------------------------------------------------------------------

# 📌 Overview

**AuthentiVision** is a hybrid AI framework for detecting
**Screen-Retake (Presentation Attack)** images.

Unlike conventional image classifiers that rely only on deep learning,
AuthentiVision combines:

-   🧠 Deep CNN Features (ResNet18)
-   🔍 Handcrafted Texture Features
-   ⚡ Support Vector Machine (SVM)

This hybrid architecture enables the system to detect subtle spoofing
artifacts such as:

-   Moiré patterns
-   Screen pixel grids
-   Display glare
-   Artificial sharpness
-   Frequency-domain inconsistencies

making it suitable for **KYC**, **Identity Verification**, **Face
Authentication**, and **Presentation Attack Detection**.

------------------------------------------------------------------------

# 🎯 Motivation

Presentation attacks are one of the major security threats in digital
identity verification.

Instead of presenting a real face, an attacker may simply show a photo
displayed on a phone, laptop, tablet, or monitor.

While humans can often recognize screen artifacts, conventional CNNs
tend to focus on semantic information rather than fine-grained texture.

AuthentiVision addresses this limitation through **Hybrid Feature
Fusion**, combining deep semantic understanding with handcrafted optical
descriptors.

------------------------------------------------------------------------

# 🏗️ Hybrid Pipeline

``` text
                  Input Image
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼

     ResNet18 CNN            Texture Extraction

 512-D Deep Embedding      FFT
                           LBP
                           GLCM
                           Laplacian
                           Sobel
                           Entropy
                           Color Statistics

        │                             │
        └──────────────┬──────────────┘
                       ▼

              532-D Feature Vector

                       │

                Standard Scaler

                       │

                 SVM Classifier

                       │

               ✅ Real / ❌ Fake
```

------------------------------------------------------------------------

# ✨ Features

-   📷 Live Webcam Scanner
-   📂 Image Upload
-   🔄 Front / Back Camera Switching
-   📉 Client-side Image Compression
-   ⚡ Optimized CPU Inference
-   ☁️ Cloud-Friendly Deployment
-   📊 Latency Benchmark Dashboard
-   💰 Cost Estimation
-   🌙 Responsive Dark UI

------------------------------------------------------------------------

# 🧠 Feature Engineering

## Deep Learning

-   Fine-tuned ResNet18
-   Identity head removed
-   512-dimensional embeddings

## Handcrafted Features

### Frequency Analysis

-   FFT

### Texture Descriptors

-   Local Binary Patterns (LBP)
-   Gray-Level Co-occurrence Matrix (GLCM)

### Edge Features

-   Laplacian Variance
-   Sobel Gradients

### Statistical Features

-   Entropy
-   RGB Mean
-   Variance

------------------------------------------------------------------------

# 📊 Dataset

To prevent train-test leakage:

-   ✅ dHash similarity checking
-   ✅ Group-stratified splitting

  Split          Real   Fake
  ------------ ------ ------
  Train            40     40
  Validation       10     10
  Test             25     25

------------------------------------------------------------------------

# ⚙️ Performance

  Metric                     Value
  ---------------- ---------------
  Mean Latency       **625.99 ms**
  Median Latency     **662.41 ms**
  Minimum            **123.72 ms**
  Maximum            **999.81 ms**
  Threshold               **0.40**

------------------------------------------------------------------------

# 💵 Deployment Cost

  Platform                        Cost
  --------------------- --------------
  On Device                 **\$0.00**
  AWS EC2 (1K Images)     **\$0.0048**
  AWS EC2 (1M Images)       **\$4.82**

------------------------------------------------------------------------

# 🌐 Web Dashboard

### 📷 Live Scanner

-   Live webcam detection
-   Real-time prediction
-   Confidence score
-   Latency measurement
-   Cost estimator

### 📈 System Benchmarks

-   Hardware specifications
-   Latency statistics
-   Cost projections
-   Evaluation methodology

------------------------------------------------------------------------

# 📁 Project Structure

``` text
AuthentiVision/
│
├── models/
├── templates/
├── train/
├── test/
│
├── app.py
├── cnn_train.py
├── train_classifier.py
├── predict.py
├── evaluate.py
├── extract_embeddings.py
├── features.py
├── data_prep.py
├── calculate_parameter.py
├── interpretability.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# 🚀 Installation

``` bash
git clone https://github.com/Saumya1517/FakeImageProject.git

cd FakeImageProject

pip install -r requirements.txt

python app.py
```

------------------------------------------------------------------------

# 🖼️ Demo

Replace these with screenshots from your project.

``` text
docs/
 ├── homepage.png
 ├── live-scanner.png
 ├── prediction.png
 └── benchmark.png
```

Example:

``` md
![Home](docs/homepage.png)

![Scanner](docs/live-scanner.png)

![Prediction](docs/prediction.png)

![Benchmarks](docs/benchmark.png)
```

------------------------------------------------------------------------

# 🛠️ Tech Stack

-   Python
-   PyTorch
-   OpenCV
-   Scikit-Learn
-   NumPy
-   Flask
-   HTML
-   CSS
-   JavaScript

------------------------------------------------------------------------

# 🔮 Future Improvements

-   MobileNetV3 deployment
-   Quantization
-   ONNX export
-   TensorRT optimization
-   Larger multi-device dataset
-   Video-based spoof detection
-   Multi-class presentation attack detection

------------------------------------------------------------------------

# 👩‍💻 Author

**Saumya Agarwal**

B.Tech Computer Science Engineering

Jaypee Institute of Information Technology

GitHub: https://github.com/Saumya1517

------------------------------------------------------------------------

::: {align="center"}
## ⭐ If you found this project useful, please consider giving it a Star!

Made with ❤️ using PyTorch, OpenCV & Flask.
:::


