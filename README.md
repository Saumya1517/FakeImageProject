👁️ AuthentiVision
Hybrid Feature Fusion for Screen-Retake Image Detection
<p align="center">
  
</p>

🛡️ Overview

AuthentiVision is a Hybrid Feature Fusion framework for detecting screen-retaken (presentation attack) images.

Unlike conventional CNN classifiers, AuthentiVision combines:

🧠 Deep CNN embeddings (ResNet18)
🔍 Traditional texture descriptors
⚡ Support Vector Machine classification

to identify subtle artifacts such as

Screen pixels
Moiré patterns
Panel glare
Artificial sharpness
Frequency-domain inconsistencies

making it suitable for KYC, Face Verification, and Identity Authentication systems.

🎯 Motivation

Presentation attacks remain one of the biggest vulnerabilities in digital identity verification.

An attacker can simply display another person's photo on

📱 Phone
💻 Laptop
🖥️ Monitor
📟 Tablet

and fool traditional face verification systems.

Humans easily notice

glare
screen pixels
moiré patterns

while CNNs often ignore these fine textures.

AuthentiVision bridges this gap using Hybrid Feature Fusion.
🏗 Architecture
                  Input Image
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼

  ResNet18 Backbone           Texture Extraction

 512-D CNN Embedding          FFT
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

✨ Features

✔ Hybrid CNN + Computer Vision pipeline
✔ Presentation Attack Detection
✔ Live Webcam Scanner
✔ Image Upload
✔ Front / Back Camera Switching
✔ Browser-side Image Compression
✔ Cloud Optimized
✔ Latency Benchmark Dashboard
✔ Cost Estimator
✔ Responsive UI

🧠 Feature Engineering
Deep Features
Fine-tuned ResNet18
Identity Head
512-dimensional embeddings
Handcrafted Features
Frequency Analysis
FFT

Detects

Screen frequencies
Moiré artifacts
Texture Features
LBP
GLCM

Extracts

Contrast
Correlation
Energy
Homogeneity
Edge Features
Laplacian Variance
Sobel Gradients

Measures

Blur
Focus
Artificial edge transitions
Statistical Features
Entropy
RGB Mean
Variance
📊 Dataset

To prevent train-test leakage,

✔ dHash similarity matching

✔ Group-based splitting

Dataset

Split	Real	Fake
Train	40	40
Validation	10	10
Test	25	25
⚡ Performance
Latency
Metric	Value
Mean	625.99 ms
Median	662.41 ms
Min	123.72 ms
Max	999.81 ms
Threshold

Optimized Screen-Retake probability threshold

0.40
Deployment Cost
Platform	Cost
On Device	$0.00
AWS EC2 (1K Images)	$0.0048
AWS EC2 (1M Images)	$4.82
🌐 Web Dashboard

The Flask dashboard provides

📷 Live Scanner
Webcam Detection
Real-time Prediction
Confidence Score
Latency
Cost Estimation
📈 Benchmarks
Hardware Specs
Latency Statistics
Cost Projection
Deployment Methodology

🛠 Tech Stack
Python
PyTorch
OpenCV
NumPy
Scikit-Learn
Flask
HTML
CSS
JavaScript

🎯 Future Improvements
MobileNetV3 Deployment
TensorRT Optimization
ONNX Export
Quantization
Larger Cross-device Dataset
Multi-class Presentation Attack Detection
Video-based Detection

⭐ If you found this project useful, consider giving it a star!
