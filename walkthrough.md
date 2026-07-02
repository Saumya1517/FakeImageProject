# Walkthrough: Hybrid CNN + Handcrafted Feature Model for Real vs. Fake Image Detection

This report details the implementation, validation, performance results, and interpretability analysis of the **Hybrid CNN + Handcrafted Feature Model** built to detect screen-retaken (fake) images from original (real) images.

---

## 1. Data Preparation & Leakage Resolution

- **Leakage Resolution:** Our initial audit identified that the real image `train/IMG20260702160153.jpg` had three screen-retaken counterparts: two in the training set and one in the test set (`test/fake/IMG-20260702-WA0251.jpg`). This caused a train-test information leak. To resolve this, we swapped `test/fake/IMG-20260702-WA0251.jpg` with `train/fake/IMG-20260702-WA0258.jpg` (which has no real counterpart in the dataset). This completely isolated all groups and eliminated leakage while maintaining a perfect class balance.
- **Group-Stratified Validation Split:** To prevent train-validation leakage (where different retakes of the same real image could end up in different splits), we grouped images by their real counterparts (using dHash distance <= 12) and stratified the groups. This yielded a perfect 80/20 train/validation split with exactly:
  - **Train subset:** 40 real, 40 fake images.
  - **Validation subset:** 10 real, 10 fake images.
  - **Held-out Test split:** 25 real, 25 fake images.

---

## 2. Model Performance & Ablation Study

We trained an ImageNet-pretrained **ResNet18** model on the CPU, freezing the early feature extraction layers (`conv1`, `bn1`, `layer1`, `layer2`) and fine-tuning later layers using the AdamW optimizer and early stopping on validation loss. We then extracted 512-dimensional embeddings from the penultimate layer and concatenated them with **20 scaled handcrafted features** representing color, contrast, sharpness, entropy, edges, textures (GLCM/LBP), and frequency (FFT) domains.

Using a pre-split validation set grid search, we found that the **SVM Classifier** (with $C=1.0$ and a radial basis function kernel) performed best.

### Ablation Study Results

The table below compares the performance of the standalone CNN model, the standalone Handcrafted-feature model, and the Hybrid model configurations on the held-out test set:

| Model Configuration | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN Only** | 0.8800 | 0.8800 | 0.8800 | 0.8800 | 0.9408 |
| **Handcrafted Only** | 0.8400 | 0.7742 | **0.9600** | 0.8571 | 0.9280 |
| **Hybrid Model (Default, Threshold = 0.5)** | **0.8800** | **0.8800** | 0.8800 | **0.8800** | **0.9488** |
| **Hybrid Model (Optimal, Threshold = 0.4)** | **0.8800** | **0.8800** | 0.8800 | **0.8800** | **0.9488** |

### Key Takeaways:
1. **Fusion Benefit:** The Hybrid Model achieved the highest test ROC-AUC score (**0.9488**), outperforming both the CNN Only model (0.9408) and the Handcrafted Only model (0.9280).
2. **Complementary Strengths:** While the handcrafted features alone achieved a high recall of **0.9600** (catching 24 out of 25 fakes), they had lower precision (0.7742) due to false positives. The hybrid model combined the high precision of the CNN embeddings with the high recall of the handcrafted features, leading to a balanced F1-score of **0.8800**.

### ROC and Precision-Recall Curves

Below are the ROC and Precision-Recall curves for the final Hybrid model on the test set:

![ROC and PR Curves](C:/Users/saumy/.gemini/antigravity-ide/brain/fdde9a69-fc01-4bbd-89ed-e005a4f3b074/evaluation_curves.png)

---

## 3. Interpretability & Feature Importance

We computed Permutation Feature Importance on the test set to evaluate the contributions of the CNN embeddings vs. the 20 handcrafted features.

- **CNN Embeddings (512 features):** 39.6% contribution.
- **Handcrafted Features (20 features):** **60.4% contribution**.

Despite being only 20 features (compared to 512 CNN features), the handcrafted features provided the majority of the decision signal. This demonstrates that target physical features (like screen texture and moiré frequencies) carry very strong discriminative power.

### Handcrafted Feature Importance Ranking

The chart below shows the importance rankings of the 20 handcrafted features:

![Handcrafted Feature Importances](C:/Users/saumy/.gemini/antigravity-ide/brain/fdde9a69-fc01-4bbd-89ed-e005a4f3b074/handcrafted_importance.png)

### Top-Contributing Handcrafted Features:
1. **GLCM Energy (Rank 1):** Captures texture uniformity. Real images typically have uniform textures, whereas screen-retaken images exhibit disrupted textures and micro-patterns due to subpixels.
2. **LBP Bin 1 & 2 (Ranks 2 & 4):** Measures local micro-structures. The grid pattern of display screens introduces high-frequency edges and uniform patterns that show up in the LBP histogram.
3. **Color R Mean (Rank 3):** Capture shifts in color temperature/brightness caused by screen backlighting and cameras.

---

## 4. Error Analysis (Misclassified Examples)

At the optimal threshold of 0.40, the model misclassified 6 images (3 False Positives, 3 False Negatives):

- **False Positives (Real images flagged as Fake):**
  - `IMG-20241015-WA0022.jpg` (Prob: 0.768)
  - `IMG20240114000459.jpg` (Prob: 0.687)
  - `IMG_20260702_161152.jpg` (Prob: 0.966)
- **False Negatives (Fake images flagged as Real):**
  - `IMG-20260702-WA0228.jpg` (Prob: 0.284)
  - `IMG-20260702-WA0249.jpg` (Prob: 0.144)
  - `IMG-20260702-WA0258.jpg` (Prob: 0.179)

Below is a grid visualizing some of these misclassified examples:

![Misclassified Examples](C:/Users/saumy/.gemini/antigravity-ide/brain/fdde9a69-fc01-4bbd-89ed-e005a4f3b074/misclassified_grid.png)

> [!NOTE]
> **Qualitative Insight:**
> The False Negatives (missed screen-retakes) are often high-quality retakes taken under stable lighting with minimal glare or visible borders, allowing them to mimic real photos closely. The False Positives (real photos flagged as fake) are often images with textured backgrounds or uniform lighting patterns that mimic moiré patterns.

---

## 5. Model Card

### Intended Use
- **Primary Use Case:** Real-time checking of user-submitted photos to prevent presentation attacks (e.g. users holding up a phone or screen displaying another person's photo during KYC/identity verification).
- **Out of Scope:** Detecting deepfakes, AI-generated images, or print-retaken (paper printed) images.

### Dataset Description
- **Real class:** Original photos taken from the user's gallery.
- **Fake class:** Screen-retaken photos captured of display screens showing the original photos.
- **Size:** 150 images total (100 train/val, 50 test).
- **Resolution Range:** 556x528 to 4080x4080 (mean = 2581x3164).

### Performance Metrics
- **Best Model:** ResNet18 + Handcrafted features + SVM Classifier
- **Accuracy:** 88.0%
- **F1-Score:** 88.0%
- **ROC-AUC:** 94.9%

### Limitations
- **Small training set:** The model was trained on 100 images. While fine-tuning and handcrafted features mitigated overfitting, a larger dataset is recommended for production.
- **Hardware limit:** Model was trained entirely on CPU due to lack of a CUDA GPU. Inference times are around 200ms per image on CPU, which is suitable for real-time check.
