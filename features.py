import os
import cv2
import numpy as np
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def extract_handcrafted_features(image_path):
    """
    Extracts exactly 20 handcrafted features from an image:
    1-3.   Color means (R, G, B)
    4-6.   Grayscale Brightness/Contrast (Mean, Std, Skewness)
    7.     Sharpness (Laplacian variance)
    8.     Entropy (Shannon entropy)
    9.     Edge Density (Canny edge ratio)
    10.    Edge Magnitude (Sobel gradient magnitude mean)
    11-13. FFT frequency energy ratios (Low, Mid, High frequency bands)
    14-17. GLCM texture features (Contrast, Homogeneity, Energy, Correlation)
    18-20. LBP texture features (First 3 bins of normalized LBP uniform histogram)
    """
    try:
        # Load image with OpenCV (BGR)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # 1-3. Color means (R, G, B)
        mean_r, mean_g, mean_b = cv2.mean(img_rgb)[:3]
        features.extend([mean_r, mean_g, mean_b])
        
        # 4-6. Grayscale Brightness/Contrast (Mean, Std, Skewness)
        mean_gray, std_gray = cv2.meanStdDev(gray)
        mean_gray = float(mean_gray[0][0])
        std_gray = float(std_gray[0][0])
        
        # Vectorized skewness calculation (identical to scipy.stats.skew but faster)
        diff = gray.astype(np.float32) - mean_gray
        skew_gray = float(np.mean(diff ** 3) / (std_gray ** 3 + 1e-7))
        features.extend([mean_gray, std_gray, skew_gray])
        
        # 7. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(laplacian)
        features.append(lap_var)
        
        # 8. Entropy (Shannon entropy)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        probs = hist / np.sum(hist)
        ent = -np.sum(probs * np.log2(probs + 1e-7))
        features.append(ent)
        
        # 9. Edge Density (Canny edge ratio)
        # Normalize/resize image temporarily to 512x512 for consistent edge density calculation
        gray_resized = cv2.resize(gray, (512, 512))
        edges = cv2.Canny(gray_resized, 100, 200)
        edge_ratio = np.mean(edges > 0)
        features.append(edge_ratio)
        
        # 10. Edge Magnitude (Sobel gradient magnitude mean)
        sobel_x = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        mean_sobel_mag = np.mean(mag)
        features.append(mean_sobel_mag)
        
        # 11-13. FFT frequency energy ratios
        fft = np.fft.fft2(gray_resized)
        fft_shift = np.fft.fftshift(fft)
        mag_spectrum = np.abs(fft_shift)
        
        # Compute coordinates relative to center
        h, w = gray_resized.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        r = np.sqrt(x**2 + y**2)
        max_r = np.sqrt(cy**2 + cx**2)
        
        # Energy in bands
        energy_total = np.sum(mag_spectrum) + 1e-7
        low_band = r <= (0.1 * max_r)
        mid_band = (r > (0.1 * max_r)) & (r <= (0.5 * max_r))
        high_band = r > (0.5 * max_r)
        
        energy_low = np.sum(mag_spectrum[low_band]) / energy_total
        energy_mid = np.sum(mag_spectrum[mid_band]) / energy_total
        energy_high = np.sum(mag_spectrum[high_band]) / energy_total
        features.extend([energy_low, energy_mid, energy_high])
        
        # 14-17. GLCM texture features
        # Quantize to 16 gray levels for GLCM to speed up and reduce sparsity
        gray_16 = (gray_resized // 16).astype(np.uint8)
        glcm = graycomatrix(gray_16, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)
        
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        features.extend([contrast, homogeneity, energy, correlation])
        
        # 18-20. LBP texture features (First 3 bins of LBP uniform histogram)
        lbp = local_binary_pattern(gray_resized, P=8, R=1, method='uniform')
        # Uniform LBP has 10 possible values (0 to 9)
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=10, range=(0, 10), density=True)
        features.extend([lbp_hist[0], lbp_hist[1], lbp_hist[2]])
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def extract_features_batch(image_paths):
    feats_list = []
    valid_paths = []
    for path in image_paths:
        feat = extract_handcrafted_features(path)
        if feat is not None:
            feats_list.append(feat)
            valid_paths.append(path)
    return np.array(feats_list, dtype=np.float32), valid_paths

if __name__ == '__main__':
    # Simple testing when run as main
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        feat = extract_handcrafted_features(path)
        if feat is not None:
            print("Extracted features (shape={}):".format(feat.shape))
            print(feat)
        else:
            print("Failed to extract features.")
    else:
        print("Please provide an image path to test.")
