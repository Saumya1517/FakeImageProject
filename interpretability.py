import os
import numpy as np
import joblib
from pathlib import Path
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import cv2

def run_interpretability():
    data_dir = Path("data")
    models_dir = Path("models")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Load data
    print("--- Loading data for interpretability analysis ---")
    X_test = np.load(str(data_dir / "test_features.npy"))
    y_test = np.load(str(data_dir / "test_labels.npy"))
    test_paths = np.load(str(data_dir / "test_paths.npy"), allow_pickle=True)
    
    clf = joblib.load(str(models_dir / "best_classifier.joblib"))
    
    # Names of the 20 handcrafted features in order of concatenation (index 512 to 531)
    handcrafted_names = [
        "Color R Mean", "Color G Mean", "Color B Mean",
        "Gray Mean", "Gray Std", "Gray Skew",
        "Laplacian Var (Sharpness)",
        "Shannon Entropy",
        "Canny Edge Ratio", "Sobel Gradient Mean",
        "FFT Low Energy Ratio", "FFT Mid Energy Ratio", "FFT High Energy Ratio",
        "GLCM Contrast", "GLCM Homogeneity", "GLCM Energy", "GLCM Correlation",
        "LBP Bin 0", "LBP Bin 1", "LBP Bin 2"
    ]
    
    # 2. Run Permutation Feature Importance
    print("\n--- Running Permutation Feature Importance ---")
    # This evaluates how much shuffling each feature decreases the ROC-AUC score on the test set
    result = permutation_importance(
        clf, X_test, y_test, scoring='roc_auc', n_repeats=5, random_state=42, n_jobs=-1
    )
    
    importances = result.importances_mean
    
    # Separate CNN and Handcrafted importances
    cnn_importances = importances[:512]
    hc_importances = importances[512:]
    
    sum_cnn = np.sum(np.maximum(cnn_importances, 0))
    sum_hc = np.sum(np.maximum(hc_importances, 0))
    total_imp = sum_cnn + sum_hc + 1e-7
    
    print("\n--- Modality Contribution Summary ---")
    print(f"Total CNN Embeddings (512 features) Importance: {sum_cnn:.4f} ({sum_cnn/total_imp*100:.1f}%)")
    print(f"Total Handcrafted (20 features) Importance:    {sum_hc:.4f} ({sum_hc/total_imp*100:.1f}%)")
    
    # Rank individual handcrafted features
    hc_ranking = sorted(zip(handcrafted_names, hc_importances), key=lambda x: x[1], reverse=True)
    
    print("\n--- Handcrafted Feature Importance Ranking ---")
    for rank, (name, imp) in enumerate(hc_ranking, 1):
        print(f"Rank {rank:02d} | {name:<30} | Importance: {imp:.6f}")
        
    # Save importance plot
    plt.figure(figsize=(10, 8))
    names_sorted = [x[0] for x in hc_ranking[::-1]]
    imps_sorted = [x[1] for x in hc_ranking[::-1]]
    colors = ['skyblue' if x >= 0 else 'lightcoral' for x in imps_sorted]
    
    plt.barh(names_sorted, imps_sorted, color=colors)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel('Mean Decrease in Test ROC-AUC')
    plt.title('Permutation Feature Importance: Handcrafted Features')
    plt.tight_layout()
    plt.savefig(str(plots_dir / "handcrafted_importance.png"))
    plt.close()
    print("Saved handcrafted feature importance plot to plots/handcrafted_importance.png")
    
    # 3. Analyze and Visualize Misclassifications
    print("\n--- Analyzing Misclassified Examples ---")
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Load optimal threshold
    opt_threshold = 0.5
    if (models_dir / "opt_threshold.npy").exists():
        opt_threshold = float(np.load(str(models_dir / "opt_threshold.npy")))
    print(f"Using threshold {opt_threshold:.2f} for misclassification check.")
    
    y_pred = (y_prob >= opt_threshold).astype(int)
    
    false_positives = [] # Real image predicted as fake (Label=0, Pred=1)
    false_negatives = [] # Fake image predicted as real (Label=1, Pred=0)
    
    for i in range(len(y_test)):
        path = test_paths[i]
        label = y_test[i]
        pred = y_pred[i]
        prob = y_prob[i]
        
        if label == 0 and pred == 1:
            false_positives.append((path, prob))
        elif label == 1 and pred == 0:
            false_negatives.append((path, prob))
            
    print(f"False Positives count: {len(false_positives)}")
    for fp_path, prob in false_positives[:3]:
        print(f"  FP: {Path(fp_path).name} (prob={prob:.4f})")
        
    print(f"False Negatives count: {len(false_negatives)}")
    for fn_path, prob in false_negatives[:3]:
        print(f"  FN: {Path(fn_path).name} (prob={prob:.4f})")
        
    # Visualize a grid of misclassifications
    misclassified = []
    for path, prob in false_positives[:2]:
        misclassified.append((path, 0, prob, "False Positive (Real as Fake)"))
    for path, prob in false_negatives[:2]:
        misclassified.append((path, 1, prob, "False Negative (Fake as Real)"))
        
    if misclassified:
        fig, axes = plt.subplots(1, min(len(misclassified), 4), figsize=(15, 4))
        if len(misclassified) == 1:
            axes = [axes]
            
        for idx, (path, label, prob, title) in enumerate(misclassified[:4]):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize image to save memory and space in plotting
            img = cv2.resize(img, (256, 256))
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{title}\nProb={prob:.3f}", fontsize=10)
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig(str(plots_dir / "misclassified_grid.png"))
        plt.close()
        print("Saved misclassifications grid to plots/misclassified_grid.png")
    else:
        print("Perfect classification! No misclassifications to plot.")

if __name__ == '__main__':
    run_interpretability()
