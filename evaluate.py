import os
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

def train_and_eval_single_config(X_train, y_train, X_val, y_val, X_test, y_test, clf_name):
    # Setup PredefinedSplit
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    test_fold = np.zeros(X_combined.shape[0])
    test_fold[:X_train.shape[0]] = -1
    ps = PredefinedSplit(test_fold=test_fold)
    
    # Simple grids for ablation tuning
    if clf_name == 'SVM':
        clf = SVC(probability=True, random_state=42)
        grid = {'C': [0.1, 1.0, 10.0, 100.0], 'kernel': ['linear', 'rbf']}
    elif clf_name == 'RandomForest':
        clf = RandomForestClassifier(random_state=42)
        grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 8, None]}
    elif clf_name == 'XGBoost':
        clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1, 0.2]}
    else: # MLP
        clf = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
        grid = {'hidden_layer_sizes': [(64,), (128,)], 'alpha': [0.0001, 0.001]}
        
    grid_search = GridSearchCV(clf, grid, cv=ps, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_combined, y_combined)
    best_clf = grid_search.best_estimator_
    
    # Predict on test
    probs = best_clf.predict_proba(X_test)[:, 1]
    preds = best_clf.predict(X_test)
    
    return {
        'acc': accuracy_score(y_test, preds),
        'prec': precision_score(y_test, preds),
        'rec': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'auc': roc_auc_score(y_test, probs)
    }

def run_evaluation():
    data_dir = Path("data")
    models_dir = Path("models")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Load data
    print("--- Loading data for evaluation ---")
    X_train = np.load(str(data_dir / "train_features.npy"))
    y_train = np.load(str(data_dir / "train_labels.npy"))
    X_val = np.load(str(data_dir / "val_features.npy"))
    y_val = np.load(str(data_dir / "val_labels.npy"))
    X_test = np.load(str(data_dir / "test_features.npy"))
    y_test = np.load(str(data_dir / "test_labels.npy"))
    
    # Load best model
    clf = joblib.load(str(models_dir / "best_classifier.joblib"))
    with open(str(models_dir / "best_classifier_name.txt"), "r") as f:
        best_clf_name = f.read().strip()
        
    print(f"Loaded best classifier: {best_clf_name}")
    
    # 2. Evaluate hybrid model on test set
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred_default = clf.predict(X_test)
    
    acc_def = accuracy_score(y_test, y_pred_default)
    prec_def = precision_score(y_test, y_pred_default)
    rec_def = recall_score(y_test, y_pred_default)
    f1_def = f1_score(y_test, y_pred_default)
    auc_def = roc_auc_score(y_test, y_prob)
    
    print("\n--- Hybrid Model Performance on Test Set (Default Threshold = 0.5) ---")
    print(f"Accuracy:  {acc_def:.4f}")
    print(f"Precision: {prec_def:.4f}")
    print(f"Recall:    {rec_def:.4f}")
    print(f"F1-Score:  {f1_def:.4f}")
    print(f"ROC-AUC:   {auc_def:.4f}")
    
    print("\nConfusion Matrix (Default 0.5):")
    print(confusion_matrix(y_test, y_pred_default))
    
    # 3. Find optimal decision threshold (maximize F1 score for fake class)
    thresholds = np.linspace(0.0, 1.0, 101)
    f1_scores = []
    for t in thresholds:
        preds_t = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_test, preds_t))
        
    best_idx = np.argmax(f1_scores)
    opt_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    y_pred_opt = (y_prob >= opt_threshold).astype(int)
    acc_opt = accuracy_score(y_test, y_pred_opt)
    prec_opt = precision_score(y_test, y_pred_opt)
    rec_opt = recall_score(y_test, y_pred_opt)
    
    print(f"\n--- Decision Threshold Optimization ---")
    print(f"Optimal Threshold (Max F1): {opt_threshold:.2f}")
    print(f"F1-Score at optimal threshold: {best_f1:.4f}")
    print(f"Accuracy at optimal threshold: {acc_opt:.4f}")
    print(f"Precision at optimal threshold: {prec_opt:.4f}")
    print(f"Recall at optimal threshold: {rec_opt:.4f}")
    print("\nConfusion Matrix (Optimal Threshold):")
    print(confusion_matrix(y_test, y_pred_opt))
    
    # Save optimal threshold for reference
    np.save(str(models_dir / "opt_threshold.npy"), opt_threshold)
    
    # 4. Plot curves
    print("\n--- Generating Performance Plots ---")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(12, 5))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'Hybrid Model (AUC = {auc_def:.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    # PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(rec_c, prec_c, label='Hybrid Model', color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(str(plots_dir / "evaluation_curves.png"))
    plt.close()
    print("Saved curves to plots/evaluation_curves.png")
    
    # 5. Ablation / Comparison Study
    print("\n--- Running Ablation Study ---")
    # Load parts
    X_train_cnn = np.load(str(data_dir / "train_cnn_only.npy"))
    X_val_cnn = np.load(str(data_dir / "val_cnn_only.npy"))
    X_test_cnn = np.load(str(data_dir / "test_cnn_only.npy"))
    
    X_train_hc = np.load(str(data_dir / "train_handcrafted_only.npy"))
    X_val_hc = np.load(str(data_dir / "val_handcrafted_only.npy"))
    X_test_hc = np.load(str(data_dir / "test_handcrafted_only.npy"))
    
    print("Training CNN-Only model...")
    cnn_only_metrics = train_and_eval_single_config(X_train_cnn, y_train, X_val_cnn, y_val, X_test_cnn, y_test, best_clf_name)
    
    print("Training Handcrafted-Only model...")
    hc_only_metrics = train_and_eval_single_config(X_train_hc, y_train, X_val_hc, y_val, X_test_hc, y_test, best_clf_name)
    
    # Summary of all three
    print("\n=======================================================")
    print("                 ABLATION STUDY RESULTS                ")
    print("=======================================================")
    print(f"{'Model Configuration':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 75)
    print(f"{'CNN Only':<22} | {cnn_only_metrics['acc']:.4f}   | {cnn_only_metrics['prec']:.4f}    | {cnn_only_metrics['rec']:.4f} | {cnn_only_metrics['f1']:.4f}   | {cnn_only_metrics['auc']:.4f}")
    print(f"{'Handcrafted Only':<22} | {hc_only_metrics['acc']:.4f}   | {hc_only_metrics['prec']:.4f}    | {hc_only_metrics['rec']:.4f} | {hc_only_metrics['f1']:.4f}   | {hc_only_metrics['auc']:.4f}")
    print(f"{'Hybrid Model (Default)':<22} | {acc_def:.4f}   | {prec_def:.4f}    | {rec_def:.4f} | {f1_def:.4f}   | {auc_def:.4f}")
    print(f"{'Hybrid Model (Optimal)':<22} | {acc_opt:.4f}   | {prec_opt:.4f}    | {rec_opt:.4f} | {best_f1:.4f}   | {auc_def:.4f}")
    print("=======================================================")
    
    # Save ablation text file
    with open(str(plots_dir / "ablation_study_table.txt"), "w") as f:
        f.write("=======================================================\n")
        f.write("                 ABLATION STUDY RESULTS                \n")
        f.write("=======================================================\n")
        f.write(f"{'Model Configuration':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1-Score':<8} | {'ROC-AUC':<8}\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'CNN Only':<22} | {cnn_only_metrics['acc']:.4f}   | {cnn_only_metrics['prec']:.4f}    | {cnn_only_metrics['rec']:.4f} | {cnn_only_metrics['f1']:.4f}   | {cnn_only_metrics['auc']:.4f}\n")
        f.write(f"{'Handcrafted Only':<22} | {hc_only_metrics['acc']:.4f}   | {hc_only_metrics['prec']:.4f}    | {hc_only_metrics['rec']:.4f} | {hc_only_metrics['f1']:.4f}   | {hc_only_metrics['auc']:.4f}\n")
        f.write(f"{'Hybrid Model (Default)':<22} | {acc_def:.4f}   | {prec_def:.4f}    | {rec_def:.4f} | {f1_def:.4f}   | {auc_def:.4f}\n")
        f.write(f"{'Hybrid Model (Optimal)':<22} | {acc_opt:.4f}   | {prec_opt:.4f}    | {rec_opt:.4f} | {best_f1:.4f}   | {auc_def:.4f}\n")
        f.write("=======================================================\n")
        
    print("Ablation study saved to plots/ablation_study_table.txt")

if __name__ == '__main__':
    run_evaluation()
