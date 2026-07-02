import os
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

def train_hybrid_classifier():
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 1. Load fused features and labels
    print("--- Loading fused features ---")
    X_train = np.load(str(data_dir / "train_features.npy"))
    y_train = np.load(str(data_dir / "train_labels.npy"))
    
    X_val = np.load(str(data_dir / "val_features.npy"))
    y_val = np.load(str(data_dir / "val_labels.npy"))
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    
    # 2. Setup predefined split for hyperparameter tuning
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    # Train indices are marked as -1, validation indices as 0
    test_fold = np.zeros(X_combined.shape[0])
    test_fold[:X_train.shape[0]] = -1
    ps = PredefinedSplit(test_fold=test_fold)
    
    # 3. Define classifiers and hyperparameter grids
    classifiers = {
        'SVM': (SVC(probability=True, random_state=42), {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }),
        'RandomForest': (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 8, None],
            'min_samples_split': [2, 5]
        }),
        'XGBoost': (xgb.XGBClassifier(random_state=42, eval_metric='logloss'), {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }),
        'MLP': (MLPClassifier(random_state=42, max_iter=1000, early_stopping=True), {
            'hidden_layer_sizes': [(64,), (128,), (64, 32)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        })
    }
    
    best_models = {}
    best_val_scores = {}
    
    print("\n--- Tuning Hyperparameters on Validation Set ---")
    for name, (model, grid) in classifiers.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=ps,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_combined, y_combined)
        
        best_est = grid_search.best_estimator_
        best_param = grid_search.best_params_
        
        # Evaluate best estimator on validation split
        y_val_pred_prob = best_est.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_prob)
        val_acc = accuracy_score(y_val, best_est.predict(X_val))
        
        print(f"  Best params: {best_param}")
        print(f"  Val ROC-AUC: {val_auc:.4f} | Val Accuracy: {val_acc:.4f}")
        
        best_models[name] = best_est
        best_val_scores[name] = val_auc
        
    # 4. Select and save best overall classifier
    best_clf_name = max(best_val_scores, key=best_val_scores.get)
    best_clf = best_models[best_clf_name]
    print(f"\n--> Best Overall Classifier: {best_clf_name} with Val ROC-AUC: {best_val_scores[best_clf_name]:.4f}")
    
    joblib.dump(best_clf, str(models_dir / "best_classifier.joblib"))
    print(f"Saved best classifier to models/best_classifier.joblib")
    
    # Save a file with the name of the best classifier for scripting reference
    with open(str(models_dir / "best_classifier_name.txt"), "w") as f:
        f.write(best_clf_name)

if __name__ == '__main__':
    train_hybrid_classifier()
