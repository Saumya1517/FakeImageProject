import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib
from pathlib import Path

# Add current directory to path just in case
sys.path.append(str(Path(__file__).resolve().parent))
from features import extract_handcrafted_features

def load_models(models_dir=None):
    """
    Loads the CNN backbone, standard scaler, trained classifier, and decision threshold.
    """
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent / "models"
        
    device = torch.device("cpu")
    
    # 1. Load CNN weights into ResNet18 structure
    cnn_model = models.resnet18()
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
    
    cnn_weights_path = models_dir / "best_cnn.pth"
    if cnn_weights_path.exists():
        cnn_model.load_state_dict(torch.load(str(cnn_weights_path), map_location=device))
    else:
        raise FileNotFoundError(f"Fine-tuned CNN weights not found at {cnn_weights_path}")
        
    # Replace FC layer with Identity to extract 512-dimensional embeddings
    cnn_model.fc = nn.Identity()
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    
    # 2. Load the Scaler
    scaler_path = models_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(str(scaler_path))
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
    # 3. Load the Classifier
    clf_path = models_dir / "best_classifier.joblib"
    if clf_path.exists():
        clf = joblib.load(str(clf_path))
    else:
        raise FileNotFoundError(f"Classifier not found at {clf_path}")
        
    # 4. Load Decision Threshold
    opt_threshold_path = models_dir / "opt_threshold.npy"
    if opt_threshold_path.exists():
        opt_threshold = float(np.load(str(opt_threshold_path)))
    else:
        opt_threshold = 0.5
        
    return cnn_model, scaler, clf, opt_threshold

def predict_image(image_path, cnn_model, scaler, clf):
    """
    Infers the realness of an image.
    Returns:
        float: Realness confidence score in range [0, 1] where 1 is Real and 0 is Fake.
    """
    # 1. Extract Handcrafted Features (20 dimensions)
    hc_feat = extract_handcrafted_features(image_path)
    if hc_feat is None:
        raise ValueError(f"Could not extract handcrafted features from {image_path}")
        
    # Scale handcrafted features
    hc_feat_scaled = scaler.transform(hc_feat.reshape(1, -1)) # Shape: (1, 20)
    
    # 2. Extract CNN Embeddings (512 dimensions)
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Could not load image file {image_path}: {e}")
        
    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = cnn_transform(img).unsqueeze(0) # Shape: (1, 3, 224, 224)
    
    with torch.no_grad():
        cnn_emb = cnn_model(img_t).numpy() # Shape: (1, 512)
        
    # 3. Feature Fusion (512 + 20 = 532 dimensions)
    fused_features = np.concatenate([cnn_emb, hc_feat_scaled], axis=1)
    
    # 4. Run Classifier Inference
    # predict_proba returns shape (1, 2) where class 0 is Real, class 1 is Fake
    probs = clf.predict_proba(fused_features)[0]
    prob_real = float(probs[0])
    
    return prob_real

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)
        
    try:
        cnn_model, scaler, clf, opt_threshold = load_models()
        confidence_real = predict_image(image_path, cnn_model, scaler, clf)
        
        # Determine classification using optimal threshold (defined on fake class probability)
        # P(Fake) = 1.0 - P(Real)
        prob_fake = 1.0 - confidence_real
        is_fake = prob_fake >= opt_threshold
        decision = "Fake" if is_fake else "Real"
        
        print(f"Confidence Score (Real): {confidence_real:.4f}")
        print(f"Predicted Class:          {decision} (using optimized fake-threshold = {opt_threshold})")
        
    except Exception as e:
        print(f"Prediction error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
