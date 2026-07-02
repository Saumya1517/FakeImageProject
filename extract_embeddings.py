import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Import our features library
from features import extract_handcrafted_features

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # Load image
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        # Determine label based on directory name
        # data/{split}/{class}/filename
        label = 1 if "fake" in str(Path(path).parent) else 0
        return img, label, str(path)

def extract_all():
    # Setup directories
    data_dir = Path("data")
    models_dir = Path("models")
    
    # 1. Image Paths
    splits = ['train', 'val', 'test']
    image_paths_by_split = {}
    
    for split in splits:
        split_dir = data_dir / split
        # Find all files recursively in this split
        files = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.jpeg"))
        image_paths_by_split[split] = sorted([str(f) for f in files])
        print(f"Found {len(image_paths_by_split[split])} images in {split} split.")
        
    # 2. ImageNet transforms for CNN embedding extraction
    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load fine-tuned CNN model and remove final FC layer
    print("\n--- Loading Fine-Tuned CNN Model ---")
    device = torch.device("cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    cnn_weights_path = models_dir / "best_cnn.pth"
    if cnn_weights_path.exists():
        model.load_state_dict(torch.load(str(cnn_weights_path), map_location=device))
        print("Successfully loaded fine-tuned CNN weights.")
    else:
        print("WARNING: Fine-tuned weights not found. Using pretrained ResNet18 defaults.")
        # Load ImageNet weights as fallback
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
    # Replace classification layer with identity to extract embeddings (512 dims)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    
    # 4. Extract CNN Embeddings and Handcrafted Features
    cnn_embeddings = {}
    handcrafted_features = {}
    labels_dict = {}
    paths_dict = {}
    
    for split in splits:
        paths = image_paths_by_split[split]
        dataset = CustomImageDataset(paths, transform=cnn_transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # A. CNN Embeddings Extraction
        embeddings_list = []
        labels_list = []
        paths_list = []
        
        print(f"Extracting CNN embeddings for {split} split...")
        with torch.no_grad():
            for inputs, labels, img_paths in loader:
                outputs = model(inputs) # Shape: (batch_size, 512)
                embeddings_list.append(outputs.numpy())
                labels_list.extend(labels.numpy())
                paths_list.extend(img_paths)
                
        cnn_embeddings[split] = np.concatenate(embeddings_list, axis=0)
        labels_dict[split] = np.array(labels_list, dtype=np.int64)
        paths_dict[split] = np.array(paths_list)
        
        # B. Handcrafted Features Extraction
        print(f"Extracting handcrafted features for {split} split...")
        hc_list = []
        for path in paths:
            hc_feat = extract_handcrafted_features(path)
            if hc_feat is None:
                # If extraction fails, use a fallback vector of zeros
                hc_feat = np.zeros(20, dtype=np.float32)
            hc_list.append(hc_feat)
            
        handcrafted_features[split] = np.array(hc_list, dtype=np.float32)
        
    # 5. Fit Scaler on training handcrafted features only and transform
    print("\n--- Scaling Handcrafted Features ---")
    scaler = StandardScaler()
    
    # Fit scaler on train only
    train_hc_scaled = scaler.fit_transform(handcrafted_features['train'])
    val_hc_scaled = scaler.transform(handcrafted_features['val'])
    test_hc_scaled = scaler.transform(handcrafted_features['test'])
    
    # Save the scaler
    joblib.dump(scaler, str(models_dir / "scaler.joblib"))
    print("Saved feature scaler to models/scaler.joblib")
    
    # 6. Feature Fusion & Save
    print("\n--- Performing Feature Fusion (CNN + Handcrafted) ---")
    for split, hc_scaled in [('train', train_hc_scaled), ('val', val_hc_scaled), ('test', test_hc_scaled)]:
        cnn_emb = cnn_embeddings[split]
        hybrid = np.concatenate([cnn_emb, hc_scaled], axis=1) # 512 + 20 = 532 dims
        
        # Verify no NaNs or Infs
        if np.isnan(hybrid).any() or np.isinf(hybrid).any():
            print(f"WARNING: NaNs or Infs detected in fused {split} features! Cleaning...")
            hybrid = np.nan_to_num(hybrid)
            
        # Save as npy files
        np.save(str(data_dir / f"{split}_features.npy"), hybrid)
        np.save(str(data_dir / f"{split}_labels.npy"), labels_dict[split])
        np.save(str(data_dir / f"{split}_paths.npy"), paths_dict[split])
        
        # Also save standalone CNN-only and Handcrafted-only features for the ablation study
        np.save(str(data_dir / f"{split}_cnn_only.npy"), cnn_emb)
        np.save(str(data_dir / f"{split}_handcrafted_only.npy"), hc_scaled)
        
        print(f"Saved {split} features: Shape={hybrid.shape}, Labels Shape={labels_dict[split].shape}")

if __name__ == '__main__':
    extract_all()
