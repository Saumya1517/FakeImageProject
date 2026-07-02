import os
import sys
import time
import base64
import platform
import numpy as np
import torch
import joblib
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Add current workspace directory to system path
workspace_dir = Path(__file__).resolve().parent
sys.path.append(str(workspace_dir))

from predict import load_models, predict_image
from calculate_parameter import get_cpu_info

app = Flask(__name__, template_folder=str(workspace_dir / "templates"))

# Pre-load models at startup for near-instant real-time web predictions
print("=== Initializing Real vs Fake Web App Backend ===")
try:
    print("Loading AI Models into memory...")
    cnn_model, scaler, clf, opt_threshold = load_models(workspace_dir / "models")
    print("Successfully loaded ResNet18 backbone, standard scaler, and SVM classifier.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load models: {e}")
    print("Please make sure cnn_train.py and train_classifier.py have been executed.")
    sys.exit(1)

# Ensure temp directory exists for uploads
temp_dir = workspace_dir / "temp_uploads"
temp_dir.mkdir(exist_ok=True)

# Hardware and OS parameters
cpu_name = get_cpu_info()
os_name = f"{platform.system()} {platform.release()}"
pytorch_device = "GPU" if torch.cuda.is_available() else "CPU"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/system_info', methods=['GET'])
def system_info():
    return jsonify({
        'cpu_name': cpu_name,
        'os': os_name,
        'pytorch_device': pytorch_device
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    t_start = time.perf_counter()
    data = request.get_json()
    
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
    try:
        # 1. Decode base64 image
        image_data = base64.b64decode(data['image'])
        temp_image_path = temp_dir / f"web_capture_{int(time.time())}.jpg"
        
        # Save temp image file
        with open(temp_image_path, "wb") as f:
            f.write(image_data)
            
        # 2. Extract features and predict image realness
        confidence_real = predict_image(str(temp_image_path), cnn_model, scaler, clf)
        
        # 3. Clean up the temp file
        if temp_image_path.exists():
            os.remove(temp_image_path)
            
        # 4. Determine classification using the loaded decision threshold
        # Threshold was optimized on P(Fake). P(Fake) = 1.0 - P(Real)
        prob_fake = 1.0 - confidence_real
        is_fake = prob_fake >= opt_threshold
        is_real = not is_fake
        
        # 5. Measure latency
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        
        return jsonify({
            'success': True,
            'confidence_real': confidence_real,
            'is_real': is_real,
            'decision_threshold': opt_threshold,
            'latency_ms': latency_ms
        })
        
    except Exception as e:
        print(f"Inference error in API endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Bind to 0.0.0.0 and dynamic port for cloud service deployment (e.g. Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
