import os
import subprocess
import sys
from pathlib import Path

def run_step(command_list, step_name):
    print(f"\n=======================================================")
    print(f" RUNNING STEP: {step_name}")
    print(f"=======================================================")
    print(f"Executing: {' '.join(command_list)}")
    
    # Run command and pipe output in real-time
    process = subprocess.Popen(
        command_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Print output line by line as it is generated
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[FAILED] STEP: {step_name} (Exit code: {process.returncode})")
        sys.exit(process.returncode)
    else:
        print(f"\n[SUCCESS] STEP: {step_name}")

def main():
    workspace = Path(".")
    venv_python = str(workspace / "venv" / "Scripts" / "python.exe")
    
    if not Path(venv_python).exists():
        print(f"Virtual environment python not found at {venv_python}!")
        print("Please ensure you have created the venv and installed the packages.")
        sys.exit(1)
        
    print("Starting Hybrid Model pipeline runner...")
    
    # Step 1: Data Preparation & Leakage Fix
    run_step([venv_python, "data_prep.py"], "Data Preparation & Leakage Resolution")
    
    # Step 2: CNN Backbone Training
    run_step([venv_python, "cnn_train.py"], "CNN Backbone fine-tuning (ResNet18)")
    
    # Step 3: Embedding Extraction & Fusion
    run_step([venv_python, "extract_embeddings.py"], "Embedding Extraction & Fusion")
    
    # Step 4: Classifier Training & Hyperparameter Tuning
    run_step([venv_python, "train_classifier.py"], "Classifier Training & Hyperparameter Tuning")
    
    # Step 5: Evaluation
    run_step([venv_python, "evaluate.py"], "Model Evaluation & Ablation Study")
    
    # Step 6: Interpretability
    run_step([venv_python, "interpretability.py"], "Feature Importance & Error Analysis")
    
    print("\n=======================================================")
    print(" HYBRID MODEL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == '__main__':
    main()
