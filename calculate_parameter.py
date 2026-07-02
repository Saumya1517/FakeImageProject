import os
import sys
import time
import platform
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).resolve().parent))
from predict import load_models, predict_image

def get_cpu_info():
    """
    Returns a human-readable processor description.
    """
    try:
        # Try importing cpuinfo if available
        import cpuinfo
        return cpuinfo.get_cpu_info()['brand_raw']
    except ImportError:
        pass
    
    # Fallback to platform details
    system = platform.system()
    machine = platform.machine()
    processor = platform.processor()
    
    if system == "Windows":
        # Sometimes platform.processor() is empty or generic, check registry
        import winreg
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            return processor_name.strip()
        except Exception:
            pass
            
    return f"{processor} ({machine})"

def main():
    print("=======================================================")
    # 1. Detect Device & System Info
    print(" DETECTING SYSTEM HARDWARE")
    print("=======================================================")
    cpu_name = get_cpu_info()
    os_name = f"{platform.system()} {platform.release()}"
    python_version = platform.python_version()
    pytorch_device = "CUDA" if torch.cuda.is_available() else "CPU"
    
    print(f"OS:             {os_name}")
    print(f"Processor CPU:  {cpu_name}")
    print(f"Python Version: {python_version}")
    print(f"PyTorch Device: {pytorch_device}")
    
    # 2. Load Models
    print("\nLoading models for latency benchmarking...")
    try:
        cnn_model, scaler, clf, opt_threshold = load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run the pipeline to train models first.")
        sys.exit(1)
    print("Models loaded successfully.")
    
    # 3. Locate Benchmarking Images
    data_dir = Path("data") / "test"
    image_paths = sorted(list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.jpeg")))
    
    if len(image_paths) == 0:
        print(f"Warning: No test images found in {data_dir}. Falling back to train/val folders.")
        data_dir = Path("data")
        image_paths = sorted(list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.jpeg")))
        
    if len(image_paths) == 0:
        print("Error: No images found to run benchmarking.")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images for benchmarking.")
    
    # 4. Warmup Passes (Standard practice to eliminate PyTorch startup overhead)
    print("Running warmup passes...")
    warmup_count = min(5, len(image_paths))
    for i in range(warmup_count):
        try:
            _ = predict_image(image_paths[i], cnn_model, scaler, clf)
        except Exception:
            pass
            
    # 5. Benchmark Latency
    print("Running benchmarking...")
    latencies = []
    
    for path in image_paths:
        start_time = time.perf_counter()
        try:
            _ = predict_image(path, cnn_model, scaler, clf)
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000.0
            latencies.append(duration_ms)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    if len(latencies) == 0:
        print("Error: Benchmark runs failed.")
        sys.exit(1)
        
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    print("\n=======================================================")
    print("                LATENCY BENCHMARK RESULTS              ")
    print("=======================================================")
    print(f"Benchmarked on:  {cpu_name} ({pytorch_device})")
    print(f"Number of runs:  {len(latencies)}")
    print(f"Mean Latency:    {mean_latency:.2f} ms")
    print(f"Median Latency:  {median_latency:.2f} ms")
    print(f"Std Dev:         {std_latency:.2f} ms")
    print(f"Min / Max:       {min_latency:.2f} ms / {max_latency:.2f} ms")
    print(f"Feel:            {'Instant (< 100 ms)' if mean_latency < 100 else 'Noticeable (> 100 ms)'}")
    print("=======================================================")
    
    # 6. Cost Estimation Analysis
    print("\n=======================================================")
    print("             COST ESTIMATION ANALYSIS AT SCALE         ")
    print("=======================================================")
    
    # Cost Option A: On-Device
    print("1. ON-DEVICE DEPLOYMENT (Mobile Phone / Client App)")
    print("   - Cost per image:      $0.00 (Runs locally on user hardware)")
    print("   - Cost per 1M images:  $0.00")
    print("   - Assumptions:         Model is ported to ONNX / CoreML / TensorFlow Lite.")
    print("                          No network latency or server maintenance costs.")
    
    # Cost Option B: Dedicated VM Instance (AWS EC2 t3.medium - 2 vCPUs, 4GB RAM)
    # Hourly cost: $0.0416
    # Let's assume average CPU utilization is 80% and the server is fully utilized.
    # Single-thread capacity = 1.0 / (latency in sec) requests/sec.
    # With 2 vCPUs, let's assume 1.5x throughput under multi-threaded server.
    execution_seconds = mean_latency / 1000.0
    req_per_second = (1.0 / execution_seconds) * 1.5
    req_per_hour = req_per_second * 3600
    
    ec2_cost_per_hour = 0.0416
    ec2_cost_per_image = ec2_cost_per_hour / req_per_hour
    cost_ec2_1k = ec2_cost_per_image * 1_000
    cost_ec2_1m = ec2_cost_per_image * 1_000_000
    
    print("\n2. DEDICATED VM CLOUD DEPLOYMENT (AWS EC2 t3.medium)")
    print(f"   - Hourly VM Instance Cost: ${ec2_cost_per_hour:.4f}")
    print(f"   - Theoretical Max req/hr:  {req_per_hour:,.0f} images/hour (assuming 100% load)")
    print(f"   - Effective Cost / 1k:     ${cost_ec2_1k:.4f}")
    print(f"   - Effective Cost / 1M:     ${cost_ec2_1m:.2f}")
    print("   - Assumptions:         Runs continuously at full/high capacity.")
    print("                          t3.medium (2 vCPUs) handling concurrent requests with a scaling factor of 1.5x.")
    print("=======================================================")

if __name__ == '__main__':
    main()
