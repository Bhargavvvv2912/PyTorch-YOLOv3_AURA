import sys
import os
import torch
import numpy as np

def smoke_test():
    print("--- Starting YOLOv3 Smoke Test (Robust) ---")
    
    # 1. Debugging: Where are we? What files exist?
    cwd = os.getcwd()
    print(f"DEBUG: Current Working Directory: {cwd}")
    print(f"DEBUG: Files in CWD: {os.listdir(cwd)}")
    
    # 2. Force Path Setup
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # 3. Flexible Import Strategy
    Darknet = None
    try:
        # Strategy A: Standard root import (common in older commits)
        from models import Darknet
        print("--> Success: Imported 'Darknet' from root 'models.py'")
    except ImportError as e1:
        print(f"    Strategy A failed: {e1}")
        try:
            # Strategy B: Package import (common in newer poetry builds)
            from pytorchyolo.models import Darknet
            print("--> Success: Imported 'Darknet' from 'pytorchyolo.models'")
        except ImportError as e2:
            print(f"    Strategy B failed: {e2}")
            print("CRITICAL ERROR: Could not import 'Darknet' model class.")
            print("Diagnosis: The repository structure does not match expectations.")
            sys.exit(1)

    # 4. Config Setup
    # Look for config in root or package
    config_path = "config/yolov3.cfg"
    if not os.path.exists(config_path):
        # Try alternate location
        config_path = "pytorchyolo/config/yolov3.cfg"
    
    if not os.path.exists(config_path):
        print("--> Config missing. Downloading default YOLOv3 config...")
        os.makedirs("config", exist_ok=True)
        os.system("wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O config/yolov3.cfg")
        config_path = "config/yolov3.cfg"

    # 5. Instantiate Model (The Numpy 2.0 Trap)
    try:
        print(f"--> Loading Darknet model from {config_path}...")
        model = Darknet(config_path, img_size=416)
        
        # 6. Dummy Forward Pass
        model.to('cpu')
        dummy_input = torch.randn(1, 3, 416, 416)
        
        print("--> Running Inference...")
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"--> Inference Successful. Output type: {type(output)}")
        print("--- SMOKE TEST PASSED ---")
        sys.exit(0)

    except AttributeError as e:
        print(f"--- SMOKE TEST FAILED (AttributeError): {e} ---")
        if "np.int" in str(e) or "module 'numpy' has no attribute 'int'" in str(e):
            print("DIAGNOSIS: SUCCESS! Caught the Numpy 2.0 Incompatibility.")
        sys.exit(1)
    except Exception as e:
        print(f"--- SMOKE TEST FAILED: {e} ---")
        sys.exit(1)

if __name__ == "__main__":
    smoke_test()