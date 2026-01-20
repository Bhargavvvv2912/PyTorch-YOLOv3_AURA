import sys
import os
import torch
import numpy as np

# Add current dir to path
sys.path.append(os.getcwd())

def smoke_test():
    print("--- Starting YOLOv3 Smoke Test ---")
    
    # 1. Check for Config
    config_path = "config/yolov3.cfg"
    if not os.path.exists(config_path):
        print(f"FAIL: Config file not found at {config_path}")
        # If config is missing, we download it (common in this repo)
        os.makedirs("config", exist_ok=True)
        print("--> Downloading yolov3.cfg...")
        os.system("wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O config/yolov3.cfg")
    
    try:
        from models import Darknet
        
        # 2. Instantiate Model
        # CRITICAL: This is where it crashes on Numpy 2.0
        # The 'parse_model_config' function inside uses 'np.int'
        print("--> Loading Darknet model...")
        model = Darknet(config_path, img_size=416)
        
        # 3. Dummy Forward Pass (CPU)
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