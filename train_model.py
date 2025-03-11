import os
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create a timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create runs directory if it doesn't exist
    os.makedirs("runs", exist_ok=True)
    
    # Get the absolute path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the absolute path to dataset.yaml
    data_yaml_path = os.path.join(current_dir, "dataset.yaml")
    
    # Model configuration
    model_size = "l"  # Changed from 'x' to 'l' (large) to reduce memory requirements
    
    # Load a pre-trained YOLOv8 model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Set training parameters
    params = {
        "data": data_yaml_path,  # Use absolute path to dataset.yaml
        "epochs": 300,           # Keep high epoch count for accuracy
        "imgsz": 1024,           # Reduced from 1280 to 1024 to save memory
        "batch": 4,              # Reduced batch size from 8 to 4 to save memory
        "device": device,        # Device to use (cuda or cpu)
        "workers": 4,            # Number of worker threads
        "patience": 50,          # Keep high patience for better accuracy
        "project": "runs",       # Project directory
        "name": f"train_{timestamp}",  # Run name
        "exist_ok": True,        # Overwrite existing run
        "pretrained": True,      # Use pretrained weights
        "optimizer": "AdamW",    # Keep AdamW for better performance
        "lr0": 0.001,            # Keep reduced initial learning rate
        "lrf": 0.001,            # Final learning rate
        "momentum": 0.937,       # SGD momentum/Adam beta1
        "weight_decay": 0.001,   # Keep increased weight decay
        "warmup_epochs": 5.0,    # Keep increased warmup epochs
        "warmup_momentum": 0.8,  # Warmup initial momentum
        "warmup_bias_lr": 0.1,   # Warmup initial bias lr
        "box": 7.5,              # Box loss gain
        "cls": 0.5,              # Cls loss gain
        "dfl": 1.5,              # DFL loss gain
        "save": True,            # Save train checkpoints
        "save_period": 10,       # Save checkpoint every 10 epochs
        "plots": True,           # Save plots during train/val
        "augment": True,         # Keep data augmentation
        "cos_lr": True,          # Keep cosine learning rate scheduler
        "mixup": 0.1,            # Keep mixup augmentation
        "copy_paste": 0.1,       # Keep copy-paste augmentation
        "degrees": 0.5,          # Keep rotation augmentation
        "translate": 0.1,        # Keep translation augmentation
        "scale": 0.5,            # Keep scale augmentation
        "fliplr": 0.5,           # Keep horizontal flip probability
        "mosaic": 1.0,           # Keep mosaic augmentation
        "cache": True,           # Add caching to improve memory efficiency
        "torch_compile": False,  # Disable torch compile which can use extra memory
        "cuda_alloc_conf": {"expandable_segments": True},  # Add CUDA memory allocation config
    }
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Model: YOLOv8{model_size}")
    print(f"Dataset: {params['data']}")
    print(f"Epochs: {params['epochs']}")
    print(f"Image Size: {params['imgsz']}")
    print(f"Batch Size: {params['batch']}")
    print(f"Device: {params['device']}")
    
    # Start training
    print("\nStarting training...")
    results = model.train(**params)
    
    # Print training results
    print("\nTraining completed!")
    print(f"Results saved to {os.path.join(params['project'], params['name'])}")
    
    # Validate the model
    print("\nValidating model...")
    val_results = model.val()
    
    # Export the model to different formats
    print("\nExporting model...")
    model.export(format="onnx")  # Export to ONNX format
    
    # Modified ensemble approach to avoid OOM errors
    print("\nTraining ensemble models sequentially to avoid memory issues...")
    ensemble_results = []
    
    # Train additional models one at a time with different seeds
    for i in range(2):  # Reduced from 3 to 2 models
        print(f"\nTraining ensemble model {i+1}/2...")
        ensemble_model = YOLO(f"yolov8{model_size}.pt")
        params["seed"] = i + 42  # Different seed for each model
        params["name"] = f"ensemble_{i}_{timestamp}"
        ensemble_model.train(**params)
        # Save results and clear model from GPU memory
        ensemble_results.append(params["name"])
        del ensemble_model
        torch.cuda.empty_cache()  # Clear GPU cache between models
    
    print(f"\nEnsemble models trained and saved to: {', '.join(ensemble_results)}")
    print("\nTraining and validation complete!")

if __name__ == "__main__":
    main()
