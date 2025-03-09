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
    
    # Model configuration
    model_size = "n"  # nano size (options: n, s, m, l, x)
    
    # Load a pre-trained YOLOv8 model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Set training parameters
    params = {
        "data": "dataset.yaml",  # Path to data config file
        "epochs": 100,           # Number of training epochs
        "imgsz": 640,            # Image size
        "batch": 16,             # Batch size
        "device": device,        # Device to use (cuda or cpu)
        "workers": 4,            # Number of worker threads
        "patience": 20,          # Early stopping patience
        "project": "runs",       # Project directory
        "name": f"train_{timestamp}",  # Run name
        "exist_ok": True,        # Overwrite existing run
        "pretrained": True,      # Use pretrained weights
        "optimizer": "Adam",     # Optimizer (SGD, Adam, AdamW, etc.)
        "lr0": 0.01,             # Initial learning rate
        "lrf": 0.01,             # Final learning rate (fraction of lr0)
        "momentum": 0.937,       # SGD momentum/Adam beta1
        "weight_decay": 0.0005,  # Optimizer weight decay
        "warmup_epochs": 3.0,    # Warmup epochs
        "warmup_momentum": 0.8,  # Warmup initial momentum
        "warmup_bias_lr": 0.1,   # Warmup initial bias lr
        "box": 7.5,              # Box loss gain
        "cls": 0.5,              # Cls loss gain
        "dfl": 1.5,              # DFL loss gain
        "save": True,            # Save train checkpoints
        "save_period": -1,       # Save checkpoint every x epochs (disabled if < 1)
        "plots": True,           # Save plots during train/val
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
    
    print("\nTraining and validation complete!")

if __name__ == "__main__":
    main()
