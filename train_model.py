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
    model_size = "x"  # Change from 'n' (nano) to 'x' (extra large) for highest accuracy
    
    # Load a pre-trained YOLOv8 model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Set training parameters
    params = {
        "data": data_yaml_path,  # Use absolute path to dataset.yaml
        "epochs": 300,           # Increased from 100 to 300 for better convergence
        "imgsz": 1280,           # Increased from 640 to 1280 for better detail capture
        "batch": 8,              # Reduced batch size to accommodate larger model and image size
        "device": device,        # Device to use (cuda or cpu)
        "workers": 4,            # Number of worker threads
        "patience": 50,          # Increased from 20 to 50 to allow more training time before early stopping
        "project": "runs",       # Project directory
        "name": f"train_{timestamp}",  # Run name
        "exist_ok": True,        # Overwrite existing run
        "pretrained": True,      # Use pretrained weights
        "optimizer": "AdamW",    # Changed from Adam to AdamW for better performance
        "lr0": 0.001,            # Reduced initial learning rate for more stable training
        "lrf": 0.001,            # Final learning rate (fraction of lr0)
        "momentum": 0.937,       # SGD momentum/Adam beta1
        "weight_decay": 0.001,   # Increased from 0.0005 to 0.001 for better regularization
        "warmup_epochs": 5.0,    # Increased from 3.0 to 5.0 for better initialization
        "warmup_momentum": 0.8,  # Warmup initial momentum
        "warmup_bias_lr": 0.1,   # Warmup initial bias lr
        "box": 7.5,              # Box loss gain
        "cls": 0.5,              # Cls loss gain
        "dfl": 1.5,              # DFL loss gain
        "save": True,            # Save train checkpoints
        "save_period": 10,       # Save checkpoint every 10 epochs
        "plots": True,           # Save plots during train/val
        "augment": True,         # Added data augmentation for better generalization
        "cos_lr": True,          # Added cosine learning rate scheduler
        "mixup": 0.1,            # Added mixup augmentation
        "copy_paste": 0.1,       # Added copy-paste augmentation
        "degrees": 0.5,          # Rotation augmentation
        "translate": 0.1,        # Translation augmentation
        "scale": 0.5,            # Scale augmentation
        "fliplr": 0.5,           # Horizontal flip probability
        "mosaic": 1.0,           # Mosaic augmentation
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
    
    # Optionally perform model ensemble for even better accuracy
    print("\nTraining ensemble models...")
    ensemble_models = []
    
    # Train 3 additional models with different seeds for ensemble
    for i in range(3):
        ensemble_model = YOLO(f"yolov8{model_size}.pt")
        params["seed"] = i + 42  # Different seed for each model
        params["name"] = f"ensemble_{i}_{timestamp}"
        ensemble_model.train(**params)
        ensemble_models.append(ensemble_model)
    
    print("\nTraining and validation complete!")

if __name__ == "__main__":
    main()
