import os
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import random
import numpy as np

def seed_everything(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Set seed for reproducibility
    seed_everything(42)
    
    # Check for GPU availability and use all available GPUs
    device = "0,1,2,3" if torch.cuda.device_count() > 1 else "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create a timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create runs directory if it doesn't exist
    os.makedirs("runs", exist_ok=True)
    
    # Get the absolute path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the absolute path to dataset.yaml
    data_yaml_path = os.path.join(current_dir, "dataset.yaml")
    
    # Model configuration - using the largest, most accurate model
    model_size = "x"  # Use the largest model for best accuracy
    
    # Load a pre-trained YOLOv8 model (or checkpoint if resuming)
    checkpoint_path = os.path.join(current_dir, "runs", "train_latest", "weights", "best.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        # Load pretrained model from COCO
        model = YOLO(f"yolov8{model_size}.pt")
    
    # Set up run directory
    run_name = f"train_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create a symbolic link to the latest run
    latest_link = os.path.join("runs", "train_latest")
    if os.path.islink(latest_link):
        os.remove(latest_link)
    elif os.path.exists(latest_link):
        os.rename(latest_link, os.path.join("runs", f"train_backup_{timestamp}"))
    os.symlink(run_name, latest_link)
    
    # Modify dataset.yaml to enable more augmentations
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Add advanced augmentation settings
        data_config['degrees'] = 10.0      # Rotation augmentation
        data_config['translate'] = 0.2     # Translation augmentation
        data_config['scale'] = 0.2         # Scale augmentation
        data_config['shear'] = 5.0         # Shear augmentation
        data_config['perspective'] = 0.001 # Perspective augmentation
        data_config['flipud'] = 0.5        # Vertical flip probability
        data_config['fliplr'] = 0.5        # Horizontal flip probability
        data_config['mosaic'] = 1.0        # Mosaic augmentation probability
        data_config['mixup'] = 0.3         # Mixup augmentation probability
        data_config['copy_paste'] = 0.3    # Copy-paste augmentation probability
        
        # Save augmented config
        augmented_yaml_path = os.path.join(run_dir, "dataset_augmented.yaml")
        with open(augmented_yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        
        # Use the augmented config
        data_yaml_path = augmented_yaml_path
        print(f"Created augmented dataset config at: {augmented_yaml_path}")
    except Exception as e:
        print(f"Warning: Could not modify dataset.yaml for augmentations: {e}")
    
    # Set training parameters optimized for accuracy
    params = {
        "data": data_yaml_path,      # Dataset configuration
        "epochs": 500,               # Extended number of epochs for convergence
        "imgsz": 1280,               # Larger image size for better accuracy
        "batch": 8,                  # Reduced batch size to accommodate larger model
        "device": device,            # Device to use
        "workers": 8,                # More worker threads
        "patience": 50,              # Increased patience for early stopping
        "project": "runs",           # Project directory
        "name": run_name,            # Run name
        "exist_ok": True,            # Overwrite existing run
        "pretrained": True,          # Use pretrained weights
        "optimizer": "AdamW",        # Better optimizer for convergence
        "lr0": 0.001,                # Lower initial learning rate
        "lrf": 0.0001,               # Lower final learning rate
        "momentum": 0.937,           # Default momentum
        "weight_decay": 0.005,       # Increased weight decay to reduce overfitting
        "warmup_epochs": 5.0,        # Extended warmup
        "warmup_momentum": 0.8,      # Default warmup momentum
        "warmup_bias_lr": 0.1,       # Default warmup bias lr
        "box": 7.5,                  # Box loss gain
        "cls": 0.5,                  # Cls loss gain
        "dfl": 1.5,                  # DFL loss gain
        "dropout": 0.1,              # Add dropout for regularization
        "label_smoothing": 0.1,      # Label smoothing for better generalization
        "cos_lr": True,              # Use cosine learning rate scheduler
        "close_mosaic": 15,          # Disable mosaic in final epochs
        "freeze": [0, 1, 2],         # Freeze early layers initially
        "amp": True,                 # Use mixed precision training
        "save": True,                # Save checkpoints
        "save_period": 10,           # Save checkpoint every 10 epochs
        "plots": True,               # Generate plots
        "rect": True,                # Use rectangular training
        "overlap_mask": True,        # Compute mask overlap metrics
        "nbs": 64,                   # Nominal batch size for scaling other parameters
        "cache": True,               # Cache images for faster training
        # EMA settings
        "fraction": 0.9,             # EMA fraction
        "profile": True,             # Profile the training process
        "multi_scale": True,         # Use multi-scale training
        # Validation settings
        "val": True,                 # Run validation
        "conf": 0.001,               # Low confidence threshold for validation
        "iou": 0.7,                  # Higher IoU threshold for NMS in validation
    }
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Model: YOLOv8{model_size}")
    print(f"Dataset: {params['data']}")
    print(f"Epochs: {params['epochs']}")
    print(f"Image Size: {params['imgsz']}")
    print(f"Batch Size: {params['batch']}")
    print(f"Device: {params['device']}")
    print(f"Optimizer: {params['optimizer']}")
    
    # Start training
    print("\nStarting training...")
    results = model.train(**params)
    
    # Print training results
    print("\nTraining completed!")
    print(f"Results saved to {os.path.join(params['project'], params['name'])}")
    
    # Test with different IoU and confidence thresholds to find the optimal settings
    print("\nValidating model with different thresholds...")
    best_map = 0
    best_conf = 0.25
    best_iou = 0.7
    
    # Grid search for optimal confidence and IoU thresholds
    for conf in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        for iou in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
            val_results = model.val(conf=conf, iou=iou)
            current_map = val_results.box.map    # Get mAP value
            print(f"Conf: {conf}, IoU: {iou}, mAP: {current_map:.4f}")
            
            if current_map > best_map:
                best_map = current_map
                best_conf = conf
                best_iou = iou
    
    print(f"\nBest thresholds - Conf: {best_conf}, IoU: {best_iou}, mAP: {best_map:.4f}")
    
    # Fine-tune with the best thresholds
    print("\nFine-tuning model with best thresholds...")
    fine_tune_params = params.copy()
    fine_tune_params.update({
        "epochs": 50,
        "lr0": 0.0001,
        "conf": best_conf,
        "iou": best_iou,
        "freeze": [],  # Unfreeze all layers
        "name": f"{run_name}_finetune",
    })
    
    fine_tune_results = model.train(**fine_tune_params)
    
    # Export the model to different formats with optimal settings
    print("\nExporting model...")
    export_params = {
        "format": "onnx",
        "imgsz": params["imgsz"],
        "conf": best_conf,
        "iou": best_iou,
        "optimize": True,
        "int8": False,  # Keep full precision for accuracy
    }
    model.export(**export_params)
    
    # Also export to other formats
    for format_type in ["torchscript", "openvino"]:
        try:
            export_params["format"] = format_type
            model.export(**export_params)
            print(f"Exported to {format_type}")
        except Exception as e:
            print(f"Failed to export to {format_type}: {e}")
    
    # Save the optimal configuration for inference
    optimal_config = {
        "model_path": os.path.join(params['project'], f"{run_name}_finetune", "weights", "best.pt"),
        "imgsz": params["imgsz"],
        "conf_threshold": best_conf,
        "iou_threshold": best_iou,
    }
    
    with open(os.path.join(params['project'], "optimal_config.yaml"), 'w') as f:
        yaml.dump(optimal_config, f)
    
    print("\nTraining, validation, and export complete!")
    print(f"Optimal configuration saved to {os.path.join(params['project'], 'optimal_config.yaml')}")

if __name__ == "__main__":
    main()