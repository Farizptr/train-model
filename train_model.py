# download_and_train.py  
  
# Langkah 1: Import necessary libraries  
from roboflow import Roboflow  
from ultralytics import YOLO  
import os  
  
# Mengatur variabel lingkungan untuk menghindari fragmentasi  
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  
  
# Langkah 2: Download the dataset using the provided Roboflow code  
def download_dataset(api_key, workspace, project_name, version_number, download_type):  
    try:  
        rf = Roboflow(api_key=api_key)  
        project = rf.workspace(workspace).project(project_name)  
        version = project.version(version_number)  
        dataset = version.download(download_type)  
        print(f"Dataset downloaded successfully to {dataset.location}")  
        return dataset.location  
    except Exception as e:  
        print(f"Error downloading dataset: {e}")  
        return None  
  
# Konfigurasi API Roboflow  
api_key = "S51A8DD08qFOOzjKkaJH"  
workspace = "building-detection-5sjgk"  
project_name = "satellite-osm"  
version_number = 4  
download_type = "yolov8"  
  
# Mengunduh dataset  
dataset_location = download_dataset(api_key, workspace, project_name, version_number, download_type)  
  
if dataset_location is None:  
    print("Failed to download dataset. Exiting script.")  
    exit(1)  
  
# Langkah 3: Set up the YOLOv8 model configuration  
# Inisialisasi model YOLOv8s (lebih kecil dari YOLOv8x)  
model = YOLO('yolov8s.yaml')  # Anda juga bisa menggunakan 'yolov8s.pt' jika Anda memiliki model pre-trained  
  
# Langkah 4: Train the model using the downloaded dataset  
def train_model(model, dataset_location, epochs=50, batch=8, imgsz=640, device='0', workers=4, optimizer='Adam', lr0=0.001, save_period=-1, save_dir='runs/train', accumulate=4, half=True):  
    try:  
        results = model.train(  
            data=f"{dataset_location}/data.yaml",  # Path ke file data.yaml  
            epochs=epochs,  # Jumlah epoch  
            batch=batch,  # Ukuran batch  
            imgsz=imgsz,  # Ukuran gambar input  
            device=device,  # Gunakan GPU dengan ID 0, gunakan 'cpu' jika tidak menggunakan GPU  
            workers=workers,  # Jumlah worker untuk memuat data  
            optimizer=optimizer,  # Optimizer yang digunakan  
            lr0=lr0,  # Learning rate awal  
            save_period=save_period,  # Simpan model setiap epoch  
            save_dir=save_dir,  # Direktori untuk menyimpan hasil pelatihan  
            accumulate=accumulate,  # Gradient accumulation  
            half=half  # Mixed precision training  
        )  
        print("Model training completed successfully.")  
        return results  
    except Exception as e:  
        print(f"Error during model training: {e}")  
        return None  
  
# Melatih model  
training_results = train_model(model, dataset_location)  
  
if training_results is None:  
    print("Model training failed. Exiting script.")  
    exit(1)  
  
# Langkah 5: Menyimpan Model yang Telah Dilatih  
# Model yang telah dilatih akan secara otomatis disimpan di direktori yang ditentukan oleh argumen save_dir.  
# Anda juga dapat menyimpan model secara manual menggunakan metode save.  
  
# Menyimpan model ke file  
model.save('best_model.pt')  
print("Model saved as 'best_model.pt'")  
