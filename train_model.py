# download_and_train.py

# Langkah 1: Import necessary libraries
from roboflow import Roboflow
from ultralytics import YOLO
import os
import torch
from datetime import datetime

# Mengatur variabel lingkungan untuk optimasi memori
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Memeriksa ketersediaan GPU
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {'CUDA GPU' if device=='0' else 'CPU'}")

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
# Menggunakan model yang lebih besar untuk akurasi yang lebih tinggi
model = YOLO('yolov8x.pt')  # Menggunakan model pre-trained YOLOv8x yang lebih besar

# Membuat direktori output unik
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"runs/train/model_training_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Langkah 4: Train the model using the downloaded dataset
def train_model(model, dataset_location, epochs=100, batch=4, imgsz=960, device='0', workers=4, optimizer='AdamW', lr0=0.0001, lrf=0.00001, save_period=10, save_dir=output_dir, patience=20, cos_lr=True):
    try:
        # Mempersiapkan parameter pelatihan yang dioptimalkan untuk akurasi
        results = model.train(
            data=f"{dataset_location}/data.yaml",  # Path ke file data.yaml
            epochs=epochs,                          # Jumlah epoch yang lebih banyak
            batch=batch,                            # Ukuran batch yang lebih kecil untuk mengurangi overfitting
            imgsz=imgsz,                            # Ukuran gambar input yang lebih besar
            device=device,                          # Gunakan GPU jika tersedia
            workers=workers,                        # Jumlah worker untuk memuat data
            optimizer=optimizer,                    # AdamW optimizer untuk kinerja yang lebih baik
            lr0=lr0,                                # Learning rate awal yang lebih kecil
            lrf=lrf,                                # Learning rate akhir yang lebih kecil
            cos_lr=cos_lr,                          # Menggunakan cosine learning rate scheduler
            patience=patience,                      # Early stopping patience
            save_period=save_period,                # Simpan model setiap 10 epoch
            save_dir=save_dir,                      # Direktori untuk menyimpan hasil
            augment=True,                           # Mengaktifkan augmentasi data
            mixup=0.15,                             # Mengaktifkan mixup untuk augmentasi
            mosaic=1.0,                             # Mengaktifkan mosaic untuk augmentasi
            degrees=0.5,                            # Rotasi maksimum selama augmentasi
            translate=0.1,                          # Translasi maksimum selama augmentasi
            scale=0.5,                              # Penskalaan maksimum selama augmentasi
            fliplr=0.5,                             # Probabilitas flip horizontal
            flipud=0.2,                             # Probabilitas flip vertikal
            hsv_h=0.015,                            # HSV hue augmentation
            hsv_s=0.7,                              # HSV saturation augmentation
            hsv_v=0.4,                              # HSV value augmentation
            warmup_epochs=3,                        # Epoch pemanasan untuk stabilitas
            weight_decay=0.0005,                    # Regularisasi L2
            overlap_mask=True,                      # Memperbaiki overlap mask untuk deteksi
            mask_ratio=4,                           # Mask ratio untuk deteksi
            dropout=0.15,                           # Dropout untuk mencegah overfitting
            val=True,                               # Melakukan validasi selama pelatihan
            amp=True,                               # Mixed precision training untuk efisiensi
            cache=True,                             # Cache data untuk akses yang lebih cepat
        )
        print("Model training completed successfully.")
        return results
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

# Melatih model
print("Starting model training with optimized parameters for high accuracy...")
training_results = train_model(model, dataset_location)

if training_results is None:
    print("Model training failed. Exiting script.")
    exit(1)

# Langkah 5: Menyimpan Model yang Telah Dilatih
# Menyimpan model ke file dengan nama timestamp
model_name = f"best_model_{timestamp}.pt"
model.save(f"{output_dir}/{model_name}")
print(f"Model saved as '{output_dir}/{model_name}'")

# Membuat salinan dengan nama tetap untuk akses mudah
model.save('best_model.pt')
print("Model also saved as 'best_model.pt'")

# Mencetak ringkasan performa model
print("\nTraining Performance Summary:")
print(f"Final mAP50: {training_results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"Final mAP50-95: {training_results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"Training completed in {training_results.results_dict.get('elapsed/epochs', 'N/A')} epochs")
print(f"Total training time: {training_results.results_dict.get('elapsed/time', 'N/A')} seconds")
print(f"Full training results saved to {output_dir}")