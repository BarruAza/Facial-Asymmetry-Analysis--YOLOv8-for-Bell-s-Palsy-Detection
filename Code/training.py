from ultralytics import YOLO
import os

def main():
    yaml_path = '/Users/Barru/Documents/Universitas Brawijaya/SEMESTER 5/Computer Vision/Project_Akhir_Compvis/Dataset/project_data.yaml'
    
    project_name = 'bell_palsy_project'
    run_name = 'train_result_v1'

    if not os.path.exists(yaml_path):
        print(f"ERROR: File konfigurasi tidak ditemukan di: {yaml_path}")
        print("Pastikan Anda sudah membuat file project_data.yaml di folder yang benar.")
        return

    print("--- Memulai Proses Training YOLOv8 ---")
    
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project=project_name,
        name=run_name,
        patience=10,
        device=0,
        verbose=True
    )

    print(f"\n✅ Training Selesai!")
    print(f"Model terbaik tersimpan di: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()