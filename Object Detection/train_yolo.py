from ultralytics import YOLO
import os

# Define paths
dataset_yaml = r"D:\Study materials/Year 2/SEGP/Code/Object Detection/yolo_dataset/data.yaml"
project_dir = r"D:/Study materials/Year 2/SEGP/Code/Object Detection/runs"

def train_model():
    print("Initializing YOLOv8 Nano model...")
    # Load a pretrained YOLO model (using Nano version as it's fastest to train)
    model = YOLO("yolov8n.pt") 

    print("\nStarting training process. This may take a while depending on your hardware.")
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=100,            # Increased to 100 (YOLO will early-stop if it finishes sooner)
        imgsz=640,             # Image size to scale to
        batch=12,              # Number of images per batch
        project=project_dir,   # Where to save training results
        name="fruit_detector", # Name of this training run
        plots=True,            # Generate training plots and metrics
        resume=False           # Start a new training run from the pre-trained weights
    )
    
    print("\nTraining complete!")
    print(f"Check the {project_dir}\\fruit_detector folder for results and your best model weights (best.pt)!")

if __name__ == '__main__':
    train_model()
