"""
Modified for YOLOv5
"""
import os
import torch
import tensorflow as tf
from yolov5 import train

def train_model():

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Is CUDA available: ", torch.cuda.is_available())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print(os.getcwd())

    # Define training settings
    training_settings = {
        'data': '../../../Desktop/NAIRS/YOLO/NAIRS2.v1i.yolov8/data.yaml',  # Dataset configuration
        'weights': 'yolov5l.pt',  # Model weights (yolov5s, yolov5m, yolov5l, yolov5x)
        'epochs': 30, #100
        'imgsz': 640,
        'batch_size': 32
    }

    # Train YOLOv5
    train.run(**training_settings)

if __name__ == "__main__":
    train_model()
