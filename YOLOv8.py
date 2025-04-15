#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:54:00 2024

@author: harisushehu
"""


import pickle
import torch
import tensorflow as tf
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
import os

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

def load_dataloaders():
    # Load dataset configuration from file
    with open('dataloaders.pkl', 'rb') as f:
        config = pickle.load(f)


    # Use configuration to set up datasets and dataloaders
    train_dataset = CustomDataset(img_dir=config['train_img_dir'])
    val_dataset = CustomDataset(img_dir=config['val_img_dir'])
    test_dataset = CustomDataset(img_dir=config['test_img_dir'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def train_model():
    
    #set device to M1 GPU if available
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print("Is CUDA available: ", torch.cuda.is_available())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Create dataset and dataloader using loaded configuration
    train_dataloader, val_dataloader, test_dataloader = load_dataloaders()

    # Load a pre-trained YOLO model
    model = YOLO('yolov8l.pt') #yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

    # Define training settings
    training_settings = {
        'data': './TomatoEbola/data.yaml',  # Path to the dataset configuration
        'epochs': 100, 
        'imgsz': 640,
        'batch': 32, #16
        'augment': True
    }

    # Train the model with the built-in augmentations
    results = model.train(**training_settings)

    # Save the trained model
    #model.save('Ttomato_yolov8.pt')

if __name__ == "__main__":
    train_model()
