"""
Data module for XAI4LtR framework.

This module contains dataset classes and data preparation functions
for loading and preprocessing medical images.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class CTScanDataset(Dataset):
    """
    Custom Dataset class for loading CT scan images and their labels.
    Handles image paths and applies transformations.
    Returns images, labels, and original global index.
    """
    
    def __init__(self, image_paths, labels, transform=None, global_indices=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Ensure global_indices is a numpy array
        self.global_indices = np.array(global_indices) if global_indices is not None else np.arange(len(image_paths))
        
        # Create mapping from global_idx to local idx in this split for convenient lookup
        self.original_indices_map = {self.global_indices[i]: i for i in range(len(self.global_indices))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels for pre-trained models
        label = self.labels[idx]
        global_idx = self.global_indices[idx]  # Return original global index

        if self.transform:
            image = self.transform(image)

        return image, label, global_idx  # Return original global index for training dynamics tracking


def prepare_datasets(cfg, data_dir=None, covid_dir=None, non_covid_dir=None):
    """
    Collect image paths and labels, then split into train/validation/test sets.
    Assign a unique global index to each image from the original dataset.
    
    Args:
        cfg: Configuration object
        data_dir: Optional override for data directory
        covid_dir: Optional override for COVID images directory
        non_covid_dir: Optional override for non-COVID images directory
        
    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    
    # Use provided directories or fall back to config
    if data_dir:
        covid_dir = covid_dir or os.path.join(data_dir, 'COVID-19/images')
        non_covid_dir = non_covid_dir or os.path.join(data_dir, 'Normal/images')
    else:
        covid_dir = covid_dir or cfg.COVID_DIR
        non_covid_dir = non_covid_dir or cfg.NON_COVID_DIR
    
    all_image_paths_raw = []
    all_labels_raw = []

    # Collect COVID images
    covid_paths = [os.path.join(covid_dir, f) for f in os.listdir(covid_dir) 
                   if f.endswith('.png') or f.endswith('.jpg')]
    all_image_paths_raw.extend(covid_paths)
    all_labels_raw.extend([1] * len(covid_paths))  # 1 for COVID

    # Collect non-COVID images
    non_covid_paths = [os.path.join(non_covid_dir, f) for f in os.listdir(non_covid_dir) 
                       if f.endswith('.png') or f.endswith('.jpg')]
    all_image_paths_raw.extend(non_covid_paths)
    all_labels_raw.extend([0] * len(non_covid_paths))  # 0 for non-COVID

    # Assign global indices
    all_global_indices = list(range(len(all_image_paths_raw)))

    print(f"Total images found: {len(all_image_paths_raw)}")
    print(f"COVID images: {len(covid_paths)}, Non-COVID images: {len(non_covid_paths)}")

    # Create stratified splits for train+validation and test sets first
    # Desired: Test = 15% of total
    train_val_paths, test_paths, train_val_labels, test_labels, \
    train_val_global_indices, test_global_indices = train_test_split(
        all_image_paths_raw, all_labels_raw, all_global_indices,
        test_size=0.15, random_state=cfg.RANDOM_SEED, stratify=all_labels_raw
    )
    
    # Then split train_val into actual train and validation sets
    # train_val is 85% of total. We want Val = 15% of total.
    # So val_size_relative_to_train_val = 0.15 / (1.0 - 0.15)
    val_size_relative_to_train_val = 0.15 / (1.0 - 0.15)
    train_paths, val_paths, train_labels, val_labels, \
    train_global_indices, val_global_indices = train_test_split(
        train_val_paths, train_val_labels, train_val_global_indices,
        test_size=val_size_relative_to_train_val, random_state=cfg.RANDOM_SEED, stratify=train_val_labels
    )

    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")
    print(f"Test set size: {len(test_paths)}")

    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Pass global_indices to Dataset constructor
    train_dataset = CTScanDataset(train_paths, train_labels, train_transform, train_global_indices)
    val_dataset = CTScanDataset(val_paths, val_labels, val_test_transform, val_global_indices)
    test_dataset = CTScanDataset(test_paths, test_labels, val_test_transform, test_global_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def create_generic_dataset(image_paths, labels, transform=None, global_indices=None):
    """
    Create a generic dataset from image paths and labels.
    Useful for creating datasets from different data sources.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        transform: Optional transform to apply
        global_indices: Optional global indices for tracking
        
    Returns:
        CTScanDataset instance
    """
    return CTScanDataset(image_paths, labels, transform, global_indices)


def create_data_loader(dataset, batch_size, shuffle=True, num_workers=2):
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size for the loader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) 