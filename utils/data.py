"""
Data preprocessing utilities for multi-label image classification with missing labels.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class MultiLabelImageDataset(Dataset):
    """
    Custom Dataset for multi-label classification with missing values.
    """
    
    def __init__(self, image_paths, labels, transform=None, image_dir='dataset/images'):
        """
        Args:
            image_paths: List of image filenames
            labels: numpy array of shape (n_samples, n_labels) with possible NaN values
            transform: Optional transform to apply to images
            image_dir: Directory containing images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_dir = image_dir
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels and create a mask for missing values
        labels = self.labels[idx]
        
        # Create mask: 1 where label is available, 0 where it's missing (NaN)
        label_mask = ~np.isnan(labels)
        
        # Replace NaN with 0 (will be ignored during loss calculation)
        labels_clean = np.nan_to_num(labels, nan=0.0)
        
        return {
            'image': image,
            'labels': torch.FloatTensor(labels_clean),
            'mask': torch.FloatTensor(label_mask.astype(float))
        }


def load_dataset(labels_path='dataset/labels.txt', train_size=0.8, val_size=0.1, random_state=42):
    """
    Load and split the dataset into train, validation, and test sets.
    
    Args:
        labels_path: Path to the labels file
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing splits with image paths and labels
    """
    # Read labels file
    df = pd.read_csv(labels_path, sep=' ', header=None)
    df.columns = ['filename', 'label_1', 'label_2', 'label_3', 'label_4']
    
    # Replace 'NA' with NaN
    label_cols = ['label_1', 'label_2', 'label_3', 'label_4']
    for col in label_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract filenames and labels
    filenames = df['filename'].values
    labels = df[label_cols].values.astype(float)
    
    # Print dataset statistics
    print(f"Total samples: {len(filenames)}")
    print(f"Number of labels: {len(label_cols)}")
    print(f"\nMissing values per label:")
    for i, col in enumerate(label_cols):
        missing = np.isnan(labels[:, i]).sum()
        print(f"  {col}: {missing} ({missing/len(labels)*100:.2f}%)")
    
    # Split into train, val, test
    test_size = 1.0 - train_size - val_size
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        filenames, labels, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return {
        'train': {'filenames': X_train, 'labels': y_train},
        'val': {'filenames': X_val, 'labels': y_val},
        'test': {'filenames': X_test, 'labels': y_test}
    }


def get_transforms(augment=True):
    """
    Get image transformations for training and validation.
    
    Args:
        augment: Whether to apply data augmentation (for training)
    
    Returns:
        torchvision transforms
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(dataset_splits, batch_size=32, num_workers=4, image_dir='dataset/images'):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        dataset_splits: Dictionary from load_dataset()
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_dir: Directory containing images
    
    Returns:
        Dictionary containing DataLoaders
    """
    dataloaders = {}
    
    # Training DataLoader
    train_dataset = MultiLabelImageDataset(
        dataset_splits['train']['filenames'],
        dataset_splits['train']['labels'],
        transform=get_transforms(augment=True),
        image_dir=image_dir
    )
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation DataLoader
    val_dataset = MultiLabelImageDataset(
        dataset_splits['val']['filenames'],
        dataset_splits['val']['labels'],
        transform=get_transforms(augment=False),
        image_dir=image_dir
    )
    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test DataLoader
    test_dataset = MultiLabelImageDataset(
        dataset_splits['test']['filenames'],
        dataset_splits['test']['labels'],
        transform=get_transforms(augment=False),
        image_dir=image_dir
    )
    dataloaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloaders


if __name__ == '__main__':
    # Test data loading
    dataset_splits = load_dataset()
    dataloaders = create_dataloaders(dataset_splits, batch_size=16, num_workers=0)
    
    # Test one batch
    batch = next(iter(dataloaders['train']))
    print(f"\nBatch shapes:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Labels: {batch['labels'].shape}")
    print(f"  Masks: {batch['mask'].shape}")
