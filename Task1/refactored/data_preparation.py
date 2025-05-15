import os
import glob
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from utils import extract_patient_id_from_filename


def prepare_dataloaders(
    images_dir,
    masks_dir,
    batch_size=1,
    target_shape=(128, 128),
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    num_workers=4
):
    """
    Prepare dataloaders for training, validation, and testing.
    
    Args:
        images_dir (str): Directory containing the images
        masks_dir (str): Directory containing the masks
        batch_size (int): Batch size for dataloaders
        target_shape (tuple): Target shape for resizing images
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        num_workers (int): Number of workers for dataloader
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from dataset import VolumeSliceDataset

    # Get all image paths and extract patient IDs
    image_paths = glob.glob(os.path.join(images_dir, "*_image.nii"))
    patient_ids = []
    
    for path in image_paths:
        patient_id = extract_patient_id_from_filename(path)
        if patient_id:
            patient_ids.append(patient_id)
    
    # Convert string IDs to integers for proper sorting
    patient_ids = [int(i) for i in patient_ids]
    patient_ids.sort()
    
    # Split into train+val and test sets
    train_val_ids, test_ids = train_test_split(
        patient_ids, test_size=test_size, random_state=random_state
    )
    
    # Further split train+val into train and validation sets
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Convert back to strings
    train_ids = [int(i) for i in train_ids]
    val_ids = [int(i) for i in val_ids]
    test_ids = [int(i) for i in test_ids]
    np.save('/kaggle/working/train.npy',train_ids)
    np.save('/kaggle/working/val.npy',val_ids)
    np.save('/kaggle/working/test.npy',test_ids)
    
    print(f"Number of patients in train set: {len(train_ids)}")
    print(f"Number of patients in validation set: {len(val_ids)}")
    print(f"Number of patients in test set: {len(test_ids)}")
    
    # Create datasets
    train_dataset = VolumeSliceDataset(
        images_dir, masks_dir, train_ids, target_shape=target_shape
    )
    
    val_dataset = VolumeSliceDataset(
        images_dir, masks_dir, val_ids, target_shape=target_shape
    )
    
    test_dataset = VolumeSliceDataset(
        images_dir, masks_dir, test_ids, target_shape=target_shape
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
