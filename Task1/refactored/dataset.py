import random
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize


class VolumeSliceDataset(Dataset):
    """
    Dataset for loading 3D medical volume data and corresponding segmentation masks.
    Samples multiple 2D slices from each volume with a preference for slices containing ROI.
    
    Args:
        images_dir (str): Directory containing image volumes
        masks_dir (str): Directory containing segmentation masks
        patient_ids (list): List of patient IDs to include in this dataset
        target_shape (tuple): Target shape for resizing slices (height, width)
        slices_per_volume (int): Number of slices to sample from each volume
        roi_ratio (float): Ratio of slices to include that contain ROI vs. empty slices
    """
    def __init__(self, images_dir, masks_dir, patient_ids, 
                 target_shape=(128, 128), slices_per_volume=16, roi_ratio=0.7):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.patient_ids = patient_ids
        self.target_shape = target_shape
        self.slices_per_volume = slices_per_volume
        self.roi_ratio = roi_ratio
        
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        """
        Load a volume, sample slices, and return a batch of image-mask pairs.
        
        Args:
            idx (int): Index of the patient ID to load
            
        Returns:
            tuple: (images, masks) tensors with shape [slices_per_volume, 1, H, W]
        """
        patient_id = self.patient_ids[idx]
        
        # Construct file paths
        img_path = f"{self.images_dir}/{patient_id}_image.nii"
        mask_path = f"{self.masks_dir}/{patient_id}_label.nii"
        
        # Load the image and mask volumes
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        # Get dimensions
        num_slices = img_data.shape[2]

        # Separate slices with and without ROI
        slices_with_roi = []
        slices_without_roi = []

        for i in range(num_slices):
            has_roi = np.any(mask_data[:, :, i] > 0)
            if has_roi:
                slices_with_roi.append(i)
            else:
                slices_without_roi.append(i)

        # Calculate how many slices to sample based on ROI ratio
        num_total = min(self.slices_per_volume, num_slices)
        num_with_roi = min(int(self.roi_ratio * num_total), len(slices_with_roi))
        num_without_roi = num_total - num_with_roi

        # Ensure we have enough slices
        if len(slices_with_roi) < num_with_roi:
            num_with_roi = len(slices_with_roi)
            num_without_roi = min(num_total - num_with_roi, len(slices_without_roi))
            
        if len(slices_without_roi) < num_without_roi:
            num_without_roi = len(slices_without_roi)
            num_with_roi = min(num_total - num_without_roi, len(slices_with_roi))

        # Randomly select slices
        random.shuffle(slices_with_roi)
        random.shuffle(slices_without_roi)

        selected_slices = slices_with_roi[:num_with_roi] 
        random.shuffle(selected_slices)

        # Process selected slices
        images = []
        masks = []

        for slice_idx in selected_slices:
            # Resize images and masks to target shape
            img_slice = resize(img_data[:, :, slice_idx], self.target_shape, 
                               preserve_range=True).astype(np.float32)
            mask_slice = resize(mask_data[:, :, slice_idx], self.target_shape, 
                                preserve_range=True).astype(np.float32)

            # Normalize image intensity
            if np.max(img_slice) > 0:
                img_slice = img_slice / np.max(img_slice)

            # Convert to tensor and add channel dimension
            img_tensor = torch.tensor(img_slice).unsqueeze(0)  # [1, H, W]
            mask_tensor = torch.tensor((mask_slice > 0).astype(np.float32)).unsqueeze(0)

            images.append(img_tensor)
            masks.append(mask_tensor)

        # Stack the slices to form a batch
        images_tensor = torch.stack(images)  # Shape: [slices_per_volume, 1, H, W]
        masks_tensor = torch.stack(masks)    # Shape: [slices_per_volume, 1, H, W]

        return images_tensor, masks_tensor
