import numpy as np
import re
import torch
from sklearn.metrics import jaccard_score


def extract_patient_id_from_filename(path):
    """
    Extract the patient ID number from a file path.
    
    Args:
        path (str): Path to the image file containing an ID (e.g., '123_image.nii')
    
    Returns:
        str or None: The extracted ID or None if no match found
    """
    match = re.search(r'(\d+)_image\.nii', path)
    if match:
        return match.group(1)
    else:
        return None


def calculate_dice_coefficient(prediction, target, smooth=1e-6):
    """
    Compute the Dice coefficient between two binary masks.
    
    Args:
        prediction (np.ndarray): Predicted binary mask. Shape (H, W) or (N, H, W).
        target (np.ndarray): Ground truth binary mask. Same shape as `prediction`.
        smooth (float): Smoothing constant to avoid division by zero.
    
    Returns:
        float: Dice coefficient score between 0 and 1
    """
    prediction = prediction.astype(np.bool_)
    target = target.astype(np.bool_)
    
    intersection = np.logical_and(prediction, target).sum()
    union = prediction.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def evaluate_model(model, dataloader, device):
    """
    Evaluate model performance using IoU and Dice scores.
    
    Args:
        model (nn.Module): PyTorch model to evaluate
        dataloader (DataLoader): DataLoader containing validation/test data
        device (str): Device to run evaluation on ('cuda' or 'cpu')
    
    Returns:
        tuple: Average IoU score and Dice coefficient across the dataset
    """
    model.eval()
    model.to(device)
    total_iou_score = 0
    total_dice_score = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device).view(-1, 1, 128, 128)
            masks = masks.view(-1, 1, 128, 128)
            
            outputs = model(images)
            predictions = torch.sigmoid(outputs).detach().cpu().numpy()
            ground_truth = masks.round().int().numpy()
            
            # Convert to binary masks and calculate metrics
            predicted_masks = predictions.round()
            ground_truth_flat = ground_truth.flatten()
            predicted_masks_flat = predicted_masks.flatten()
            
            # Calculate IoU (Jaccard score)
            iou = jaccard_score(ground_truth_flat, predicted_masks_flat)
            
            # Calculate Dice coefficient
            dice = calculate_dice_coefficient(predicted_masks, ground_truth)
            
            total_iou_score += iou
            total_dice_score += dice
    
    avg_iou = total_iou_score / len(dataloader)
    avg_dice = total_dice_score / len(dataloader)
    
    return avg_iou, avg_dice
