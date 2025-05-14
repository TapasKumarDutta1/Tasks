import argparse
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


def run_inference_on_volume(model, volume, device):
    """
    Run inference on all slices of a 3D volume.

    Args:
        model (torch.nn.Module): Trained segmentation model.
        volume (np.ndarray): 3D numpy array of shape (D, H, W).
        device (str): Computation device.

    Returns:
        np.ndarray: Predicted segmentation mask volume of shape (D, H, W).
    """
    model.eval()
    model.to(device)

    pred_masks = []

    with torch.no_grad():
        for slice_ in volume:
            slice_ = slice_/np.max(slice_)
            input_tensor = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().to(device)  # shape: (1, 1, H, W)
            output = model(input_tensor)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0] > 0.5
            pred_masks.append(pred.astype(np.uint8))

    return np.stack(pred_masks)


def get_args():
    parser = argparse.ArgumentParser(description='Run segmentation on a single 3D image volume')

    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing image volumes')
    parser.add_argument('--masks-dir', type=str, required=True, help='Directory containing image volumes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--encoder', type=str, default='resnet50', help='Encoder backbone used in the U-Net model')
    parser.add_argument('--img-size', type=int, default=128, help='Target size for image resizing')
    parser.add_argument('--patient-id', type=str, required=True, help='Patient ID to run inference on')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output segmentation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


def main():
    args = get_args()

    image_path = os.path.join(args.images_dir, f"{args.patient_id}_image.nii")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Volume for patient ID {args.patient_id} not found at {image_path}")

    # Load and preprocess volume
    volume_nifti = nib.load(image_path)
    volume = volume_nifti.get_fdata()
    volume_preprocessed = volume

    # Load model
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device)

    # Run inference
    pred_volume = run_inference_on_volume(model, volume_preprocessed, args.device)


    mask_path = os.path.join(args.masks_dir, f"{args.patient_id}_label.nii")
    mask_nifti = nib.load(mask_path)
    mask_volume = mask_nifti.get_fdata().astype(np.uint8)
    # mask_volume_resized = preprocess_volume(mask_volume, target_shape=(args.img_size, args.img_size))  # shape: (D, H, W)

    # Ensure shapes match
    mask_volume_resized = mask_volume
    print(pred_volume.shape, mask_volume_resized.shape)
        
    # Compute Dice and IoU
    intersection = np.logical_and(pred_volume, mask_volume_resized).sum()
    union = np.logical_or(pred_volume, mask_volume_resized).sum()
    pred_sum = pred_volume.sum()
    gt_sum = mask_volume_resized.sum()

    dice = (2. * intersection) / (pred_sum + gt_sum + 1e-8)
    iou = intersection / (union + 1e-8)

    print(f"Patient {args.patient_id} - Dice: {dice:.4f}, IoU: {iou:.4f}")


    
    # Save prediction as a Nifti file
    pred_nifti = nib.Nifti1Image(pred_volume.astype(np.uint8), affine=volume_nifti.affine)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.patient_id}_seg.nii")
    nib.save(pred_nifti, output_path)

    print(f"Saved predicted segmentation volume to {output_path}")


if __name__ == "__main__":
    main()
