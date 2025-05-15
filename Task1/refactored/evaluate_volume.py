
import argparse
import os
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from sklearn.metrics import jaccard_score
import segmentation_models_pytorch as smp

def calculate_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return (2. * intersection) / (pred.sum() + gt.sum() + 1e-8)

def evaluate_full_volumes(model, images_dir, masks_dir, patient_ids, target_shape=(128, 128), device='cuda', save_dir="3d_high_dice_cases"):
    model.eval()
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    total_dice = 0
    total_iou = 0
    count = 0

    for patient_id in patient_ids:
        img_path = os.path.join(images_dir, f"{patient_id}_image.nii")
        mask_path = os.path.join(masks_dir, f"{patient_id}_label.nii")

        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        img_vol = img_nii.get_fdata()
        mask_vol = mask_nii.get_fdata()

        original_shape = img_vol.shape
        predicted_vol = np.zeros_like(mask_vol)

        for i in tqdm(range(original_shape[2]), desc=f"Processing {patient_id}"):
            img_slice = resize(img_vol[:, :, i], target_shape, preserve_range=True).astype(np.float32)
            img_slice = img_slice / (img_slice.max() + 1e-8)
            input_tensor = torch.tensor(img_slice).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()
                pred_binary = (pred > 0.5).astype(np.uint8)

            pred_resized = resize(pred_binary, (original_shape[0], original_shape[1]), order=0, preserve_range=True).astype(np.uint8)
            predicted_vol[:, :, i] = pred_resized

        mask_binary = (mask_vol > 0).astype(np.uint8)
        dice = calculate_dice(predicted_vol, mask_binary)
        iou = jaccard_score(mask_binary.flatten(), predicted_vol.flatten())

        total_dice += dice
        total_iou += iou
        count += 1

        print(f"{patient_id}: Dice = {dice:.4f}")
        print("*" * 20)

        nib.save(nib.Nifti1Image(img_vol, img_nii.affine), os.path.join(save_dir, f"{patient_id}_image.nii.gz"))
        nib.save(nib.Nifti1Image(mask_binary, mask_nii.affine), os.path.join(save_dir, f"{patient_id}_label.nii.gz"))
        nib.save(nib.Nifti1Image(predicted_vol, img_nii.affine), os.path.join(save_dir, f"{patient_id}_prediction.nii.gz"))

        print(f"Saved predictions for patient {patient_id} (Dice: {dice:.4f})")

    avg_dice = total_dice / count
    avg_iou = total_iou / count
    print(f"\nAverage Dice: {avg_dice:.4f}, Average IoU: {avg_iou:.4f}")
    return avg_iou, avg_dice

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D volumes using a 2D segmentation model.")
    parser.add_argument('--images_dir', type=str, required=True, help='Path to the directory with image .nii files')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to the directory with mask .nii files')
    parser.add_argument('--patient_ids_file', type=str, required=True, help='Path to .npy file containing list of patient IDs')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--save_dir', type=str, default='3d_high_dice_cases', help='Directory to save predictions')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu)')
    args = parser.parse_args()

    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
    patient_ids = np.load(args.patient_ids_file).tolist()

    evaluate_full_volumes(
        model=model,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        patient_ids=patient_ids,
        target_shape=(128, 128),
        device=args.device,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main()
